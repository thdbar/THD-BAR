
import os
import time
import argparse
from contextlib import nullcontext

from einops import rearrange
import numpy as np
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.model_BAR_mix import BAR
from model.model_msvq_wo_ema import MSVQ
from model.model_neural_transformer import NTConfig
from model.model_GPT4BAR_mix import GPTConfig


from pathlib import Path
import tiktoken
from utils import prepare_TUAB_dataset4BAR, prepare_TUEV_dataset4BAR, prepare_TUSL_dataset4BAR, prepare_HMC_dataset4BAR, prepare_Workload_dataset4BAR, prepare_SEED_dataset4BAR, cosine_scheduler, get_metrics
from torch.utils.data.dataset import ConcatDataset
from collections import OrderedDict


master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None


def change_Lms2BN(msinds_LmsN, msfeats_LmsND, ms_bool, gpt_mask, input_time, num_chans, vocab_size):
    # print(ms_bool, num_chans)
    B, _, _, _ = gpt_mask.shape
    N = num_chans
    num_times = torch.max(input_time, dim=1).values + 1
    # N = num_chans
    L, msN, D = msfeats_LmsND.shape

    msinds_BN = torch.full((B, N), fill_value=-1-vocab_size).to(device, non_blocking=True)

    msfeats_BND = torch.zeros((B, N, D), dtype=msfeats_LmsND.dtype, device=msfeats_LmsND.device)  # Initialize the output tensor
    mschindex_BN = torch.zeros((B, N), dtype=torch.int64, device=msfeats_LmsND.device)  # Initialize the output tensor
    mstimeindex_BN = torch.zeros((B, N), dtype=torch.int64, device=msfeats_LmsND.device)  # Initialize the output tensor
    # remove ch
    cur_L = 0
    for i in range(B):
        msinds_slice = msinds_LmsN[cur_L+1:cur_L+num_times[i], ms_bool[i]].view(-1)  # remove all chans of first time for each sample
        msinds_BN[i, :msinds_slice.size(0)] = msinds_slice
        msfeats_slice = msfeats_LmsND[cur_L:cur_L+num_times[i], ms_bool[i]].view(-1, D)
        msfeats_BND[i, :msfeats_slice.size(0)] = msfeats_slice
        mschidex_slice = torch.nonzero(ms_bool[i]).squeeze().repeat(num_times[i])
        # import pdb; pdb.set_trace()
        mschindex_BN[i, :mschidex_slice.size(0)] = mschidex_slice
        mschidex_slice = torch.repeat_interleave(torch.arange(0, num_times[i], dtype=torch.int64, device=msfeats_LmsND.device), repeats=len(mschidex_slice)//num_times[i], dim=0)
        mstimeindex_BN[i, :mschidex_slice.size(0)] = mschidex_slice
        cur_L += num_times[i]
    return msfeats_BND, msinds_BN, mschindex_BN, mstimeindex_BN



def init(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank
    # various inits, derived attributes, I/O setup
    backend = 'nccl' # 'nccl', 'gloo', etc.
    device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        seed_offset = 0
        ddp_world_size = 1

    torch.manual_seed(args.seed + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)


def get_instruct_datasets(args, downstream_dataset: str, eeg_max_len=-1, text_max_len=-1):
        dataset_info = {'name': downstream_dataset}
        if downstream_dataset == 'SEED':
            dataset_train, dataset_test, dataset_val = prepare_SEED_dataset4BAR(Path(args.dataset_dir, 'SEED_new'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)
            print('SEED: ', len(dataset_train), len(dataset_val), len(dataset_test))

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 3
            dataset_info['result_idx'] = 11
            dataset_info['label_dic'] = {'Positive': 0, 'Neutral': 1, 'Negative': 2}
        elif downstream_dataset == 'TUAB':
            dataset_train, dataset_test, dataset_val = prepare_TUAB_dataset4BAR(Path(args.dataset_dir, 'TUAB/processed'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
            dataset_info['is_binary'] = True
            dataset_info['result_idx'] = 7
            dataset_info['label_dic'] = {'Yes': 1, 'No': 0}
        elif downstream_dataset == 'TUEV':
            dataset_train, dataset_test, dataset_val = prepare_TUEV_dataset4BAR(Path(args.dataset_dir, 'TUEV'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 6
            dataset_info['result_idx'] = 34
            dataset_info['label_dic'] = {'(A)': 0, '(B)': 1, '(C)': 2, '(D)': 3, '(E)': 4, '(F)': 5}
        elif downstream_dataset == 'TUSL':
            dataset_train, dataset_test, dataset_val = prepare_TUSL_dataset4BAR(Path(args.dataset_dir, 'TUSL'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 3
            dataset_info['result_idx'] = 17
            dataset_info['label_dic'] = {'(G)': 0, '(H)': 1, '(I)': 2}
        elif downstream_dataset == 'HMC':
            dataset_train, dataset_test, dataset_val = prepare_HMC_dataset4BAR(Path(args.dataset_dir, 'HMC'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["accuracy", "balanced_accuracy", "cohen_kappa", "f1_weighted"]
            dataset_info['is_binary'] = False
            dataset_info['num_classes'] = 5
            dataset_info['result_idx'] = 22
            dataset_info['label_dic'] = {'(J)': 0, '(K)': 1, '(L)': 2, '(M)': 3, '(N)': 4}
        elif downstream_dataset == 'Workload':
            dataset_train, dataset_test, dataset_val = prepare_Workload_dataset4BAR(Path(args.dataset_dir, 'EEGWorkload'), is_instruct=True, 
                                                                            eeg_max_len=eeg_max_len, text_max_len=text_max_len)

            dataset_info['metrics'] = ["pr_auc", "roc_auc", "accuracy", "balanced_accuracy"]
            dataset_info['is_binary'] = True
            dataset_info['result_idx'] = 9
            dataset_info['label_dic'] = {'Yes': 1, 'No': 0}

        dataset_info['dataset_train'] = dataset_train
        dataset_info['dataset_val'] = dataset_val
        dataset_info['dataset_test'] = dataset_test

        bs = args.eeg_batch_size
        if ddp:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
            )
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=bs,
                num_workers=10,
                pin_memory=True,
                drop_last=True,
            )
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
            )
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val, sampler=sampler_val,
                batch_size=int(bs * 5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
            )
            sampler_test = torch.utils.data.DistributedSampler(
                dataset_test, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
            )
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test, sampler=sampler_test,
                batch_size=int(bs * 5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
            )
        else:
            data_loader_train = torch.utils.data.DataLoader(
                dataset_train,
                batch_size=bs,
                num_workers=10,
                pin_memory=True,
                drop_last=True,
                shuffle=True
            )
            data_loader_val = torch.utils.data.DataLoader(
                dataset_val,
                batch_size=int(bs*5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
            data_loader_test = torch.utils.data.DataLoader(
                dataset_test,
                batch_size=int(bs*5),
                num_workers=10,
                pin_memory=True,
                drop_last=False,
                shuffle=False
            )
        dataset_info['data_loader_train'] = data_loader_train
        dataset_info['data_loader_val'] = data_loader_val
        dataset_info['data_loader_test'] = data_loader_test
        return dataset_info


def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/{}'.format(args.wandb_runname))
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    # text data loader
    data_dir = os.path.join(args.out_dir, 'text')
    def get_batch(split):
        # We recreate np.memmap every batch to avoid a memory leak, as per
        # https://stackoverflow.com/questions/45132940/numpy-memmap-memory-usage-want-to-iterate-once/61472122#61472122
        if split == 'train':
            data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
        else:
            data = np.memmap(os.path.join(data_dir, 'val.bin'), dtype=np.uint16, mode='r')
        ix = torch.randint(len(data) - args.block_size, (args.text_batch_size,))
        x = torch.stack([torch.from_numpy((data[i:i + args.block_size]).astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy((data[i + 1:i + 1 + args.block_size]).astype(np.int64)) for i in ix])
        if device_type == 'cuda':
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
        else:
            x, y = x.to(device), y.to(device)
        return x, y

    concat_datasets = False
    all_datasets = []
    names = ['TUAB', 'TUEV', 'SEED', 'HMC', 'Workload', 'TUSL']
    for name in names:
        all_datasets.append(get_instruct_datasets(args, name, eeg_max_len=276, text_max_len=80))
    
    if concat_datasets:
        merge_datasets = ConcatDataset([dataset_info['dataset_train'] for dataset_info in all_datasets])
        if ddp:
            sampler_merge = torch.utils.data.DistributedSampler(
                merge_datasets, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
            )
            data_loader_merge = torch.utils.data.DataLoader(
                merge_datasets, sampler=sampler_merge,
                batch_size=args.eeg_batch_size,
                num_workers=10,
                pin_memory=True,
                drop_last=True
            )
        else:
            data_loader_merge = torch.utils.data.DataLoader(
                merge_datasets,
                batch_size=args.eeg_batch_size,
                num_workers=10,
                pin_memory=True,
                drop_last=True,
                shuffle=True
            ) 
            
    # init these up here, can override if init_from='resume' (i.e. from a checkpoint
    iter_num = 0
    #=======================================================================================
    # load tokenizer
    encoder_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0., num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0., num_classes=0, in_chans=128)
    tokenizer_ckpt_path = os.path.join(args.out_dir, args.tokenizer_path)
    tokenizer_checkpoint = torch.load(tokenizer_ckpt_path, map_location=device, weights_only=False)
    tokenizer_checkpoint_model_args = tokenizer_checkpoint['encoder_args']
    # force these config attributes to be equal otherwise we can't even resume training
    # the rest of the attributes (e.g. dropout) can stay as desired from command line
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
        encoder_args[k] = tokenizer_checkpoint_model_args[k]
    tokenizer_checkpoint_model_args = tokenizer_checkpoint['decoder_args']
    for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias']:
        decoder_args[k] = tokenizer_checkpoint_model_args[k] 
    # create the model
    encoder_conf = NTConfig(**encoder_args)
    decoder_conf = NTConfig(**decoder_args)
    tokenizer = MSVQ(encoder_conf, decoder_conf)
    tokenizer_state_dict = tokenizer_checkpoint['model']
    # fix the keys of the state dictionary :(
    # honestly no idea how checkpoints sometimes get this prefix, have to debug more
    unwanted_prefix = '_orig_mod.'
    for k,v in list(tokenizer_state_dict.items()):
        if k.startswith(unwanted_prefix):
            tokenizer_state_dict[k[len(unwanted_prefix):]] = tokenizer_state_dict.pop(k)
    all_keys = list(tokenizer_state_dict.keys())
    new_dict = OrderedDict()
    for key in all_keys:
        if key.startswith('VQ.'):
            new_dict[key[3:]] = tokenizer_state_dict[key]
    print(f"Initializing MSVQ from weights: {tokenizer_ckpt_path}")

    tokenizer.load_state_dict(new_dict, strict=False)
    tokenizer.eval()
    tokenizer.to(device)
    # free up memory
    tokenizer_checkpoint = None
    #=======================================================================================

    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')):
        init_from = 'resume'
    else:
        init_from = 'pretrained'

    # model init
    n_layer = 12
    n_head = 12
    n_embd = 768

    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=args.block_size,
                    bias=bias, vocab_size=50257, dropout=dropout) # start with model_args from command line
    if init_from == 'resume':
        print(f"Resuming training from {args.out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        print(model_args)
        gptconf = GPTConfig(**model_args)
        model = BAR(gptconf, init_from='scratch')
        state_dict = checkpoint['model']

        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
    elif init_from == 'gpt':
        print(f"Initializing from tokenizer weights: {init_from}")
        # initialize from EEGPT weights
        model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=args.block_size,
                        bias=bias, vocab_size=50257, dropout=dropout) # start with model_args from command line
        # create the model
        gptconf = GPTConfig(**model_args)
        model = BAR(gptconf, tokenizer_ckpt_path, init_from='gpt2')
        start_epoch = 0
    elif init_from == 'pretrained':
        print(f"Initializing training from {args.BAR_path}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(args.out_dir, args.BAR_path)
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        checkpoint_model_args = checkpoint['model_args']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = checkpoint_model_args[k]
        # create the model
        gptconf = GPTConfig(**model_args)
        model = BAR(gptconf, init_from='scratch')
        state_dict = checkpoint['model']
        # fix the keys of the state dictionary :(
        # honestly no idea how checkpoints sometimes get this prefix, have to debug more
        unwanted_prefix = '_orig_mod.'
        for k,v in list(state_dict.items()):
            if k.startswith(unwanted_prefix):
                state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
        model.load_state_dict(state_dict)
        start_epoch = 0

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None 

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])
        model._set_static_graph()

    # logging
    if args.wandb_log and master_process:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_key 
        if init_from == 'resume':
            wandb.init(project=args.wandb_project, entity=args.entity, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'), resume=True)
        else:
            wandb.init(project=args.wandb_project, entity=args.entity, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'))

    num_training_steps_per_epoch=0
    for i, name in enumerate(names):
        dataset = all_datasets[i]['dataset_train']
        bs = args.eeg_batch_size
        
        per_gpu_samples = len(dataset) // ddp_world_size
        steps = per_gpu_samples // bs
        num_training_steps_per_epoch += steps

    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs, warmup_steps=int(args.warmup_ratio * num_training_steps_per_epoch * args.epochs)
    )                                 

    enc = tiktoken.get_encoding("gpt2")
    decode = lambda l: enc.decode(l)
    
    # training loop
    datasets = [{'data_loader_train': data_loader_merge}] if concat_datasets else all_datasets

    X_text2, Y_text2 = get_batch('train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    if args.eval_only:
        start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        for dataset_info in datasets:
            if args.eval_only:
                break
            for step, (batch) in enumerate(dataset_info['data_loader_train']):
                # determine and set the learning rate for this iteration
                lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
                for param_group in optimizer.param_groups:
                    param_group['lr'] = lr

                X_eeg, X_text, Y_text, input_chans, input_time, input_mask, gpt_mask, ms_bool, len_eeg_gpt_mask = batch

                X_eeg = X_eeg.float().to(device, non_blocking=True)
                X_text = X_text.to(device, non_blocking=True)
                Y_text = Y_text.to(device, non_blocking=True)
                input_chans = input_chans.to(device, non_blocking=True)
                input_time = input_time.to(device, non_blocking=True)
                input_mask = input_mask.to(device, non_blocking=True)
                gpt_mask = gpt_mask.to(device, non_blocking=True)
                ms_bool = ms_bool.to(device, non_blocking=True)
                with torch.no_grad():
                    with ctx:
                        msinds_LmsN, msfeats_LmsND = tokenizer.get_codebook_msinds_and_msfeats(X_eeg, input_chans, input_time, input_mask)
                        X_eeg, Y_eeg, mschindex_BN, mstimeindex_BN = change_Lms2BN(msinds_LmsN, msfeats_LmsND, ms_bool, gpt_mask, input_time, len_eeg_gpt_mask[0], raw_model.GPT2.config.vocab_size)
                
                X_eeg, Y_eeg = X_eeg.detach(), Y_eeg.detach()
                if input_mask is not None:
                    input_mask = input_mask.to(device, non_blocking=True)

                Y_eeg = torch.full((Y_eeg.size(0), Y_eeg.size(1)), fill_value=-1-raw_model.GPT2.config.vocab_size).to(device, non_blocking=True)

                # forward backward update, with optional gradient accumulation to simulate larger batch size
                # and using the GradScaler if data type is float16
                if ddp:
                    # in DDP training we only need to sync gradients at the last micro step.
                    # the official way to do this is with model.no_sync() context manager, but
                    # I really dislike that this bloats the code and forces us to repeat code
                    # looking at the source of that context manager, it just toggles this variable
                    model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0

                with ctx:
                    loss1, log1, logits = model(X_eeg, Y_eeg, X_text, Y_text, 
                                                input_chans, input_time, 
                                                input_mask, eeg_text_mask=gpt_mask, 
                                                mschindex_BN=mschindex_BN, mstimeindex_BN=mstimeindex_BN)
                    loss2, log2, _ = model(None, None, X_text2, Y_text2)
                    
                    model.train()

                    loss = (loss1 + loss2) / args.gradient_accumulation_steps # scale the loss to account for gradient accumulation
                # immediately async prefetch next batch while model is doing the forward pass on the GPU
                # backward pass, with gradient scaling if training in fp16
                
                scaler.scale(loss).backward()

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # clip the gradient
                    if args.grad_clip != 0.0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    # step the optimizer and scaler if training in fp16
                    scaler.step(optimizer)
                    scaler.update()
                    # flush the gradients as soon as we can, no need for this memory anymore
                    optimizer.zero_grad(set_to_none=True)
                
                X_text2, Y_text2 = get_batch('train')

                # evaluate the loss on train/val sets and write checkpoints
                if (iter_num + 1) % args.log_interval == 0 and master_process:
                    print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: train total loss {log1['train/loss'] + log2['train/loss']:.4f}, instruction loss {log1['train/loss']:.4f}, text loss {log2['train/loss']:.4f}")
                    if args.wandb_log:
                        log = {
                            "train/total_loss": log1['train/loss']  + log2['train/loss'] ,
                            "train/instruction_loss": log1['train/loss'],
                            "train/text_loss": log2['train/loss'],
                            "train/instruction_accuracy": log1['train/accuracy'],
                            "train/text_accuracy": log2['train/accuracy'],
                            "lr": lr
                        }
                        wandb.log(log)

                if iter_num == 0 and args.eval_only:
                    break

                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                iter_num += 1
                local_iter_num += 1
        
        if master_process and (not args.eval_only):
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_args': model_args,
                'iter_num': iter_num,
                'epoch': epoch
            }
            print(f"saving checkpoint to {checkpoint_out_dir}")
            torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt.pt'))
            if (epoch + 1) % args.save_ckpt_freq == 0:
                print(f"saving checkpoint to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))
        
        # validation and test
        for dataset_info in all_datasets:
            print('Dataset:', dataset_info['name'])
            results_val = evaluate(raw_model, tokenizer, dataset_info, dataset_info['data_loader_val'], decode, raw_model.GPT2.config.vocab_size)
            print('=' * 10)
            print('Eval:')
            for metric in results_val.keys():
                print(metric + ':', results_val[metric])
            results_test = evaluate(raw_model, tokenizer, dataset_info, dataset_info['data_loader_test'], decode, raw_model.GPT2.config.vocab_size)
            print('=' * 10)
            print('Test:')
            for metric in results_test.keys():
                print(metric + ':', results_test[metric])
            print('=' * 10)
            if args.wandb_log and master_process:
                log = {}
                for metric in results_val.keys():
                    log['val_' + dataset_info['name'] + '/' + metric] = results_val[metric]
                    log[f'test_' + dataset_info['name'] + '/' + metric] = results_test[metric]
                wandb.log(log)
        if args.eval_only:
            break

    if ddp:
        destroy_process_group()


def get_pred(pred_string, dataset_info):
    if dataset_info['name'] == 'zuco':
        pred = pred_string[17:].split('<|endoftext|>')[0]
    else:
        pred = -1
        try:
            pred = pred_string.split(' ')[dataset_info['result_idx']]
            if pred.startswith('('):
                pred = pred[:3]
            pred = dataset_info['label_dic'][pred]
        except:
            pred = -1
    return pred

@torch.no_grad()
def evaluate(model, tokenizer, dataset_info, dataloader, decode, vocab_size):
    model.eval()
    preds = []
    targets = []
    for eval_step, (batch) in enumerate(dataloader):
        print('eval_step', eval_step)
        X_eeg, X_text, label, input_chans, input_time, input_mask, gpt_mask, ms_bool, len_eeg_gpt_mask = batch

        X_eeg = X_eeg.float().to(device, non_blocking=True)
        X_text = X_text.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True)
        input_chans = input_chans.to(device, non_blocking=True)
        input_time = input_time.to(device, non_blocking=True)
        input_mask = input_mask.to(device, non_blocking=True)
        gpt_mask = gpt_mask.to(device, non_blocking=True)
        ms_bool = ms_bool.to(device, non_blocking=True)

        with torch.no_grad():
            with ctx:
                msinds_LmsN, msfeats_LmsND = tokenizer.get_codebook_msinds_and_msfeats(X_eeg, input_chans, input_time, input_mask)
                X_eeg, Y_eeg, mschindex_BN, mstimeindex_BN = change_Lms2BN(msinds_LmsN, msfeats_LmsND, ms_bool, gpt_mask, input_time, len_eeg_gpt_mask[0], vocab_size)
        if input_mask is not None:
            input_mask = input_mask.to(device, non_blocking=True)

        with ctx:
            text = model.generate(X_eeg, X_text, 
                                  input_chans, input_time, 
                                  input_mask, eeg_text_mask=gpt_mask, 
                                  mschindex_BN=mschindex_BN, mstimeindex_BN=mstimeindex_BN, 
                                  max_new_tokens=5)
            text = text[:, 1:] # remove [SEP] token
            for i, t in enumerate(text):
                pred_string = decode(t.tolist())
                pred = get_pred(pred_string, dataset_info)
                if not dataset_info['is_binary']:
                    pred = np.eye(dataset_info['num_classes'])[pred]
                preds.append(pred)
            targets.append(label)
    
    model.train()

    targets = torch.cat(targets, dim=0).cpu().numpy()
    preds = np.array(preds)
    results = get_metrics(preds, targets, dataset_info['metrics'], dataset_info['is_binary'])

    return results


def get_args():
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--out_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--dataset_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--tokenizer_path', default='checkpoints/VQ.py', help='path where tokenizer is')
    parser.add_argument('--BAR_path', default='checkpoints/BAR-B.pt', help='path where BAR model is')
    parser.add_argument('--log_interval', default=10, type=int)
    parser.add_argument('--eval_only', default=False, action='store_true')
    # wandb
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='BAR')
    parser.add_argument('--entity', default=None, help='name of team')
    parser.add_argument('--wandb_runname', default='instruction-B')
    parser.add_argument('--wandb_api_key', type=str)
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--eeg_batch_size', default=16, type=int)  # 64
    parser.add_argument('--text_batch_size', default=4, type=int)  # 16
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--warmup_epochs', default=1, type=int)
    parser.add_argument('--warmup_ratio', type=float, default=0.1)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--block_size', default=1024, type=int)

    parser.add_argument('--learning_rate', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')
    parser.add_argument('--min_lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=1e-1,
                        help='weight decay (default: 1e-1)')
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.95)
    parser.add_argument('--grad_clip', type=float, default=1.0,
                        help='clip gradients at this value, or disable if == 0.0')
    parser.add_argument('--decay_lr', default=True, action='store_false')
    parser.add_argument('--seed', default=1337, type=int)

    parser.add_argument('--compile', default=False, action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    main(args)
