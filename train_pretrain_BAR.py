import os
import time
import math
import argparse
from contextlib import nullcontext

import numpy as np
import torch
from einops import rearrange, repeat

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from model.model_BAR_mix import BAR
from model.model_msvq_wo_ema import MSVQ
from model.model_neural_transformer import NTConfig
from model.model_GPT4BAR_mix import GPTConfig

from dataset4BAR_mix import PickleLoader, collate_fn
from pathlib import Path
from utils import cosine_scheduler
from model.standard_1020_chorder import remove_unused_ch
from collections import OrderedDict

master_process = None; device = None; dtype = None
ctx = None; ddp_rank = None; device_type = None
ddp = None; ddp_world_size = None; ddp_local_rank = None


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


def change_Lms2BN(msinds_LmsN, msfeats_LmsND, ms_bool, gpt_mask, input_time, num_chans, vocab_size):
    B, _, N, N = gpt_mask.shape
    L, msN, D = msfeats_LmsND.shape

    num_chan_mix = torch.min(num_chans)

    msinds_BN = torch.full((B, N), fill_value=-1-vocab_size).to(device, non_blocking=True)

    msfeats_BND = torch.zeros((B, N, D), dtype=msfeats_LmsND.dtype, device=msfeats_LmsND.device)  # Initialize the output tensor
    mschindex_BN = torch.zeros((B, N), dtype=torch.int64, device=msfeats_LmsND.device)  # Initialize the output tensor
    mstimeindex_BN = torch.zeros((B, N), dtype=torch.int64, device=msfeats_LmsND.device)  # Initialize the output tensor

    num_times = torch.max(input_time, dim=1).values + 1
    
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



def get_batch(args, split):
    # text data loader
    data_dir = os.path.join(args.out_dir, 'text')
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
    
def main(args):
    global ctx, master_process, ddp, ddp_world_size, ddp_rank, device, dtype, device_type, ddp_local_rank

    init(args)

    checkpoint_out_dir = os.path.join(args.out_dir, 'checkpoints/{}'.format(args.wandb_runname))
    if master_process:
        os.makedirs(checkpoint_out_dir, exist_ok=True)

    train_files = Path(args.dataset_dir, 'train_regroup_robustscale_clip100').rglob('*.pkl')
    train_files = [file for file in train_files]
    dataset_train = PickleLoader(train_files, GPT_training=True)
    val_files = Path(args.dataset_dir, 'val').rglob('*.pkl')
    val_files = [file for file in val_files]
    dataset_val = PickleLoader(val_files, GPT_training=True)

    if ddp:
        sampler_train = torch.utils.data.DistributedSampler(
            dataset_train, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=True
        )
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=args.eeg_batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate_fn
        )
        sampler_val = torch.utils.data.DistributedSampler(
            dataset_val, num_replicas=ddp_world_size, rank=ddp_rank, shuffle=False
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=int(1.5 * args.eeg_batch_size),
            num_workers=10,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn
        )
    else:
        data_loader_train = torch.utils.data.DataLoader(
            dataset_train,
            batch_size=args.eeg_batch_size,
            num_workers=10,
            pin_memory=True,
            drop_last=True,
            shuffle=True
        )
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val,
            batch_size=int(1.5 * args.eeg_batch_size),
            num_workers=10,
            pin_memory=True,
            drop_last=False,
            shuffle=False
        )

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)

    # load tokenizer
    encoder_args = dict(n_layer=12, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0.1, num_classes=0, in_chans=1, out_chans=16)
    decoder_args = dict(n_layer=4, n_head=12, n_embd=768, block_size=1024,
                    bias=False, dropout=0.1, num_classes=0, in_chans=128)

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
    tokenizer.load_state_dict(new_dict, strict=False)
    tokenizer.eval()
    tokenizer.to(device)
    # free up memory
    tokenizer_checkpoint = None

    if os.path.exists(os.path.join(checkpoint_out_dir, 'ckpt.pt')):
        init_from = 'resume'
    else:
        init_from = 'gpt2'

    iter_num = 0
    n_layer = 24
    n_head = 16
    n_embd = 1024

    dropout = 0.1 # for pretraining 0 is good, for finetuning try 0.1+
    bias = False # do we use bias inside LayerNorm and Linear layers?
    model_args = dict(n_layer=n_layer, n_head=n_head, n_embd=n_embd, block_size=args.block_size,
                    bias=bias, vocab_size=50257, dropout=dropout) # start with model_args from command line
    if init_from == 'scratch':
        # init a new model from scratch
        print("Initializing a new model from scratch")
        # determine the vocab size we'll use for from-scratch training
        gptconf = GPTConfig(**model_args)
        model = BAR(gptconf, init_from=init_from)
        start_epoch = 0
    elif init_from == 'resume':
        print(f"Resuming training from {checkpoint_out_dir}")
        # resume training from a checkpoint.
        ckpt_path = os.path.join(checkpoint_out_dir, 'ckpt.pt')
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
        iter_num = checkpoint['iter_num']
        start_epoch = checkpoint['epoch'] + 1
    elif init_from.startswith('gpt2'):
        print(f"Initializing from OpenAI GPT-2 weights: {init_from}")
        # initialize from OpenAI GPT-2 weights
        gptconf = GPTConfig(**model_args)
        model = BAR(gptconf, tokenizer_ckpt_path, init_from=init_from)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']:
            model_args[k] = getattr(model.GPT2.config, k)
        start_epoch = 0
    print('Number parameters:', model.get_num_params())

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.amp.GradScaler('cuda', enabled=(dtype == 'float16'))

    # optimizer
    optimizer = model.configure_optimizers(args.weight_decay, args.learning_rate, (args.beta1, args.beta2), device_type)
    if init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
    checkpoint = None # free up memory

    # compile the model
    if compile:
        print("compiling the model... (takes a ~minute)")
        unoptimized_model = model
        model = torch.compile(model) # requires PyTorch 2.0

    # wrap model into DDP container
    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank], find_unused_parameters=True)
        model._set_static_graph()

    # logging
    if args.wandb_log and master_process:
        import wandb
        os.environ["WANDB_API_KEY"] = args.wandb_api_key 
        if init_from == 'resume':
            wandb.init(project=args.wandb_project, entity=args.entity, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'), resume=True)
        else:
            wandb.init(project=args.wandb_project, entity=args.entity, name=args.wandb_runname, dir=os.path.join(args.out_dir, 'wandb'))

    num_training_steps_per_epoch = len(dataset_train) // args.eeg_batch_size // ddp_world_size
    lr_schedule_values = cosine_scheduler(
        args.learning_rate, args.min_lr, args.epochs, num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs
    )

    # training loop
    X_text, Y_text = get_batch(args, split='train') # fetch the very first batch
    t0 = time.time()
    local_iter_num = 0 # number of iterations in the lifetime of this process
    raw_model = model.module if ddp else model # unwrap DDP container if needed
    for epoch in range(start_epoch, args.epochs):
        for step, (batch) in enumerate(data_loader_train):
            # determine and set the learning rate for this iteration
            lr = lr_schedule_values[iter_num] if args.decay_lr else args.learning_rate
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            X_eeg, input_chans, input_time, input_mask, gpt_mask, num_chans, num_tokens, ms_bool = batch
            X_eeg = X_eeg.float().to(device, non_blocking=True)
            assert not torch.isnan(X_eeg).any() and not torch.isinf(X_eeg).any(), "输入数据包含NaN/INF！"
            input_chans = input_chans.to(device, non_blocking=True)
            input_time = input_time.to(device, non_blocking=True)
            input_mask = input_mask.to(device, non_blocking=True)
            gpt_mask = gpt_mask.to(device, non_blocking=True)
            ms_bool = ms_bool.to(device, non_blocking=True)
            with torch.no_grad():
                with ctx:
                    msinds_LmsN, msfeats_LmsND = tokenizer.get_codebook_msinds_and_msfeats(X_eeg, input_chans, input_time, input_mask)
                    X_eeg, Y_eeg, mschindex_BN, mstimeindex_BN = change_Lms2BN(msinds_LmsN, msfeats_LmsND, ms_bool, gpt_mask, input_time, num_chans, raw_model.GPT2.config.vocab_size)
            X_eeg, Y_eeg = X_eeg.detach(), Y_eeg.detach()
            if ddp:
                model.require_backward_grad_sync = (step + 1) % args.gradient_accumulation_steps == 0
            torch.autograd.set_detect_anomaly(True)
            with ctx:
                # eeg
                loss1, log1, _ = model(X_eeg, Y_eeg, None, None, 
                                        input_chans, input_time, input_mask, eeg_mask=gpt_mask, 
                                        mschindex_BN=mschindex_BN, mstimeindex_BN=mstimeindex_BN)
                # text
                loss2, log2, _ = model(None, None, X_text, Y_text)
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
            X_text, Y_text = get_batch(args, split='train')
            # evaluate the loss on train/val sets and write checkpoints
            if (iter_num + 1) % args.log_interval == 0 and master_process:
                print(f"epoch {epoch} step [{step + 1}/{num_training_steps_per_epoch}]: train total loss {log1['train/loss'] + log2['train/loss']:.4f}, eeg loss {log1['train/loss']:.4f}, text loss {log2['train/loss']:.4f}")
                if args.wandb_log:
                    wandb.log({
                        "iter": iter_num,
                        "train/total_loss": log1['train/loss'] + log2['train/loss'],
                        "train/eeg_loss": log1['train/loss'],
                        "train/text_loss": log2['train/loss'],
                        "train/eeg_accuracy": log1['train/accuracy'],
                        "train/text_accuracy": log2['train/accuracy'],
                        "lr": lr,
                        "epoch": epoch,
                    })
            # timing and logging
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            iter_num += 1
            local_iter_num += 1
        
        # valiation
        loss, accuracy = evaluate(model, raw_model, tokenizer, data_loader_val)
        if master_process:
            print('='* 10)
            print(f"Evaluate : loss {loss:.4f}, accuracy {accuracy:.4f}, perplexity {math.exp(loss):.4f}")
            print('='* 10)
            if args.wandb_log:
                wandb.log({
                            "val/eeg_loss": loss,
                            "val/eeg_accuracy": accuracy,
                            'val/perplexity': math.exp(loss),
                        })
        
        if master_process:
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
                print(f"saving checkpoint {epoch} to {checkpoint_out_dir}")
                torch.save(checkpoint, os.path.join(checkpoint_out_dir, f'ckpt-{epoch}.pt'))
    
    if ddp:
        destroy_process_group()


@torch.no_grad()
def evaluate(model, raw_model, tokenizer, dataloader):
    model.eval()
    loss = []
    acc = []
    for _, (batch) in enumerate(dataloader):
        X_eeg, input_chans, input_time, input_mask, gpt_mask, num_chans, num_tokens = batch
        X_eeg = X_eeg.float().to(device, non_blocking=True)
        input_chans = input_chans.to(device, non_blocking=True)
        input_time = input_time.to(device, non_blocking=True)
        input_mask = input_mask.to(device, non_blocking=True)
        gpt_mask = gpt_mask.to(device, non_blocking=True)

        with ctx:
            Y_eeg = torch.full((X_eeg.size(0), X_eeg.size(1)), fill_value=-1-raw_model.GPT2.config.vocab_size).to(device, non_blocking=True)
            codebook_indices = tokenizer.get_codebook_indices(X_eeg, input_chans, input_time, input_mask)
            for i, (num_chan, num_token) in enumerate(zip(num_chans, num_tokens)):
                Y_eeg[i, :num_token - num_chan] = codebook_indices[i, num_chan:num_token]

        with ctx:
            _, log, _ = model(X_eeg, Y_eeg, None, None, input_chans, input_time, input_mask, eeg_mask=gpt_mask)
        
        loss.append(log['val/loss'])
        acc.append(log['val/accuracy'])

    model.train()
    
    return np.mean(loss), np.mean(acc)


def get_args():
    parser = argparse.ArgumentParser('VQ training script', add_help=False)
    parser.add_argument('--out_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--dataset_dir', default='./', help='path where to save, empty for no saving')
    parser.add_argument('--tokenizer_path', default='checkpoints/VQ.pt', help='path where tokenizer is')
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--wandb_log', default=False, action='store_true')
    parser.add_argument('--wandb_project', default='BAR')
    parser.add_argument('--entity', default=None, help='name of team')
    parser.add_argument('--wandb_runname', default='pretrain')
    parser.add_argument('--wandb_api_key', type=str)
    # training args
    parser.add_argument('--gradient_accumulation_steps', default=1, type=int)
    parser.add_argument('--eeg_batch_size', default=15, type=int)  # 60
    parser.add_argument('--text_batch_size', default=1, type=int)  # 4
    parser.add_argument('--epochs', default=20, type=int)
    parser.add_argument('--warmup_epochs', default=2, type=int)
    parser.add_argument('--save_ckpt_freq', default=1, type=int)
    parser.add_argument('--block_size', default=1024, type=int)

    parser.add_argument('--learning_rate', type=float, default=1e-4, metavar='LR',
                        help='learning rate (default: 1e-4)')  # 6e-4
    parser.add_argument('--min_lr', type=float, default=1e-6)  # 6e-5
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
