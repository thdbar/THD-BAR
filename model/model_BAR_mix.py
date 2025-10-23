

import inspect
import torch
import torch.nn as nn
from torch.nn import functional as F
from model.model_GPT4BAR_mix import *
from torch.autograd import Function
from model.model_neural_transformer import NTConfig
from model.model_neural_transformer import NeuralTransformer
from model.norm_ms_quantizer import NormMSVectorQuantizer 

from collections import OrderedDict
from einops import rearrange, repeat


class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class BAR(nn.Module):
    def __init__(self,
                 GPT_config,
                 tokenizer_ckpt_path=None, 
                 init_from='gpt2',
                 n_embd=768,
                 n_quant=128,
                 eeg_vocab_size=8192,
                 ):
        super().__init__()
        print('gpt_initfrom', init_from)

        if init_from == 'scratch':
            print('gpt_initfrom', init_from)

            self.GPT2 = GPT(GPT_config)
        elif init_from.startswith('gpt2'):
            override_args = dict(dropout=0.0)
            print('gpt_initfrom', init_from)
            self.GPT2 = GPT.from_pretrained(init_from, override_args)
        self.GPT2.enlarge_wte(50304)
        self.GPT2.enlarge_lm_head(self.GPT2.config.vocab_size + eeg_vocab_size)

        self.ch_embed = nn.Embedding(142, self.GPT2.config.n_embd)
        self.time_embed = nn.Embedding(64, self.GPT2.config.n_embd)

        # task layer
        self.encode_transform_layer = nn.Sequential(
            nn.Linear(n_quant, self.GPT2.config.n_embd),
            nn.GELU(),
        ) if n_quant != self.GPT2.config.n_embd else nn.Identity()

        self.encode_transform_layer.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.01) 
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x_eeg=None, y_eeg=None, x_text=None, y_text=None, 
                input_chans=None, input_time=None, input_mask=None, 
                eeg_mask=None, eeg_text_mask=None, mschindex_BN=None, mstimeindex_BN=None):
        """
        x_eeg: shape [B, times, msN, D] 
        x_text: shape [B, N2]
        """
        if x_eeg is not None:
            x_eeg = self.encode_transform_layer(x_eeg)
            B, N, D = x_eeg.shape # N == T * msN
            x_eeg += self.ch_embed(mschindex_BN)
        logits, loss, accuracy = self.GPT2(x_eeg, y_eeg, x_text, y_text, 
                                           input_time, 
                                           eeg_mask, eeg_text_mask, 
                                           mstimeindex_BN=mstimeindex_BN)
        log = {}
        split="train" if self.training else "val"
        if loss is not None:
            log[f'{split}/loss'] = loss.item()
        if accuracy is not None:
            log[f'{split}/accuracy'] = accuracy.item()
        return loss, log, logits
    
    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.GPT2.transformer.wpe.weight.numel()
        return n_params

    @torch.no_grad()
    def generate(self, 
                 x_eeg=None, x_text=None, 
                input_chans=None, input_time=None, 
                input_mask=None, eeg_mask=None, 
                eeg_text_mask=None, mschindex_BN=None, mstimeindex_BN=None, 
                max_new_tokens=10, temperature=1.0, top_k=1):
        if x_eeg is not None:
            x_eeg = self.encode_transform_layer(x_eeg)
            B, N, D = x_eeg.shape 
            x_eeg += self.ch_embed(mschindex_BN)
        
        for _ in range(max_new_tokens):
            logits, _, _ = self.GPT2(x_eeg=x_eeg, x_text=x_text, 
                                     eeg_time_idx=input_time, 
                                     eeg_mask=eeg_mask, eeg_text_mask=eeg_text_mask, 
                                     mstimeindex_BN=mstimeindex_BN)
            logits = logits[:, -1, :50257] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits.to(torch.float32) + 1e-6, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

            x_text = torch.cat((x_text, idx_next), dim=1)
            if eeg_text_mask is not None:
                eeg_text_mask = torch.cat((eeg_text_mask, torch.zeros((eeg_text_mask.size(0), eeg_text_mask.size(1), eeg_text_mask.size(2), 1), device=eeg_text_mask.device)), dim=-1)
                eeg_text_mask = torch.cat((eeg_text_mask, torch.ones((eeg_text_mask.size(0), eeg_text_mask.size(1), 1, eeg_text_mask.size(3)), device=eeg_text_mask.device)), dim=-2)
    
        return x_text

    
    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")
        return optimizer

