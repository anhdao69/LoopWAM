"""Causal video prefill and suffix-aligned action execution.

Slots are explicit arguments, including during checkpoint recomputation. Caches
retain autograd history in training; no detached observation shortcut is used.
"""
from __future__ import annotations
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import flash_attention, modulate, rope_apply
from .schedule import video_schedule, action_schedule, validate_budget


class LoopMoT(nn.Module):
    def __init__(self, video, action, arch='loopwam', alignment='late', gradient_checkpointing=False):
        super().__init__()
        self.mixtures = nn.ModuleDict({'video': video, 'action': action})
        self.arch, self.alignment = arch, alignment
        self.num_heads, self.attn_head_dim = video.num_heads, video.attn_head_dim
        if (action.num_heads, action.attn_head_dim) != (self.num_heads, self.attn_head_dim):
            raise ValueError('Video and action self-attention must share the head space')
        self.gradient_checkpointing = gradient_checkpointing
        self.compile_training_layers = False
        self.kv = self.ka = 4
        self.num_layers = len(video_schedule(4, arch))

    def set_budget(self, kv, ka):
        validate_budget(kv, ka)
        self.kv, self.ka = kv, ka

    @staticmethod
    def _io(block, slot, x, t_mod, freqs):
        if slot is not None:
            return block.attention_io(x, t_mod, freqs, slot)
        sm, sc, gm, sf, sfc, gf = MoT._split_modulation(block, t_mod)
        h = modulate(block.norm1(x), sm, sc)
        q = rope_apply(block.self_attn.norm_q(block.self_attn.q(h)), freqs, block.num_heads)
        k = rope_apply(block.self_attn.norm_k(block.self_attn.k(h)), freqs, block.num_heads)
        return q, k, block.self_attn.v(h), gm, sf, sfc, gf

    @staticmethod
    def _post(block, slot, x, out, gm, sf, sc, gf, context, context_mask):
        if slot is not None:
            return block.post_attention(x, out, gm, sf, sc, gf, context, context_mask, slot)
        return MoT._apply_expert_post_block_tensor(block, x, out, gm, sf, sc, gf, context, context_mask)

    def _video_layer(self, block, slot, x, freqs, t_mod, context, context_mask, mask, first):
        q,k,v,gm,sf,sc,gf = self._io(block, slot, x, t_mod, freqs)
        out = flash_attention(q,k,v,self.num_heads,ctx_mask=mask)
        y = self._post(block,slot,x,out,gm,sf,sc,gf,context,context_mask)
        return y, k[:,:first], v[:,:first]

    def _action_layer(self, block, slot, x, freqs, t_mod, context, context_mask, vk, vv):
        q,k,v,gm,sf,sc,gf = self._io(block,slot,x,t_mod,freqs)
        out = flash_attention(q,torch.cat((vk,k),dim=1),torch.cat((vv,v),dim=1),self.num_heads)
        return self._post(block,slot,x,out,gm,sf,sc,gf,context,context_mask)

    def _run(self, fn, *args):
        if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
            return checkpoint(fn,*args,use_reentrant=False)
        return fn(*args)

    def video_forward(self, x, freqs, t_mod, context, context_mask, mask, tokens_per_frame, kv=4):
        cache, exits = {}, {}
        for index, slot, key in video_schedule(kv,self.arch):
            x,k,v=self._run(self._video_layer,self.mixtures['video'].blocks[index],slot,x,freqs,t_mod,context,context_mask,mask,tokens_per_frame)
            cache[key] = (k,v)
            if key[0]=='core' and key[2]==5:
                exits[key[1]]=x
        return x,cache,exits

    def add_exit_coda(self, cache, exits, kv, freqs, t_mod, context, context_mask, tokens_per_frame):
        if self.arch != 'loopwam' or ('coda',kv,0) in cache:
            return cache
        first=tokens_per_frame
        x=exits[kv][:,:first]
        t=t_mod[:,:first] if t_mod.ndim==4 else t_mod
        cm=context_mask[:,:first]
        mask=torch.ones((first,first),device=x.device,dtype=torch.bool)
        for j in range(3):
            x,k,v=self._run(self._video_layer,self.mixtures['video'].blocks[9+j],None,x,freqs[:first],t,context,cm,mask,first)
            cache['coda',kv,j]=(k,v)
        return cache

    def action_forward(self,x,freqs,t_mod,context,context_mask,cache,kv=4,ka=4):
        for index,slot,key in action_schedule(kv,ka,self.arch,self.alignment):
            k,v=cache[key]
            x=self._run(self._action_layer,self.mixtures['action'].blocks[index],slot,x,freqs,t_mod,context,context_mask,k,v)
        return x

    def forward_joint_core(self, video_tokens,action_tokens,video_freqs,action_freqs,video_t_mod,action_t_mod,video_context,video_context_mask,action_context,action_context_mask,attention_mask):
        nv=video_tokens.shape[1]
        # Joint mask action rows expose precisely the first-frame video prefix.
        first=int(attention_mask[nv,:nv].sum().item())
        v,cache,_=self.video_forward(video_tokens,video_freqs,video_t_mod,video_context,video_context_mask,attention_mask[:nv,:nv],first,self.kv)
        a=self.action_forward(action_tokens,action_freqs,action_t_mod,action_context,action_context_mask,cache,self.kv,self.ka)
        return v,a

    def prefill_video_cache_tensor(self,video_tokens,video_freqs,video_t_mod,video_context,video_context_mask,video_attention_mask):
        # Inference contains first-frame tokens only. This interface is retained
        # for FastWAM.infer_action and one compiled graph per video budget.
        _,cache,_=self.video_forward(video_tokens,video_freqs,video_t_mod,video_context,video_context_mask,video_attention_mask,video_tokens.shape[1],self.kv)
        entries=video_schedule(self.kv,self.arch)
        return [cache[e[2]][0] for e in entries],[cache[e[2]][1] for e in entries]

    def forward_action_with_video_cache_tensor(self,action_tokens,action_freqs,action_t_mod,action_context,action_context_mask,video_cache_k,video_cache_v,action_attention_mask):
        cache={entry[2]:(k,v) for entry,k,v in zip(video_schedule(self.kv,self.arch),video_cache_k,video_cache_v)}
        return self.action_forward(action_tokens,action_freqs,action_t_mod,action_context,action_context_mask,cache,self.kv,self.ka)

    def forward(self,embeds_all,attention_mask,freqs_all,context_all,t_mod_all):
        v,a=self.forward_joint_core(embeds_all['video'],embeds_all['action'],freqs_all['video'],freqs_all['action'],t_mod_all['video'],t_mod_all['action'],context_all['video']['context'],context_all['video']['mask'],context_all['action']['context'],context_all['action']['mask'],attention_mask)
        return {'video':v,'action':a}
