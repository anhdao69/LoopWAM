"""Fixed-noise open-loop validation on an immutable, twenty-demonstration panel."""
from __future__ import annotations
from contextlib import contextmanager, nullcontext
import json
import random
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import default_collate
from .model import load_teacher
from .losses import action_loss


@torch.no_grad()
def lora_norm_ratios(module):
    """Exact ||BA||F / ||W||F per slot, using rank-sized Gram matrices.

    trace((B.T B)(A A.T)) equals ||BA||F squared, so the full residual
    matrices are never allocated. Norms describe the caller's current weights,
    including EMA when the training callback is inside its EMA context.
    """
    from .slots import SlotLinear
    records = []
    for name, layer in module.named_modules():
        if not isinstance(layer, SlotLinear):
            continue
        with torch.autocast(layer.weight.device.type, enabled=False):
            shared = torch.linalg.vector_norm(layer.weight.detach().float())
            a, b = layer.lora_a.detach().float(), layer.lora_b.detach().float()
            gram_a, gram_b = a @ a.mT, b.mT @ b
            adapter = (gram_a * gram_b).sum(dim=(-1, -2)).clamp_min(0).sqrt()
            values = torch.cat((shared.reshape(1), adapter)).cpu().tolist()
        if not all(np.isfinite(value) for value in values):
            raise ValueError(f'Nonfinite LoRA diagnostic for {name}')
        for slot, norm in enumerate(values[1:]):
            records.append(dict(module=name, slot=slot, shared_frobenius=values[0],
                                adapter_frobenius=norm, ratio=norm / values[0] if values[0] else None,
                                status='ok' if values[0] else 'zero_shared_norm'))
    return records


def cross_loop_cka(features):
    """Centered linear CKA across clip-level mean features [loop, clip, width].

    Constant or single-clip representations have undefined CKA; those entries
    are explicitly null and identified in valid_loops, never fabricated zeros.
    """
    if features.ndim != 3:
        raise ValueError('CKA requires features with shape [loop, clip, width]')
    features = features.detach().to(device='cpu', dtype=torch.float64)
    if not torch.isfinite(features).all():
        raise ValueError('CKA features contain nonfinite values')
    centered = features - features.mean(dim=1, keepdim=True)
    grams = centered @ centered.mT
    norms = grams.flatten(1).norm(dim=1)
    valid = norms > 0
    normalized = grams.flatten(1) / norms.clamp_min(torch.finfo(norms.dtype).tiny)[:, None]
    result = (normalized @ normalized.mT).clamp(0, 1)
    matrix = [[float(result[i, j]) if valid[i] and valid[j] else None
               for j in range(len(valid))] for i in range(len(valid))]
    return dict(matrix=matrix, valid_loops=valid.tolist())


@contextmanager
def capture_loop_dynamics(mot):
    """Observe the first full (4,4) pass and ignore later shallow action passes.

    Only pooled CPU features and scalars survive a layer call. The previous
    exit is held by reference for the next loop's update norm; it is not cloned.
    Scoped method wrappers are restored even if the forward raises.
    """
    if mot.arch != 'loopwam':
        raise ValueError('Loop dynamics require the loopwam architecture')
    traces = {'video': {}, 'action': {}}
    previous = {}
    originals = {name: (getattr(mot, name), name in mot.__dict__)
                 for name in ('_video_layer', '_action_layer')}

    def observe(kind, block, slot, tokens):
        if len(traces[kind]) == 4:
            return
        if tokens.shape[0] != 1:
            raise ValueError('Fixed-panel loop diagnostics require one clip per pass')
        blocks = mot.mixtures[kind].blocks
        if block is blocks[2] and kind not in previous:
            previous[kind] = tokens.detach()
        elif block is blocks[8] and slot is not None:
            loop = slot + 1
            if loop in traces[kind]:
                return
            if loop != len(traces[kind]) + 1 or kind not in previous:
                raise ValueError('Loop diagnostics must capture the full (4,4) configuration first')
            value = tokens.detach().float()
            delta = value - previous[kind].float()
            state_l2, update_l2 = value.norm(), delta.norm()
            scalars = torch.stack((state_l2, update_l2, state_l2 / value.numel() ** .5,
                                   update_l2 / value.numel() ** .5))
            compact = torch.cat((scalars, value.mean(dim=1)[0])).cpu()
            if not torch.isfinite(compact).all():
                raise ValueError(f'Nonfinite {kind} loop {loop} diagnostic')
            row = dict(zip(('state_l2', 'update_l2', 'state_rms', 'update_rms'), compact[:4].tolist()))
            row.update(features=compact[4:], tokens=tokens.shape[1])
            traces[kind][loop] = row
            previous[kind] = tokens.detach()

    def video_layer(block, slot, *args, **kwargs):
        result = originals['_video_layer'][0](block, slot, *args, **kwargs)
        observe('video', block, slot, result[0])
        return result

    def action_layer(block, slot, *args, **kwargs):
        result = originals['_action_layer'][0](block, slot, *args, **kwargs)
        observe('action', block, slot, result)
        return result

    mot._video_layer, mot._action_layer = video_layer, action_layer
    try:
        yield traces
        if any(set(rows) != {1, 2, 3, 4} for rows in traces.values()):
            raise ValueError('Loop diagnostics did not observe four complete loops in both streams')
    finally:
        for name, (original, had_override) in originals.items():
            if had_override:
                setattr(mot, name, original)
            else:
                delattr(mot, name)


def _summarize_dynamics(traces, windows):
    if len(traces) != len(windows) or not traces:
        raise ValueError('Loop diagnostic samples do not match the fixed panel')
    record = dict(configuration=[4, 4], tau=.5, clips=len(traces), window_ids=list(windows),
                  scope='full-budget pass only; later shallow configurations are excluded',
                  norms='mean across clips of the Frobenius norm and RMS over all token/hidden elements',
                  updates='exit minus preceding loop exit; loop 1 uses the prelude output',
                  cka='centered linear CKA across clips after mean pooling all video or action tokens; null means constant features')
    for kind in ('video', 'action'):
        loops = []
        for loop in range(1, 5):
            rows = [trace[kind][loop] for trace in traces]
            summary = {key: sum(row[key] for row in rows) / len(rows)
                       for key in ('state_l2', 'update_l2', 'state_rms', 'update_rms')}
            summary.update(loop=loop, slot=loop - 1, tokens=rows[0]['tokens'])
            loops.append(summary)
        features = torch.stack([torch.stack([trace[kind][loop]['features'] for trace in traces])
                                for loop in range(1, 5)])
        record[kind] = dict(loops=loops, cross_loop_cka=cross_loop_cka(features))
    return record


@contextmanager
def _preserve_rng(device):
    device = torch.device(device)
    devices = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == 'cuda' else []
    python_state, numpy_state = random.getstate(), np.random.get_state()
    try:
        with torch.random.fork_rng(devices=devices):
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


def panel_indices(dataset):
    """One midpoint unpadded clip per held-out demo, stable across all runs.

    This is a diagnostic panel, not a claim to cover all 5,438 validation
    windows. Saving original IDs makes every measured clip reproducible.
    """
    positions={window:index for index,window in enumerate(dataset.indices)}
    selected=[]
    for episode in dataset.manifest['episodes']:
        if episode['split']=='validation':
            if episode['unpadded_windows']<1:
                raise ValueError('Validation demonstrations must contain an unpadded clip')
            window=episode['window_start']+(episode['unpadded_windows']-1)//2
            selected.append(positions[window])
    if len(selected)!=20:
        raise ValueError('Expected exactly twenty held-out demonstrations')
    return selected


def normalize_diagnostic_pairs(pairs, arch='loopwam'):
    """Validate explicit budgets and put the full pass first for loop capture."""
    result=[]
    for pair in pairs:
        if isinstance(pair,str):
            fields=pair.split(',')
            if len(fields)!=2:
                raise ValueError('Diagnostic pairs must use KV,KA tokens, for example 4,4 2,2 1,1')
            try: pair=tuple(int(value) for value in fields)
            except ValueError as error: raise ValueError('Diagnostic budgets must be integers') from error
        if (len(pair)!=2 or any(not isinstance(value,int) or isinstance(value,bool) for value in pair)
                or not 1<=pair[1]<=pair[0]<=4):
            raise ValueError(f'Invalid diagnostic budget: {pair!r}; require 1 <= KA <= KV <= 4')
        pair=tuple(pair)
        if pair in result: raise ValueError(f'Duplicate diagnostic budget: {pair}')
        result.append(pair)
    if not result: raise ValueError('Explicit diagnostic pairs cannot be empty')
    if arch!='loopwam' and any(pair!=(4,4) for pair in result):
        raise ValueError('Untied controls support only diagnostic budget (4,4)')
    return ((4,4),)+tuple(pair for pair in result if pair!=(4,4))


def diagnostic_pairs(model,step,pairs=None):
    if pairs is not None: return normalize_diagnostic_pairs(pairs,model.meta['arch'])
    if model.meta['arch']!='loopwam' or model.mode=='fixed': return ((4,4),)
    if model.mode=='konly': return ((4,4),(4,2),(4,1))
    if model.mode=='coupled' or (model.mode=='three_stage' and step<14000): return ((4,4),(2,2),(1,1))
    return ((4,4),(4,2),(4,1),(2,2),(1,1))


@torch.no_grad()
def _sample_actions(model,first_frame,context,mask,pair,seed=1234):
    kv,ka=pair
    vp=model.video_expert.prepare(first_frame,torch.zeros(1,device=first_frame.device,dtype=first_frame.dtype),context,mask,action=None,fuse_vae_embedding_in_latents=True)
    x,_,tm,c,cm,f,_,_,_,first=vp
    vmask=torch.ones((first,first),device=x.device,dtype=torch.bool)
    _,cache,_=model.mot.video_forward(x,f,tm,c,cm,vmask,first,kv)
    generator=torch.Generator(device=first_frame.device).manual_seed(seed)
    a=torch.randn((1,32,model.action_expert.action_dim),generator=generator,device=first_frame.device,dtype=torch.float32)
    timesteps,deltas=model.infer_action_scheduler.build_inference_schedule(10,first_frame.device,a.dtype,5.)
    for t,delta in zip(timesteps,deltas):
        ax,_,am,ac,acm,af=model.action_expert.prepare(a,t.reshape(1),context,mask)
        velocity=model.action_expert.post(model.mot.action_forward(ax,af,am,ac,acm,cache,kv,ka))
        a=model.infer_action_scheduler.step(velocity,delta,a)
    return a


@torch.no_grad()
def run_open_loop(model,validation_dataset,output_dir,global_step,teacher_checkpoint,seed=1234,pairs=None):
    """OL1 velocity MSE, OL2 first-ten L1, OL3 future-video velocity MSE.

    Called on rank zero between optimizer steps under the caller's selected raw
    or EMA weights. This routine itself does not swap weights or step anything.
    """
    start=time.monotonic()
    modes=[(module,module.training) for module in model.modules()]
    teacher=model.teacher
    temporary=teacher is None
    pairs=diagnostic_pairs(model,global_step,pairs)
    totals={f'{v}_{a}':{'ol1':0.,'ol2':0.,'ol3':0.} for v,a in pairs}
    windows=[]
    loop_traces=[]
    try:
        # Include temporary teacher construction and dataset processing: both may
        # consume random numbers even though network evaluation itself is frozen.
        with _preserve_rng(model.device):
            model.eval()
            if temporary:
                teacher=load_teacher(teacher_checkpoint,model.device,model.vae)
            with torch.autocast(torch.device(model.device).type,dtype=torch.bfloat16):
                for panel_id,index in enumerate(panel_indices(validation_dataset)):
                    sample=default_collate([validation_dataset[index]])
                    windows.append(int(sample['window_id'][0]))
                    clean,base,mask,proprio,action=model._inputs(sample)
                    context,cm=model._append_proprio_to_context(base,mask,proprio)
                    tc,tcm=teacher._append_proprio_to_context(base,mask,proprio)
                    generator=torch.Generator(device=model.device).manual_seed(seed+panel_id)
                    nv=torch.randn(clean.shape,generator=generator,device=model.device,dtype=clean.dtype)
                    na=torch.randn(action.shape,generator=generator,device=model.device,dtype=action.dtype)
                    apad=sample['action_is_pad'].to(model.device)
                    vpad=sample['image_is_pad'].to(model.device)
                    for tau in (.1,.3,.5,.7,.9):
                        t=torch.tensor([tau*1000],device=model.device,dtype=torch.float32)
                        x=model.train_video_scheduler.add_noise(clean,nv,t);x[:,:,0:1]=clean[:,:,0:1]
                        a=model.train_action_scheduler.add_noise(action,na,t)
                        tv,ta=teacher._predict_joint_noise(x,a,t,t,tc,tcm,True)
                        capture = capture_loop_dynamics(model.mot) if tau == .5 and model.meta['arch'] == 'loopwam' else nullcontext()
                        with capture as trace:
                            sv,sa=model.denoise_configurations(x,a,t,t,context,cm,pairs,return_video_exits=True)
                        if trace is not None:
                            loop_traces.append(trace)
                        for pair in pairs:
                            row=totals[f'{pair[0]}_{pair[1]}']
                            row['ol1']+=float(action_loss(sa[pair],ta,torch.ones(1,device=model.device),apad))/5
                            row['ol3']+=float(model._compute_video_loss_per_sample(sv[pair[0]][:,:,1:],tv[:,:,1:],vpad,False).mean())/5
                    for pair in pairs:
                        pred=_sample_actions(model,clean[:,:,:1],context,cm,pair,seed+panel_id)
                        error=(pred[:,:10].float()-action[:,:10].float()).abs().mean(-1)
                        valid=(~apad[:,:10]).float()
                        totals[f'{pair[0]}_{pair[1]}']['ol2']+=float((error*valid).sum()/valid.sum().clamp_min(1.))
            ratios=lora_norm_ratios(model.mot)
        for row in totals.values():
            for key in row: row[key]/=len(windows)
        record=dict(global_step=global_step,panel='one fixed unpadded midpoint clip per held-out demonstration',window_ids=windows,
                    tau_grid=[.1,.3,.5,.7,.9],seed=seed,metrics=totals,mode=model.mode,
                    evaluated_pairs=[list(pair) for pair in pairs],
                    loop_dynamics=_summarize_dynamics(loop_traces,windows) if loop_traces else dict(status='not_applicable',architecture=model.meta['arch']),
                    lora_norm_ratios=ratios,seconds=time.monotonic()-start)
        path=Path(output_dir)/'open_loop';path.mkdir(parents=True,exist_ok=True)
        target=path/f'step_{global_step:08d}.json'
        temporary_path=target.with_suffix('.json.tmp');temporary_path.write_text(json.dumps(record,indent=2,allow_nan=False)+'\n');temporary_path.replace(target)
        return record
    finally:
        for module,training in modes:
            module.training=training
        if temporary:
            del teacher
            if torch.device(model.device).type == 'cuda':
                torch.cuda.empty_cache()
