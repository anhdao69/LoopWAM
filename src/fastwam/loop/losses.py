"""FastWAM-compatible per-sample flow/KD reductions."""
import torch


def assert_shifts(student_video,student_action,teacher_video=5.,teacher_action=5.):
    shifts=(student_video,student_action,teacher_video,teacher_action)
    if any(float(s)!=5.0 for s in shifts):
        raise ValueError(f'LoopWAM requires teacher and student video/action sigma shift = 5.0; got {shifts}')


def action_loss(pred,target,weight,is_pad=None,is_augmented=None):
    error=(pred.float()-target.float()).square().mean(dim=-1)
    if is_pad is None:
        per_sample=error.mean(dim=-1)
    else:
        valid=(~is_pad).to(error)
        per_sample=(error*valid).sum(dim=-1)/valid.sum(dim=-1).clamp_min(1.)
    if is_augmented is not None:
        per_sample=per_sample*(~is_augmented.to(device=error.device,dtype=torch.bool)).to(error)
    return (per_sample*weight.to(per_sample)).mean()
