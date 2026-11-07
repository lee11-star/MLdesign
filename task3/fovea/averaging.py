"""Exponential parameter averaging without changing the source model."""
import torch


@torch.no_grad()
def update_ema(averaged, source, decay):
    for target, current in zip(averaged.parameters(), source.parameters()):
        target.lerp_(current.detach(), 1-decay)
    for target, current in zip(averaged.buffers(), source.buffers()):
        target.copy_(current)
