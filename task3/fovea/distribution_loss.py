"""DSNT-inspired symmetric distribution regularization for spatial localization."""
import math

import torch


def heatmap_js_loss(logits, xy, sigma=.025, temperature=1.):
    with torch.autocast(logits.device.type, enabled=False):
        height, width = logits.shape[-2:]
        y = torch.linspace(0, 1, height, device=logits.device)
        x = torch.linspace(0, 1, width, device=logits.device)
        distance = ((x[None, None, :]-xy[:, 0, None, None])**2+
                    (y[None, :, None]-xy[:, 1, None, None])**2)
        logq = (-distance/(2*sigma**2)).flatten(1).log_softmax(1)
        logp = (logits.float()/temperature).flatten(1).log_softmax(1)
        logm = torch.logaddexp(logp, logq)-math.log(2)
        return (.5*((logp.exp()*(logp-logm)).sum(1)+
                     (logq.exp()*(logq-logm)).sum(1))).mean()
