"""Train / predict loops, shared by both arms.

One trainer, two arms. The frame-wise and temporal models differ only in the
shape of the tensors flowing through, so they share optimiser, scheduler, LR,
weight decay, epoch count, early-stopping rule, augmentation, and class
weighting. Any of those diverging would give the temporal arm an advantage
that has nothing to do with temporal modelling, which is precisely the
confound the study is about.
"""
from __future__ import annotations

import copy
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .utils import get_logger

log = get_logger("engine")


def make_optimizer(model, cfg):
    t = cfg.get("optimizer", "adamw").lower()
    lr, wd = cfg.get("lr", 1e-4), cfg.get("weight_decay", 1e-4)
    params = [p for p in model.parameters() if p.requires_grad]
    if t == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=wd)
    if t == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd,
                               nesterov=True)
    raise ValueError(t)


def make_scheduler(opt, cfg, steps_per_epoch: int):
    epochs = cfg.get("epochs", 20)
    warm = cfg.get("warmup_epochs", 1)
    total = epochs * steps_per_epoch
    wsteps = max(1, warm * steps_per_epoch)

    def fn(step):
        if step < wsteps:
            return step / wsteps
        prog = (step - wsteps) / max(1, total - wsteps)
        return 0.5 * (1 + np.cos(np.pi * min(1.0, prog)))

    return torch.optim.lr_scheduler.LambdaLR(opt, fn)


def _forward_loss(model, batch, criterion, device, temporal: bool):
    if temporal:
        x, y, _ = batch                                  # (B,T,3,H,W), (B,T)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)                                # (B,T,C)
        loss = criterion(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
    else:
        x, y, _ = batch
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
    return loss, logits, y


def train_one_epoch(model, loader, criterion, opt, sched, device, temporal,
                    scaler=None, grad_clip=1.0):
    model.train()
    tot, n = 0.0, 0
    for batch in loader:
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss, _, y = _forward_loss(model, batch, criterion, device, temporal)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(opt)
            scaler.update()
        else:
            loss, _, y = _forward_loss(model, batch, criterion, device, temporal)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()
        sched.step()
        bs = y.numel()
        tot += loss.item() * bs
        n += bs
    return tot / max(1, n)


@torch.no_grad()
def predict(model, loader, device, temporal: bool, n_rows: int) -> np.ndarray:
    """Return P(standard plane) per frame, indexed by dataframe row.

    For the temporal arm, clip windows overlap, so a frame can be predicted
    more than once. We average the probabilities over all windows containing
    it -- this is standard sliding-window inference and it is also what makes
    the temporal arm's output directly comparable to the frame-wise arm's:
    both produce exactly one probability per frame of the test set.
    """
    model.eval()
    acc = np.zeros(n_rows, dtype=np.float64)
    cnt = np.zeros(n_rows, dtype=np.float64)

    for batch in loader:
        if temporal:
            x, _, rows = batch
            p = torch.softmax(model(x.to(device)), dim=-1)[..., 1].cpu().numpy()
            r = rows.numpy()
            np.add.at(acc, r.reshape(-1), p.reshape(-1))
            np.add.at(cnt, r.reshape(-1), 1.0)
        else:
            x, _, idx = batch
            p = torch.softmax(model(x.to(device)), dim=-1)[:, 1].cpu().numpy()
            i = idx.numpy()
            acc[i] += p
            cnt[i] += 1.0

    if (cnt == 0).any():
        log.warning("%d frames received no prediction", int((cnt == 0).sum()))
        cnt[cnt == 0] = 1.0
    return acc / cnt


def fit(model, train_loader, val_loader, val_df, cfg, device, temporal,
        class_weight=None):
    """Train with early stopping on validation macro-F1.

    Selecting on macro-F1 rather than loss or accuracy matters under class
    imbalance: a model can improve its loss while getting worse at the
    minority class that the entire task is about.
    """
    from .metrics import frame_metrics, pick_threshold

    criterion = nn.CrossEntropyLoss(
        weight=None if class_weight is None else class_weight.to(device))
    opt = make_optimizer(model, cfg)
    sched = make_scheduler(opt, cfg, max(1, len(train_loader)))
    use_amp = bool(cfg.get("amp", True)) and device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    epochs = cfg.get("epochs", 20)
    patience = cfg.get("patience", 6)
    best = {"macro_f1": -1.0, "epoch": -1}
    best_state, bad = None, 0
    history = []

    for ep in range(1, epochs + 1):
        t0 = time.time()
        tr_loss = train_one_epoch(model, train_loader, criterion, opt, sched,
                                  device, temporal, scaler,
                                  cfg.get("grad_clip", 1.0))
        vp = predict(model, val_loader, device, temporal, len(val_df))
        thr = pick_threshold(val_df.label.values, vp, "macro_f1")
        m = frame_metrics(val_df.label.values, vp, thr)
        m.update(epoch=ep, train_loss=tr_loss, secs=round(time.time() - t0, 1))
        history.append(m)
        log.info("ep %02d loss %.4f | val macroF1 %.4f balAcc %.4f auprc %.4f "
                 "thr %.2f | %.0fs", ep, tr_loss, m["macro_f1"],
                 m["balanced_accuracy"], m["auprc"], thr, m["secs"])

        if m["macro_f1"] > best["macro_f1"]:
            best = {**m, "epoch": ep, "threshold": thr}
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                log.info("early stop at epoch %d (best was %d)", ep, best["epoch"])
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best, history


def make_loader(ds, batch_size, shuffle, workers=4, pin=True):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, pin_memory=pin,
                      drop_last=False, persistent_workers=workers > 0)
