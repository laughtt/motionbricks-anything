"""SOMA-30 world clips -> MotionBricks model-internal features (exactly the runtime path of predict()).

A training/eval window = 4L frames (L tokens, 6..16). The window is aligned like the runtime does it: the last context
frame (frame 3) is put at the ground origin facing +x (mb_common.alignment_of). Then the same steps as
mmm_inference_class.predict(): recentre to frame 0, normalise global root values, retarget rotations
(capture -> standard T pose, as BatchedSOMA.forward), prepend hip height, normalise local pose features.
"""
from __future__ import annotations

import numpy as np
import torch

import mb_common as mb

JN = mb.joint_names()
R_LEG, L_LEG = JN.index("RightLeg"), JN.index("LeftLeg")


def encode_frames(pos: np.ndarray, rot: np.ndarray, align=None, align_frame: int = 3):
    """World pos [F,30,3], rot [F,30,3,3] -> (g [F,5], lp [F,267], lr [F,4], align). Same maths as encode_request."""
    F = pos.shape[0]
    origin, R = mb.alignment_of(pos[align_frame]) if align is None else align
    ap = (pos - origin) @ R
    ar = np.einsum("ij,fkjl->fkil", R.T, rot)
    head = mb._heading(ap[:, R_LEG], ap[:, L_LEG])
    nxt = np.minimum(np.arange(F) + 1, F - 1)
    dh = head[nxt] - head
    dh = np.arctan2(np.sin(dh), np.cos(dh))
    hips, nh = ap[:, 0], ap[nxt, 0]
    g = np.stack([hips[:, 0], hips[:, 1], hips[:, 2], np.cos(head), np.sin(head)], -1)
    lr = np.stack([dh * mb.FPS, (nh[:, 0] - hips[:, 0]) * mb.FPS, (nh[:, 2] - hips[:, 2]) * mb.FPS, hips[:, 1]], -1)
    rootp = hips * np.array([1.0, 0.0, 1.0])
    cp = (ap[:, 1:] - rootp[:, None]).reshape(F, 87)
    r6 = np.concatenate([ar[..., 0], ar[..., 1]], -1).reshape(F, 180)
    lp = np.concatenate([cp, r6], -1)
    return g.astype(np.float32), lp.astype(np.float32), lr.astype(np.float32), (origin, R)


class Featurizer:
    """Torch side of the runtime preprocessing, bound to a loaded BatchedSOMA."""

    def __init__(self, model):
        from motionbricks_onnx.models.features import extract_feature_from_motion_rep
        from motionbricks_onnx.models.representation import global_root_to_local_onnx_compatible
        from motionbricks_onnx.models.soma.rotations import (cont6d_to_matrix, matrix_to_cont6d,
                                                            change_t_pose_global_mats_onnx_compatible)
        self.m = model
        self.inf = model.model
        self.grep = self.inf.global_motion_rep
        self.lrep = self.inf.local_motion_rep
        self._x = extract_feature_from_motion_rep
        self._g2l = global_root_to_local_onnx_compatible
        self._c2m, self._m2c, self._tp = cont6d_to_matrix, matrix_to_cont6d, change_t_pose_global_mats_onnx_compatible
        self.idx = extract_feature_from_motion_rep(torch.zeros([1, 1, len(self.lrep.indices["all"])]), self.lrep,
                                                   self.inf.INTERNAL_POSE_FEATURE_MODE, fetch_feat_idx=True)
        dev = model.capture_offsets.device
        self.mean = self.lrep.stats.mean[self.idx].to(dev)
        self.std = self.lrep.stats.std[self.idx].to(dev)
        self.rs = model.local_rotation_slice

    def retarget(self, lp: torch.Tensor, inverse=False) -> torch.Tensor:
        """[B,F,267] capture-T-pose rotations -> standard (BatchedSOMA.forward). inverse: standard -> capture."""
        B, F, _ = lp.shape
        r = lp[:, :, self.rs].reshape(B, F, 30, 6)
        mats = self._c2m(r)
        a, b = (self.m.capture_offsets, self.m.standard_offsets)
        mats = self._tp(mats, b, a) if inverse else self._tp(mats, a, b)
        return torch.cat([lp[..., :self.rs.start], self._m2c(mats).reshape(B, F, 180), lp[..., self.rs.stop:]], -1)

    def internal(self, g: torch.Tensor, lp: torch.Tensor):
        """g [B,F,5], lp [B,F,267] (aligned frame, ONNX convention) -> dict of normalised model features."""
        g = g.clone()
        g[:, :, [0, 2]] -= g[:, :1, [0, 2]]
        g_norm = self.grep.normalize(g)
        lp_std = self.retarget(lp)
        x = torch.cat([g[:, :, 1:2], lp_std], -1)
        lp_norm = (x - self.mean) / torch.sqrt(self.std ** 2 + self.inf.EPS)
        root_cond = self._x(g_norm, self.grep, "root_without_hip_height")
        F = g.shape[1]
        lr_norm = self._g2l(g_norm, self.inf.motion_rep.dual_rep, torch.full((g.shape[0], 1), F, device=g.device))
        return dict(g_norm=g_norm, lp_norm=lp_norm, root_cond=root_cond, lr_norm=lr_norm)

    @torch.no_grad()
    def tokenize(self, lp_norm: torch.Tensor) -> torch.Tensor:
        """[B,4L,268] normalised -> pose tokens [B,L,5] (nearest code per head, L2)."""
        vq = self.inf._vqvae_pose_model
        z = vq.encoder(lp_norm.permute(0, 2, 1))           # [B,160,L]
        z = z.permute(0, 2, 1)                              # [B,L,160]
        cb = vq.quantizer.vq.codebook                       # see quantizer.MultiHeadCodebook
        B, L, _ = z.shape
        H = vq.quantizer.num_heads
        zh = z.reshape(B, L, H, -1)
        cbt = cb if cb.dim() == 3 else cb.reshape(H, -1, zh.shape[-1])
        d = ((zh[:, :, :, None, :] - cbt[None, None]) ** 2).sum(-1)    # [B,L,H,16]
        return d.argmin(-1)
