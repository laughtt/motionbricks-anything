"""Foot-skate losses on decoded pose features. All differentiable; skeleton-agnostic (contact joint ids from the config)."""
import torch


def local_pos(feat_raw, J):
    B, T, _ = feat_raw.shape
    return feat_raw[..., :J * 3].reshape(B, T, J, 3)


def world_vel(loc, root_raw, fps):
    """World velocity (expressed in the heading frame of frame t) of points given by their heading-frame positions.
    loc [B,T,K,3], root_raw [B,T,4] = [vx, vz, yaw rate, height] -> [B,T-1,K,3].
    pos_world(t) = rot_y(yaw_t) loc_t + r_t  =>  v = (rot_y(dyaw) loc_{t+1} - loc_t) * fps + [vx, 0, vz]."""
    dy = (root_raw[:, :-1, 2] / fps)[..., None]
    c, s = torch.cos(dy), torch.sin(dy)
    a, b = loc[:, 1:], loc[:, :-1]
    x = c * a[..., 0] + s * a[..., 2] - b[..., 0]
    z = -s * a[..., 0] + c * a[..., 2] - b[..., 2]
    y = a[..., 1] - b[..., 1]
    v = torch.stack([x, y, z], -1) * fps
    add = torch.stack([root_raw[:, :-1, 0], torch.zeros_like(dy[..., 0]), root_raw[:, :-1, 1]], -1)[:, :, None]
    return v + add


def skate_losses(rec_norm, gt_norm, root_raw, cfg, pm, ps):
    """rec_norm / gt_norm [B,T,D] normalized features, root_raw [B,T,4]. Returns dict of scalar losses (metres/second units):
    slide = decoded paw world speed (horizontal) on GT contact frames; pawvel = decoded paw world velocity vs GT (all frames);
    pawpos = paw position error (m); contact = contact channel error."""
    J, Cn = cfg.J, cfg.C
    rec = rec_norm * ps + pm; gt = gt_norm * ps + pm
    lr = local_pos(rec, J)[:, :, cfg.contacts]; lg = local_pos(gt, J)[:, :, cfg.contacts]
    vr = world_vel(lr, root_raw, cfg.fps); vg = world_vel(lg, root_raw, cfg.fps)
    con = (gt[:, :-1, -Cn:] > 0.5).float()
    hs = torch.sqrt(vr[..., 0] ** 2 + vr[..., 2] ** 2 + 1e-6)
    return dict(slide=(hs * con).sum() / con.sum().clamp(min=1), pawvel=(vr - vg).abs().mean(), pawpos=(lr - lg).abs().mean(),
                contact=(rec[..., -Cn:] - gt[..., -Cn:]).abs().mean())
