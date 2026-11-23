"""Canonical spider skeleton: 38 joints. Order is the single source of truth (spider_spec.json is written from it)."""
NAMES = ["body", "head", "abdomen", "abdomen_end"]
for _s in "LR":
    for _i in range(1, 5):
        NAMES += [f"{_s}{_i}_hip", f"{_s}{_i}_knee", f"{_s}{_i}_ankle", f"{_s}{_i}_tip"]
NAMES += ["palp_L", "palp_R"]
IDX = {n: i for i, n in enumerate(NAMES)}
PAR = [-1, 0, 0, 2]
for _s in "LR":
    for _i in range(1, 5):
        h = len(PAR); PAR += [0, h, h + 1, h + 2]
PAR += [1, 1]
J = len(NAMES)
LEGS = [[IDX[f"{s}{i}_{k}"] for k in ("hip", "knee", "ankle", "tip")] for s in "LR" for i in range(1, 5)]   # L1..L4, R1..R4
FEET = [l[3] for l in LEGS]
MIRROR = list(range(J))
for _i in range(1, 5):
    for _k in ("hip", "knee", "ankle", "tip"):
        a, b = IDX[f"L{_i}_{_k}"], IDX[f"R{_i}_{_k}"]; MIRROR[a] = b; MIRROR[b] = a
MIRROR[IDX["palp_L"]] = IDX["palp_R"]; MIRROR[IDX["palp_R"]] = IDX["palp_L"]
ROOT, FRONT = 0, 1
