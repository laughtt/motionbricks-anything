# MotionBricks Anything

**Unofficial extension of NVIDIA MotionBricks. Not affiliated with NVIDIA.**

<p align="center">
  <a href="https://ko-fi.com/josemauriciocarpioherrera"><img src="https://ko-fi.com/img/githubbutton_sm.svg" alt="Support me on Ko-fi" height="54"></a>
</p>
<p align="center">
  <b>This work is free. If it helps you, please support it on Ko-fi.</b><br>
  <a href="https://x.com/carpio_her8717">X</a> ·
  <a href="https://www.linkedin.com/in/mauriciocarpioherrera/">LinkedIn</a>
</p>

**Train a MotionBricks motion model for any animal, then play it live in Godot.** Read the full story in [`creatures/`](creatures/README.md).

<table>
  <tr><th>Spider (shipped model)</th><th>Husky (dog model, train it yourself)</th></tr>
  <tr>
    <td><img src="creatures/docs/media/sandbox_spider.gif" alt="Spider driven by the shipped model in the Godot sandbox" width="100%"></td>
    <td><img src="creatures/docs/media/sandbox_husky.gif" alt="Husky driven by the dog model in the Godot sandbox" width="100%"></td>
  </tr>
</table>

| [Results](creatures/README.md#results) | [Try it](creatures/README.md#try-it) | [How it works](creatures/README.md#how-it-works) | [Train your own animal](creatures/README.md#train-your-own-animal) | [The 100 MB model](creatures/README.md#the-100-mb-motionbricks-model) |
|---|---|---|---|---|

## Get the files (Git LFS)

Install [Git LFS](https://git-lfs.com/) first. A normal `git clone` then downloads only the creature models and media (`creatures/**`, set in [`.lfsconfig`](.lfsconfig)).
To also get the large NVIDIA human models (about 2.4 GB in total), run:

```sh
git lfs pull --include="*"
```

**Prebuilt binaries: coming soon (GitHub Release).** Until then, build the Godot extension yourself: see [Try it](creatures/README.md#try-it).

## NVIDIA MotionBricks (upstream)

This repository started from [NVIDIA MotionBricks](https://github.com/NVlabs/motionbricks) (Apache-2.0). The original NVIDIA README is unchanged in [README_UPSTREAM.md](README_UPSTREAM.md).
The upstream code, models and licences (`LICENSE`, `licenses/`) are kept as they are. The additions are in [`creatures/`](creatures/README.md); see [`creatures/NOTICE`](creatures/NOTICE).
