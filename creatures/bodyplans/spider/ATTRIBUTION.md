# Attribution: spider animation sources

The spider data comes from two packs on Fab. Both are CC BY per the Fab listing (confirmed by the author on the Fab listings).
No file from these packs is in this repository. Get your own copies from Fab, then run `extract_clips.py`.

| Pack | Author | Licence | Source | Used for |
|---|---|---|---|---|
| "Spider animated character" (idle, idle long, walk, attack 1, attack 2, damage, death) | Bestgameassets | CC BY (confirmed on the Fab listing) | https://www.fab.com/listings/62456bda-b5f6-4372-8673-6471359c5e50 | skeleton proportions, the gait of the walk, real clips |
| "Spiders - characters with animations" (10 spiders; the Argiope is used) | Mixall | CC BY (confirmed on the Fab listing) | https://www.fab.com/listings/41eeeaca-af9e-43b2-8a71-0874356cebb2 | real clips (idle, attack) |

What we did to the data: we took out the joint positions, mapped them to a 38-joint skeleton, retargeted, mirrored and time-warped them, and mixed them with generated walking clips.
Files in this repository that come from these packs: `spider_spec.json` (rest pose), `spider_collision.json`, the rig maps, and the weights in `creatures/models/spider_v3/`.
Give this attribution when you show or ship anything made from these files.
