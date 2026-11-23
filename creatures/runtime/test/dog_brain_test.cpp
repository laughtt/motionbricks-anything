// Console test (no Godot): load the model, compare with the PyTorch test vector, time the plans, run a scripted command sequence.
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include "dog_brain_core.hpp"

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: dog_brain_test <model_dir> [threads]\n"); return 2; }
    dog::Brain b;
    const int threads = argc > 2 ? std::atoi(argv[2]) : 4;
    if (!b.load(argv[1], threads)) { std::printf("LOAD FAILED: %s\n", b.error().c_str()); return 1; }
    int mm = -1;
    const double d = b.self_test(&mm);
    std::printf("self_test: max abs diff %.3g (normalized), token mismatches %d, load %.1f ms, threads %d\n", d, mm, b.stats().load_ms, threads);
    const double dr = b.self_test_root();
    std::printf("root net: %s, max abs diff %.3g (normalized)\n", b.has_root_net() ? "yes" : "no (old model: command path)", dr);
    int ms_ = -1;
    const double ds = b.self_test_seam(&ms_);
    std::printf("seam contract: %s, max abs diff %.3g (normalized), token mismatches %d\n", b.has_seam() ? "yes" : "no", ds, ms_);
    for (int a = 3; a + 1 < argc; a += 2) {   // optional overrides: replan <n> crossfade <n> idle <n> rootnet <0|1> seam <0|1>
        const std::string k = argv[a];
        if (k == "replan") b.cfg.replan_frames = std::atoi(argv[a + 1]);
        if (k == "crossfade") b.cfg.crossfade_frames = std::atoi(argv[a + 1]);
        if (k == "idle") b.cfg.idle_replan_frames = std::atoi(argv[a + 1]);
        if (k == "rootnet") b.cfg.use_root_net = std::atoi(argv[a + 1]) != 0;
        if (k == "seam") b.cfg.use_seam = std::atoi(argv[a + 1]) != 0;
    }
    float walk = 1.0f, fast = 2.2f;
    for (int a = 3; a + 1 < argc; a += 2) {   // creature-generic: walk <m/s> fast <m/s>
        if (std::string(argv[a]) == "walk") walk = (float)std::atof(argv[a + 1]);
        if (std::string(argv[a]) == "fast") fast = (float)std::atof(argv[a + 1]);
    }
    struct Seg { double s; float f, y; } seq[] = {{4, 0, 0}, {3, walk, 0}, {3, walk, 0.8f}, {3, fast, 0}, {4, 0, 0}};
    const std::vector<int> feet = b.feet();
    const int NF = (int)feet.size();
    std::printf("joints %d, feet %d\n", b.joint_count(), NF);
    int bad = 0;
    std::vector<dog::V3> last;
    for (const Seg& sg : seq) {
        b.set_command(sg.f, 0, sg.y);
        double paw = 0, slide = 0; int np = 0, ns = 0, planted = 0;
        for (int i = 0; i < (int)(sg.s * 60); ++i) {
            b.step(1.0 / 60);
            for (const auto& v : b.local_joints()) bad += !(std::isfinite(v.x) && std::isfinite(v.y) && std::isfinite(v.z));
            const std::vector<dog::V3> w = b.world_joints();
            if (!last.empty() && i >= 60)   // skip the first second of the segment (transition)
                for (int f = 0; f < NF; ++f) {
                    const double v = std::hypot(w[feet[f]].x - last[feet[f]].x, w[feet[f]].z - last[feet[f]].z) * 60;
                    paw += v; ++np; planted += v < 0.2;
                    if (b.local_joints()[feet[f]].y < 0.03f) { slide += v; ++ns; }
                }
            last = w;
        }
        const dog::V3 r = b.root_position();
        std::printf("cmd fwd %.1f yaw %.1f -> root (%.2f, %.2f) yaw %.2f, applied fwd %.2f, hips y %.3f | paw speed mean %.3f m/s, low paws (<3 cm) %.3f m/s, paw-frames slower than 0.2 m/s %.0f%%\n", sg.f, sg.y,
                    r.x, r.z, b.root_yaw(), b.applied_root()[1], b.local_joints()[0].y, paw / std::max(np, 1), slide / std::max(ns, 1), 100.0 * planted / std::max(np, 1));
    }
    const dog::Stats& s = b.stats();
    std::printf("plans %lld frames %lld plan ms avg %.2f max %.2f nan %d bad %d\n", s.plans, s.frames, s.plan_ms_avg, s.plan_ms_max, s.nan_events, bad);
    const bool root_ok = !b.has_root_net() || (dr >= 0 && dr < 1e-3);
    const bool seam_ok = !b.has_seam() || (ds >= 0 && ds < 1e-3 && ms_ == 0);
    return (d >= 0 && d < 1e-3 && mm == 0 && bad == 0 && s.nan_events == 0 && root_ok && seam_ok) ? 0 : 1;
}
