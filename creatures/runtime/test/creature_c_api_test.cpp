// Console test through the C API only: load the model, drive it for a few seconds, check that nothing is NaN and the root moves.
// usage: creature_c_api_test <model_dir> [threads] [walk_speed_m_s]
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "creature_runtime.h"

int main(int argc, char** argv) {
    if (argc < 2) { std::printf("usage: creature_c_api_test <model_dir> [threads] [walk_speed]\n"); return 2; }
    const int threads = argc > 2 ? std::atoi(argv[2]) : 2;
    const float walk = argc > 3 ? (float)std::atof(argv[3]) : 1.0f;
    creature_runtime* rt = creature_create();
    if (!rt) { std::printf("FAIL: create\n"); return 1; }
    if (!creature_load(rt, argv[1], threads)) { std::printf("LOAD FAILED: %s\n", creature_last_error(rt)); creature_destroy(rt); return 1; }
    const int nj = creature_joint_count(rt), nc = creature_contact_count(rt);
    std::printf("C API: joints %d, contacts %d\n", nj, nc);
    std::vector<float> joints(3 * (size_t)nj), contacts((size_t)nc);
    int bad = 0, err = 0;
    struct Seg { double s; float f, y; } seq[] = {{2, 0, 0}, {3, walk, 0}, {2, walk, 0.6f}, {2, 0, 0}};
    for (const Seg& sg : seq) {
        creature_set_command(rt, sg.f, 0, sg.y);
        for (int i = 0; i < (int)(sg.s * 60); ++i) {
            if (creature_step(rt, 1.0 / 60) < 0) { ++err; break; }
            if (creature_get_world_joints(rt, joints.data(), nj) != nj) ++err;
            for (float v : joints) bad += !std::isfinite(v);
            if (creature_get_contacts(rt, contacts.data(), nc) != nc) ++err;
            for (float v : contacts) bad += !std::isfinite(v);
        }
        float p[3] = {0, 0, 0}, yaw = 0;
        creature_get_root(rt, p, &yaw);
        std::printf("cmd fwd %.2f yaw %.2f -> root (%.2f, %.2f, %.2f) yaw %.2f\n", sg.f, sg.y, p[0], p[1], p[2], yaw);
        if (!std::isfinite(p[0]) || !std::isfinite(p[2]) || !std::isfinite(yaw)) ++bad;
    }
    float p[3];
    creature_get_root(rt, p, nullptr);
    const bool moved = std::hypot(p[0], p[2]) > 1.0;
    std::printf("non-finite values %d, API errors %d, model nan events %d, root moved: %s\n", bad, err, creature_nan_events(rt), moved ? "yes" : "NO");
    creature_reset(rt);
    creature_get_root(rt, p, nullptr);
    const bool reset_ok = std::hypot(p[0], p[2]) < 1e-3;
    std::printf("reset: root back at origin: %s\n", reset_ok ? "yes" : "NO");
    const int nan_events = creature_nan_events(rt);
    creature_destroy(rt);
    const bool ok = bad == 0 && err == 0 && nan_events == 0 && moved && reset_ok;
    std::printf("C_API_TEST %s\n", ok ? "PASS" : "FAIL");
    return ok ? 0 : 1;
}
