// C API over dog::Brain. See creature_runtime.h.
#include "creature_runtime.h"

#include <exception>
#include <string>

#include "dog_brain_core.hpp"

struct creature_runtime {
    dog::Brain brain;
    std::string error;
};

namespace {
void set_err(creature_runtime* rt, const char* msg) { if (rt) rt->error = msg ? msg : "unknown error"; }
}  // namespace

extern "C" {

creature_runtime* creature_create(void) {
    try { return new creature_runtime(); } catch (...) { return nullptr; }
}

void creature_destroy(creature_runtime* rt) { delete rt; }

int creature_load(creature_runtime* rt, const char* model_dir_utf8, int threads) {
    if (!rt || !model_dir_utf8) { set_err(rt, "null argument"); return 0; }
    try {
        rt->error.clear();
        if (!rt->brain.load(model_dir_utf8, threads < 1 ? 1 : threads)) { rt->error = rt->brain.error(); return 0; }
        return 1;
    } catch (const std::exception& e) { set_err(rt, e.what()); return 0; }
    catch (...) { set_err(rt, "unknown exception"); return 0; }
}

void creature_reset(creature_runtime* rt) {
    if (!rt || !rt->brain.loaded()) return;
    try { rt->brain.reset(); } catch (const std::exception& e) { set_err(rt, e.what()); }
}

void creature_set_command(creature_runtime* rt, float forward, float side, float yaw_rate) {
    if (!rt || !rt->brain.loaded()) return;
    rt->brain.set_command(forward, side, yaw_rate);
}

int creature_step(creature_runtime* rt, double dt) {
    if (!rt || !rt->brain.loaded()) { set_err(rt, "no model loaded"); return -1; }
    try { return rt->brain.step(dt); } catch (const std::exception& e) { set_err(rt, e.what()); return -1; }
    catch (...) { set_err(rt, "unknown exception"); return -1; }
}

int creature_joint_count(const creature_runtime* rt) { return rt && rt->brain.loaded() ? rt->brain.joint_count() : 0; }
int creature_contact_count(const creature_runtime* rt) { return rt && rt->brain.loaded() ? rt->brain.contact_count() : 0; }

int creature_get_world_joints(const creature_runtime* rt, float* out_xyz, int max_joints) {
    if (!rt || !rt->brain.loaded() || !out_xyz || max_joints <= 0) return 0;
    const std::vector<dog::V3> w = rt->brain.world_joints();
    const int n = (int)w.size() < max_joints ? (int)w.size() : max_joints;
    for (int i = 0; i < n; ++i) { out_xyz[3 * i] = w[i].x; out_xyz[3 * i + 1] = w[i].y; out_xyz[3 * i + 2] = w[i].z; }
    return n;
}

void creature_get_root(const creature_runtime* rt, float* out_xyz, float* out_yaw) {
    if (!rt || !rt->brain.loaded()) return;
    if (out_xyz) { const dog::V3 v = rt->brain.root_position(); out_xyz[0] = v.x; out_xyz[1] = v.y; out_xyz[2] = v.z; }
    if (out_yaw) *out_yaw = rt->brain.root_yaw();
}

int creature_get_contacts(const creature_runtime* rt, float* out, int max_contacts) {
    if (!rt || !rt->brain.loaded() || !out || max_contacts <= 0) return 0;
    const int n = rt->brain.contact_count() < max_contacts ? rt->brain.contact_count() : max_contacts;
    for (int i = 0; i < n; ++i) out[i] = rt->brain.contacts()[i];
    return n;
}

int creature_nan_events(const creature_runtime* rt) { return rt && rt->brain.loaded() ? rt->brain.stats().nan_events : 0; }

const char* creature_last_error(const creature_runtime* rt) { return rt ? rt->error.c_str() : "null handle"; }

}  // extern "C"
