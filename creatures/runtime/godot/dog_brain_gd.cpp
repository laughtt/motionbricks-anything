// GDExtension wrapper: class DogBrain (RefCounted). View / input code is GDScript; all model work is in dog_brain_core.cpp.
#include <godot_cpp/classes/ref_counted.hpp>
#include <godot_cpp/core/class_db.hpp>
#include <godot_cpp/godot.hpp>
#include <godot_cpp/variant/array.hpp>
#include <godot_cpp/variant/dictionary.hpp>
#include <godot_cpp/variant/packed_float32_array.hpp>
#include <godot_cpp/variant/packed_int32_array.hpp>
#include <godot_cpp/variant/packed_string_array.hpp>
#include <godot_cpp/variant/packed_vector3_array.hpp>
#include <godot_cpp/variant/quaternion.hpp>
#include <godot_cpp/variant/string.hpp>
#include <godot_cpp/variant/vector3.hpp>

#include "dog_brain_core.hpp"

using namespace godot;

class DogBrain : public RefCounted {
    GDCLASS(DogBrain, RefCounted)
    dog::Brain b_;

    static PackedVector3Array pack(const std::vector<dog::V3>& v) {
        PackedVector3Array a;
        a.resize((int64_t)v.size());
        for (size_t i = 0; i < v.size(); ++i) a[(int64_t)i] = Vector3(v[i].x, v[i].y, v[i].z);
        return a;
    }

protected:
    static void _bind_methods() {
        ClassDB::bind_method(D_METHOD("load", "model_dir", "threads"), &DogBrain::load, DEFVAL(4));
        ClassDB::bind_method(D_METHOD("is_loaded"), &DogBrain::is_loaded);
        ClassDB::bind_method(D_METHOD("get_error"), &DogBrain::get_error);
        ClassDB::bind_method(D_METHOD("reset"), &DogBrain::reset);
        ClassDB::bind_method(D_METHOD("set_command", "forward_speed", "side_speed", "yaw_rate"), &DogBrain::set_command);
        ClassDB::bind_method(D_METHOD("step", "delta"), &DogBrain::step);
        ClassDB::bind_method(D_METHOD("configure", "settings"), &DogBrain::configure);
        ClassDB::bind_method(D_METHOD("get_config"), &DogBrain::get_config);
        ClassDB::bind_method(D_METHOD("get_stats"), &DogBrain::get_stats);
        ClassDB::bind_method(D_METHOD("get_joint_count"), &DogBrain::get_joint_count);
        ClassDB::bind_method(D_METHOD("get_joint_names"), &DogBrain::get_joint_names);
        ClassDB::bind_method(D_METHOD("get_joint_parents"), &DogBrain::get_joint_parents);
        ClassDB::bind_method(D_METHOD("get_rest_positions"), &DogBrain::get_rest_positions);
        ClassDB::bind_method(D_METHOD("get_rig_scale"), &DogBrain::get_rig_scale);
        ClassDB::bind_method(D_METHOD("get_paw_rest_height"), &DogBrain::get_paw_rest_height);
        ClassDB::bind_method(D_METHOD("get_joint_positions"), &DogBrain::get_joint_positions);
        ClassDB::bind_method(D_METHOD("get_world_joint_positions"), &DogBrain::get_world_joint_positions);
        ClassDB::bind_method(D_METHOD("get_joint_rotations"), &DogBrain::get_joint_rotations);
        ClassDB::bind_method(D_METHOD("get_root_position"), &DogBrain::get_root_position);
        ClassDB::bind_method(D_METHOD("get_root_yaw"), &DogBrain::get_root_yaw);
        ClassDB::bind_method(D_METHOD("get_contacts"), &DogBrain::get_contacts);
        ClassDB::bind_method(D_METHOD("get_feet"), &DogBrain::get_feet);
        ClassDB::bind_method(D_METHOD("self_test"), &DogBrain::self_test);
    }

public:
    bool load(const String& model_dir, int threads) { return b_.load(model_dir.utf8().get_data(), threads); }
    bool is_loaded() const { return b_.loaded(); }
    String get_error() const { return String::utf8(b_.error().c_str()); }
    void reset() { b_.reset(); }
    void set_command(double forward, double side, double yaw_rate) { b_.set_command((float)forward, (float)side, (float)yaw_rate); }
    int step(double delta) { return b_.step(delta); }

    void configure(const Dictionary& d) {
        dog::Config& c = b_.cfg;
        if (d.has("replan_frames")) c.replan_frames = (int)d["replan_frames"];
        if (d.has("crossfade_frames")) c.crossfade_frames = (int)d["crossfade_frames"];
        if (d.has("idle_replan_frames")) c.idle_replan_frames = (int)d["idle_replan_frames"];
        if (d.has("min_replan_gap")) c.min_replan_gap = (int)d["min_replan_gap"];
        if (d.has("replan_delta_speed")) c.replan_delta_speed = (float)(double)d["replan_delta_speed"];
        if (d.has("replan_delta_yaw")) c.replan_delta_yaw = (float)(double)d["replan_delta_yaw"];
        if (d.has("accel_forward")) c.accel_forward = (float)(double)d["accel_forward"];
        if (d.has("accel_side")) c.accel_side = (float)(double)d["accel_side"];
        if (d.has("accel_yaw")) c.accel_yaw = (float)(double)d["accel_yaw"];
        if (d.has("grounding")) c.grounding = (bool)d["grounding"];
        if (d.has("ground_smooth")) c.ground_smooth = (float)(double)d["ground_smooth"];
        if (d.has("ground_clearance")) c.ground_clearance = (float)(double)d["ground_clearance"];
        if (d.has("root_orient")) c.root_orient = (bool)d["root_orient"];
        if (d.has("interpolate")) c.interpolate = (bool)d["interpolate"];
        if (d.has("use_root_net")) c.use_root_net = (bool)d["use_root_net"];
        if (d.has("use_seam")) c.use_seam = (bool)d["use_seam"];
    }
    Dictionary get_config() const {
        const dog::Config& c = b_.cfg;
        Dictionary d;
        d["replan_frames"] = c.replan_frames; d["crossfade_frames"] = c.crossfade_frames; d["min_replan_gap"] = c.min_replan_gap; d["idle_replan_frames"] = c.idle_replan_frames;
        d["replan_delta_speed"] = c.replan_delta_speed; d["replan_delta_yaw"] = c.replan_delta_yaw;
        d["accel_forward"] = c.accel_forward; d["accel_side"] = c.accel_side; d["accel_yaw"] = c.accel_yaw;
        d["grounding"] = c.grounding; d["ground_smooth"] = c.ground_smooth; d["ground_clearance"] = c.ground_clearance;
        d["root_orient"] = c.root_orient; d["interpolate"] = c.interpolate; d["use_root_net"] = c.use_root_net; d["use_seam"] = c.use_seam;
        return d;
    }
    Dictionary get_stats() const {
        const dog::Stats& s = b_.stats();
        Dictionary d;
        d["plan_ms_last"] = s.plan_ms_last; d["plan_ms_avg"] = s.plan_ms_avg; d["plan_ms_max"] = s.plan_ms_max; d["load_ms"] = s.load_ms;
        d["plans"] = (int64_t)s.plans; d["frames"] = (int64_t)s.frames; d["frames_last_plan"] = s.frames_last_plan;
        d["frames_since_replan"] = s.frames_since_replan; d["nan_events"] = s.nan_events;
        const float* r = b_.applied_root();
        const float* t = b_.target();
        d["speed_side"] = r[0]; d["speed_forward"] = r[1]; d["yaw_rate"] = r[2]; d["hips_height_cmd"] = r[3];
        d["root_net"] = b_.has_root_net() && b_.cfg.use_root_net;
        d["target_forward"] = t[0]; d["target_side"] = t[1]; d["target_yaw_rate"] = t[2];
        return d;
    }
    int get_joint_count() const { return b_.joint_count(); }
    PackedStringArray get_joint_names() const {
        PackedStringArray a;
        for (const auto& n : b_.joint_names()) a.push_back(String::utf8(n.c_str()));
        return a;
    }
    PackedInt32Array get_joint_parents() const {
        PackedInt32Array a;
        for (int p : b_.parents()) a.push_back(p);
        return a;
    }
    PackedVector3Array get_rest_positions() const { return pack(b_.rest()); }
    double get_rig_scale() const { return b_.rig_scale(); }
    double get_paw_rest_height() const { return b_.paw_rest_height(); }
    PackedVector3Array get_joint_positions() const { return pack(b_.local_joints()); }
    PackedVector3Array get_world_joint_positions() const { return pack(b_.world_joints()); }
    Array get_joint_rotations() const {
        Array a;
        for (const auto& q : b_.rotations()) a.push_back(Quaternion(q.x, q.y, q.z, q.w));
        return a;
    }
    Vector3 get_root_position() const {
        const dog::V3 v = b_.root_position();
        return Vector3(v.x, v.y, v.z);
    }
    double get_root_yaw() const { return b_.root_yaw(); }
    PackedInt32Array get_feet() const {
        PackedInt32Array a;
        for (int f : b_.feet()) a.push_back(f);
        return a;
    }
    PackedFloat32Array get_contacts() const {
        PackedFloat32Array a;
        for (int i = 0; i < b_.contact_count(); ++i) a.push_back(b_.contacts()[i]);
        return a;
    }
    // {max_abs_diff (normalized pose vs the PyTorch test vector), token_mismatches}; max_abs_diff < 0 = could not run
    Dictionary self_test() {
        int mm = -1;
        Dictionary d;
        d["max_abs_diff"] = b_.self_test(&mm);
        d["token_mismatches"] = mm;
        d["root_net_max_abs_diff"] = b_.self_test_root();   // < 0 when the model has no root net
        int sm = -1;
        d["seam_max_abs_diff"] = b_.self_test_seam(&sm);    // < 0 when the model has no seam block
        d["seam_token_mismatches"] = sm;
        return d;
    }
};

static void dogbrain_init(ModuleInitializationLevel level) {
    if (level != MODULE_INITIALIZATION_LEVEL_SCENE) return;
    GDREGISTER_CLASS(DogBrain);
}
static void dogbrain_term(ModuleInitializationLevel) {}

extern "C" {
GDExtensionBool GDE_EXPORT dogbrain_library_init(GDExtensionInterfaceGetProcAddress get_proc, GDExtensionClassLibraryPtr lib,
                                                 GDExtensionInitialization* init) {
    GDExtensionBinding::InitObject obj(get_proc, lib, init);
    obj.register_initializer(dogbrain_init);
    obj.register_terminator(dogbrain_term);
    obj.set_minimum_library_initialization_level(MODULE_INITIALIZATION_LEVEL_SCENE);
    return obj.init();
}
}
