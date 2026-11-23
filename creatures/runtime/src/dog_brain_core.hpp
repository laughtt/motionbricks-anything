// DogBrain core: the dog MotionBricks model at runtime (ONNX Runtime, CPU). No Godot types here.
// Axes: Y up, forward +Z, left +X, metres (the same as Godot after the glTF import of the dog SOMA rig).
#pragma once
#include <memory>
#include <string>
#include <vector>

namespace dog {

struct V3 { float x = 0, y = 0, z = 0; };
struct Quat { float x = 0, y = 0, z = 0, w = 1; };

struct Config {
    int replan_frames = 12;        // a new plan every N frames (30 fps); 1..28
    int crossfade_frames = 4;      // blend old plan -> new plan over these frames
    int idle_replan_frames = 28;   // while the dog stands (zero command, zero root speed): replan only every N frames (less idle jitter)
    int min_replan_gap = 4;        // a command change triggers an early replan, but not sooner than this
    float replan_delta_speed = 0.25f;  // m/s change of the target that triggers an early replan
    float replan_delta_yaw = 0.35f;    // rad/s change of the target that triggers an early replan
    float accel_forward = 2.5f;    // m/s^2 limit of the commanded root
    float accel_side = 1.5f;       // m/s^2
    float accel_yaw = 5.0f;        // rad/s^2
    bool grounding = true;         // keep the lowest paw on the floor (smoothed)
    float ground_smooth = 0.25f;   // 0..1 per frame (1 = no smoothing)
    float ground_clearance = 0.01f;    // height of the lowest paw JOINT when the paw mesh touches the floor (per body)
    bool root_orient = true;       // root bone rotates with the pelvis (resolves the free twist; no skin flips on turns)
    bool interpolate = true;       // interpolate between the 30 fps model frames for the render frame
    bool use_seam = true;          // model v2: when the spec has "seam", a regular replan keeps the previous plan's tokens of the context frames
                                   // (+ carry_tokens more) and decodes with left_ctx previous tokens in front of the new ones (smaller seam gap)
    bool use_root_net = true;      // model v2: when the spec has "root_net" (enabled), the root path of a plan comes from root_net.onnx
                                   // (command + root history + pose context -> root values); false = the old acceleration-limited command path
};

struct Stats {
    double plan_ms_last = 0, plan_ms_avg = 0, plan_ms_max = 0, load_ms = 0;
    long long plans = 0, frames = 0;
    int frames_last_plan = 0;      // frames played from the previous plan before the last replan
    int frames_since_replan = 0;
    int nan_events = 0;
};

class Brain {
public:
    Brain();
    ~Brain();
    bool load(const std::string& model_dir_utf8, int threads);
    bool loaded() const;
    const std::string& error() const;
    void reset();
    void set_command(float forward, float side, float yaw_rate);
    int step(double delta);        // returns the number of 30 fps frames advanced

    Config cfg;
    const Stats& stats() const;

    int joint_count() const;
    const std::vector<std::string>& joint_names() const;
    const std::vector<int>& parents() const;
    const std::vector<V3>& rest() const;          // SOMA rest joints, metres
    float rig_scale() const;
    float paw_rest_height() const;                // lowest paw joint in the rest pose, metres

    // Output of the last step (render frame).
    const std::vector<V3>& local_joints() const;  // character space (root xz = 0, heading +Z), bone lengths = SOMA, grounded
    const std::vector<Quat>& rotations() const;   // per joint j: rotation (character space) of the bone that ends at joint j; [0] = root / pelvis
    std::vector<V3> world_joints() const;
    V3 root_position() const;                     // world, y = 0
    float root_yaw() const;                       // rad about +Y; 0 = facing +Z
    const float* applied_root() const;            // [vx side, vz forward, yaw rate, hips height] of the current frame
    const float* target() const;                  // [forward, side, yaw rate] command
    int contact_count() const;                    // number of feet / contact values (4 for the dog, 8 for the spider)
    const std::vector<int>& feet() const;         // foot joint indices
    const float* contacts() const;                // 4 foot contact values (model output, about 0..1)

    // Test hooks.
    bool plan_once(const float* ctx_norm, const float* root_norm, float* pose_out, long long* tokens_out);
    bool has_seam() const;                        // the spec has the v2 seam block
    double self_test_seam(int* token_mismatches); // seam path (given tokens + left context) vs the PyTorch test vector; < 0 = none / failure
    double self_test(int* token_mismatches);      // max abs diff (normalized pose) vs the PyTorch test vector in the spec; < 0 = failure
    bool has_root_net() const;                    // the spec has a root net and it loaded
    double self_test_root();                      // max abs diff (normalized root values) vs the PyTorch root-net test vector; < 0 = no root net / failure
    int pose_dim() const;
    int plan_frames() const;

private:
    struct Impl;
    std::unique_ptr<Impl> p_;
};

}  // namespace dog
