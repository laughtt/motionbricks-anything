#include "dog_brain_core.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>

#include <nlohmann/json.hpp>
#include <onnxruntime_cxx_api.h>

namespace dog {
namespace {

Ort::Env& shared_env() {   // never destroyed: no destructor runs while onnxruntime.dll unloads at exit
    static Ort::Env* env = new Ort::Env(ORT_LOGGING_LEVEL_ERROR, "dogbrain");
    return *env;
}

inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float len(V3 a) { return std::sqrt(dot(a, a)); }
inline V3 norm(V3 a, V3 fallback = {0, 0, 1}) {
    const float l = len(a);
    return l > 1e-8f ? a * (1.0f / l) : fallback;
}
inline Quat qmul(Quat a, Quat b) {
    return {a.w * b.x + a.x * b.w + a.y * b.z - a.z * b.y, a.w * b.y - a.x * b.z + a.y * b.w + a.z * b.x,
            a.w * b.z + a.x * b.y - a.y * b.x + a.z * b.w, a.w * b.w - a.x * b.x - a.y * b.y - a.z * b.z};
}
inline Quat qnorm(Quat q) {
    const float l = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (l < 1e-12f) return {};
    return {q.x / l, q.y / l, q.z / l, q.w / l};
}
inline V3 qrot(Quat q, V3 v) {
    const V3 u{q.x, q.y, q.z};
    const V3 t = cross(u, v) * 2.0f;
    return v + t * q.w + cross(u, t);
}
// Minimal rotation that takes unit vector a to unit vector b (the DAMPED_TRACK rotation).
Quat swing(V3 a, V3 b) {
    const float d = dot(a, b);
    if (d < -0.999999f) {   // opposite: any axis normal to a
        V3 ax = cross(V3{1, 0, 0}, a);
        if (len(ax) < 1e-4f) ax = cross(V3{0, 1, 0}, a);
        ax = norm(ax);
        return {ax.x, ax.y, ax.z, 0};
    }
    const V3 c = cross(a, b);
    return qnorm({c.x, c.y, c.z, 1.0f + d});
}
// Rotation matrix with columns x, y, z (orthonormal) -> quaternion.
Quat from_axes(V3 x, V3 y, V3 z) {
    const float m00 = x.x, m10 = x.y, m20 = x.z, m01 = y.x, m11 = y.y, m21 = y.z, m02 = z.x, m12 = z.y, m22 = z.z;
    const float tr = m00 + m11 + m22;
    Quat q;
    if (tr > 0) {
        const float s = std::sqrt(tr + 1.0f) * 2;
        q = {(m21 - m12) / s, (m02 - m20) / s, (m10 - m01) / s, 0.25f * s};
    } else if (m00 > m11 && m00 > m22) {
        const float s = std::sqrt(1.0f + m00 - m11 - m22) * 2;
        q = {0.25f * s, (m01 + m10) / s, (m02 + m20) / s, (m21 - m12) / s};
    } else if (m11 > m22) {
        const float s = std::sqrt(1.0f + m11 - m00 - m22) * 2;
        q = {(m01 + m10) / s, 0.25f * s, (m12 + m21) / s, (m02 - m20) / s};
    } else {
        const float s = std::sqrt(1.0f + m22 - m00 - m11) * 2;
        q = {(m02 + m20) / s, (m12 + m21) / s, 0.25f * s, (m10 - m01) / s};
    }
    return qnorm(q);
}
inline float toward(float c, float t, float d) { return c + std::clamp(t - c, -d, d); }

struct Frame {
    std::vector<V3> pos;      // local joints, SOMA bone lengths, not grounded
    double x = 0, z = 0, yaw = 0;
    float ground = 0;         // smoothed height of the lowest paw joint
    float root[4] = {0, 0, 0, 0};
    std::vector<float> contact;   // one value per foot (creature-generic)
};

}  // namespace

struct Brain::Impl {
    std::string error;
    bool loaded = false;
    // spec
    int L = 8, FPT = 4, T = 32, CTX = 4, H = 5, NC = 64, D = 166, J = 27, C = 4, steps = 6;
    // creature-generic (spider, 2026-11-23): the body frame joints come from the spec block "orient"; without it = the dog (hips 16 / 20, Spine1 2, tail 24)
    int o_left = 16, o_right = 20, o_front = 2, o_back = 24;   // x = left - right, up = (front - back) x X
    long long mask_id = 64;
    float fps = 30, rig_scale = 4;
    std::vector<int> keep, parents, feet;
    std::vector<std::string> names;
    std::vector<V3> rest;
    std::vector<float> bone_len;
    std::vector<V3> rest_dir;
    std::vector<float> pm, ps, rm, rs, h_speed, h_value, ctx_start, test_root, test_pose;
    std::vector<long long> test_tokens;
    float paw_rest = 0.01f;
    // pelvis rest frame (transposed use)
    V3 prx, pry, prz;
    // model
    std::unique_ptr<Ort::Session> step_s, dec_s, root_s;
    // root net (model v2, optional)
    int root_fut = 0;
    float root_idle_eps = 0.08f;
    std::vector<float> root_start;              // [CTX*4] raw root values of the start context
    std::vector<float> rt_cmd, rt_hist, rt_ctx, rt_out;   // root-net test vector (normalized)
    // seam contract (model v2, optional)
    int seam_carry = -1, seam_left = 0;         // -1 = off
    std::vector<int> keep_seam;                 // keep schedule when (1 + seam_carry) tokens are given
    std::vector<long long> plan_tok;            // tokens of the current plan [L*H]
    std::vector<long long> st_init, st_left, st_tokens;   // seam test vector
    std::vector<float> st_pose;
    Ort::MemoryInfo mem = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    // planner state
    std::vector<float> plan, old_plan;          // [T*D] normalized
    std::vector<float> plan_root;               // [T*4] raw
    int cursor = 0, old_cursor = 0, xfade_i = 0, xfade_n = 0, since_replan = 0;
    bool have_old = false;
    std::vector<float> hist;                    // last CTX output frames, normalized [CTX*D]
    std::vector<float> hist_root;               // last CTX applied root values [CTX*4]
    float tgt[3] = {0, 0, 0};                   // forward, side, yaw
    float plan_tgt[3] = {0, 0, 0};
    double x = 0, z = 0, yaw = 0, accum = 0;
    float ground_s = 0;
    bool ground_init = false;
    Frame prev, cur;
    Stats st;
    // output
    std::vector<V3> out_pos;
    std::vector<Quat> out_rot;
    double out_x = 0, out_z = 0, out_yaw = 0;
    float out_root[4] = {0, 0, 0, 0};
    std::vector<float> out_contact;

    float height_of(float speed) const {
        const int n = (int)h_speed.size();
        if (n == 0) return 0.45f;
        if (speed <= h_speed[0]) return h_value[0];
        if (speed >= h_speed[n - 1]) return h_value[n - 1];
        for (int i = 1; i < n; ++i)
            if (speed <= h_speed[i]) {
                const float w = (speed - h_speed[i - 1]) / (h_speed[i] - h_speed[i - 1]);
                return h_value[i - 1] + (h_value[i] - h_value[i - 1]) * w;
            }
        return h_value[n - 1];
    }

    // init: n_init given tokens (the first n_init of the plan); left: n_left tokens put in front of the plan for the decoder only.
    bool run_plan(const float* ctx, const float* root_norm, float* pose_out, long long* tokens_out, const long long* init = nullptr, int n_init = 0,
                  const long long* left = nullptr, int n_left = 0) {
        const int n = L * H;
        std::vector<int64_t> tok(n, (int64_t)mask_id), pred(n, 0);
        std::vector<char> known(n, 0);
        for (int i = 0; i < n_init * H; ++i) { tok[i] = init[i]; known[i] = 1; }
        const std::vector<int>& ks = n_init > 0 ? keep_seam : keep;
        std::vector<float> conf(n), sorted(n);
        const int64_t s_tok[3] = {1, L, H}, s_ctx[3] = {1, CTX, D}, s_root[3] = {1, T, 4};
        const char* in1[3] = {"tokens", "ctx", "root"};
        const char* out1[1] = {"logits"};
        Ort::Value vals[3] = {
            Ort::Value::CreateTensor<int64_t>(mem, tok.data(), (size_t)n, s_tok, 3),
            Ort::Value::CreateTensor<float>(mem, const_cast<float*>(ctx), (size_t)CTX * D, s_ctx, 3),
            Ort::Value::CreateTensor<float>(mem, const_cast<float*>(root_norm), (size_t)T * 4, s_root, 3)};
        for (int s = 0; s < steps; ++s) {
            auto out = step_s->Run(Ort::RunOptions{nullptr}, in1, vals, 3, out1, 1);
            const float* lg = out[0].GetTensorData<float>();
            for (int i = 0; i < n; ++i) {
                const float* l = lg + (size_t)i * NC;
                int best = 0;
                for (int c = 1; c < NC; ++c)
                    if (l[c] > l[best]) best = c;
                double sum = 0;
                for (int c = 0; c < NC; ++c) sum += std::exp((double)l[c] - (double)l[best]);
                pred[i] = best;
                conf[i] = known[i] ? 2.0f : (float)(1.0 / sum);
            }
            sorted = conf;
            const int k = std::clamp(ks[s], 1, n) - 1;
            std::nth_element(sorted.begin(), sorted.begin() + k, sorted.end(), std::greater<float>());
            const float thr = sorted[k];
            for (int i = 0; i < n; ++i)
                if (!known[i] && conf[i] >= thr) {
                    tok[i] = pred[i];
                    known[i] = 1;
                }
        }
        for (int i = 0; i < n; ++i)
            if (!known[i]) tok[i] = pred[i];
        const char* in2[1] = {"tokens"};
        const char* out2[1] = {"pose"};
        if (n_left > 0) {   // decoder left context: [left tokens, plan tokens] -> drop the frames of the left tokens
            std::vector<int64_t> all((size_t)(n_left + L) * H);
            for (int i = 0; i < n_left * H; ++i) all[i] = left[i];
            std::copy(tok.begin(), tok.end(), all.begin() + (size_t)n_left * H);
            const int64_t s_all[3] = {1, n_left + L, H};
            Ort::Value v = Ort::Value::CreateTensor<int64_t>(mem, all.data(), all.size(), s_all, 3);
            auto out = dec_s->Run(Ort::RunOptions{nullptr}, in2, &v, 1, out2, 1);
            std::memcpy(pose_out, out[0].GetTensorData<float>() + (size_t)n_left * FPT * D, sizeof(float) * (size_t)T * D);
        } else {
            auto out = dec_s->Run(Ort::RunOptions{nullptr}, in2, vals, 1, out2, 1);
            std::memcpy(pose_out, out[0].GetTensorData<float>(), sizeof(float) * (size_t)T * D);
        }
        plan_tok.assign(tok.begin(), tok.end());
        if (tokens_out)
            for (int i = 0; i < n; ++i) tokens_out[i] = tok[i];
        return true;
    }

    // Normalized pose features -> local joints with the SOMA bone lengths.
    void to_joints(const float* feat, std::vector<V3>& q) const {
        std::vector<V3> p(J);
        for (int j = 0; j < J; ++j)
            p[j] = {feat[j * 3] * ps[j * 3] + pm[j * 3], feat[j * 3 + 1] * ps[j * 3 + 1] + pm[j * 3 + 1], feat[j * 3 + 2] * ps[j * 3 + 2] + pm[j * 3 + 2]};
        fix_lengths(p, q);
    }
    void fix_lengths(const std::vector<V3>& p, std::vector<V3>& q) const {
        q.resize(J);
        q[0] = p[0];
        for (int j = 1; j < J; ++j) {
            const int a = parents[j];
            q[j] = bone_len[j] > 1e-9f ? q[a] + norm(p[j] - p[a], rest_dir[j]) * bone_len[j] : q[a];
        }
    }
    float lowest_paw(const std::vector<V3>& q) const {
        float low = 1e9f;
        for (int f : feet) low = std::min(low, q[f].y);
        return low;
    }

    // Root net: cmd [1,3] = normalized [side, forward, yaw rate], root_hist [1,CTX,4] normalized, ctx [1,CTX,D] -> root [1,fut,4] normalized.
    void run_root(const float* cmd_n, const float* hist_n, const float* ctx, float* out_n) {
        const int64_t s_cmd[2] = {1, 3}, s_h[3] = {1, CTX, 4}, s_ctx[3] = {1, CTX, D};
        const char* in[3] = {"cmd", "root_hist", "ctx"};
        const char* on[1] = {"root"};
        Ort::Value vals[3] = {
            Ort::Value::CreateTensor<float>(mem, const_cast<float*>(cmd_n), 3, s_cmd, 2),
            Ort::Value::CreateTensor<float>(mem, const_cast<float*>(hist_n), (size_t)CTX * 4, s_h, 3),
            Ort::Value::CreateTensor<float>(mem, const_cast<float*>(ctx), (size_t)CTX * D, s_ctx, 3)};
        auto out = root_s->Run(Ort::RunOptions{nullptr}, in, vals, 3, on, 1);
        std::memcpy(out_n, out[0].GetTensorData<float>(), sizeof(float) * (size_t)root_fut * 4);
    }
    bool root_on(const Config& cfg) const { return root_s && cfg.use_root_net && root_fut == T - CTX; }

    void replan(const Config& cfg, bool changed) {
        std::vector<float> root(T * 4), root_n(T * 4);
        std::memcpy(root.data(), hist_root.data(), sizeof(float) * CTX * 4);
        float side = hist_root[(CTX - 1) * 4 + 0], fwd = hist_root[(CTX - 1) * 4 + 1], yw = hist_root[(CTX - 1) * 4 + 2];
        const float dt = 1.0f / fps;
        const auto t0 = std::chrono::steady_clock::now();
        const bool use_root = root_on(cfg);
        if (use_root) {
            const float cmd_n[3] = {(tgt[1] - rm[0]) / rs[0], (tgt[0] - rm[1]) / rs[1], (tgt[2] - rm[2]) / rs[2]};
            std::vector<float> hist_n((size_t)CTX * 4), fut((size_t)root_fut * 4);
            for (int i = 0; i < CTX * 4; ++i) hist_n[i] = (hist_root[i] - rm[i % 4]) / rs[i % 4];
            run_root(cmd_n, hist_n.data(), hist.data(), fut.data());
            for (int i = 0; i < root_fut * 4; ++i) {
                const float v = fut[i] * rs[i % 4] + rm[i % 4];
                root[(size_t)CTX * 4 + i] = std::isfinite(v) ? v : hist_root[(size_t)(CTX - 1) * 4 + i % 4];
            }
        }
        for (int i = CTX; i < T && !use_root; ++i) {
            fwd = toward(fwd, tgt[0], cfg.accel_forward * dt);
            side = toward(side, tgt[1], cfg.accel_side * dt);
            yw = toward(yw, tgt[2], cfg.accel_yaw * dt);
            root[i * 4 + 0] = side;
            root[i * 4 + 1] = fwd;
            root[i * 4 + 2] = yw;
            root[i * 4 + 3] = height_of(std::sqrt(fwd * fwd + side * side));
        }
        for (int i = 0; i < T * 4; ++i) root_n[i] = (root[i] - rm[i % 4]) / rs[i % 4];
        std::vector<float> fresh((size_t)T * D);
        // seam contract: regular replan (token-aligned cursor, command unchanged) -> keep the old tokens of the context frames (+ seam_carry), left context for the decoder
        const int a0 = cursor / FPT - 1;
        if (cfg.use_seam && seam_carry >= 0 && (int)plan_tok.size() == L * H && !plan.empty() && cursor % FPT == 0 && a0 >= seam_left && a0 + 1 + seam_carry <= L && !changed) {
            const std::vector<long long> old = plan_tok;
            run_plan(hist.data(), root_n.data(), fresh.data(), nullptr, &old[(size_t)a0 * H], 1 + seam_carry, &old[(size_t)(a0 - seam_left) * H], seam_left);
        } else {
            run_plan(hist.data(), root_n.data(), fresh.data(), nullptr);
        }
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        bool bad = false;
        for (float v : fresh)
            if (!std::isfinite(v)) { bad = true; break; }
        if (bad) {   // never play a broken plan: stand on the start context
            ++st.nan_events;
            for (int f = 0; f < T; ++f) std::memcpy(&fresh[(size_t)f * D], &ctx_start[(size_t)(CTX - 1) * D], sizeof(float) * D);
        }
        st.plan_ms_last = ms;
        st.plan_ms_max = std::max(st.plan_ms_max, ms);
        st.plan_ms_avg = st.plans == 0 ? ms : st.plan_ms_avg * 0.9 + ms * 0.1;
        ++st.plans;
        st.frames_last_plan = since_replan;
        have_old = !plan.empty() && cfg.crossfade_frames > 0;
        if (have_old) {
            old_plan = plan;
            old_cursor = cursor;
            xfade_i = 0;
            xfade_n = cfg.crossfade_frames;
        }
        plan = std::move(fresh);
        plan_root = std::move(root);
        cursor = CTX;
        since_replan = 0;
        for (int i = 0; i < 3; ++i) plan_tgt[i] = tgt[i];
    }

    void tick(const Config& cfg) {
        const float* lr = &hist_root[(size_t)(CTX - 1) * 4];
        const bool idle = tgt[0] == 0 && tgt[1] == 0 && tgt[2] == 0 && std::fabs(lr[0]) + std::fabs(lr[1]) + std::fabs(lr[2]) < (root_on(cfg) ? root_idle_eps : 1e-4f);
        const int every = std::clamp(idle ? std::max(cfg.idle_replan_frames, cfg.replan_frames) : cfg.replan_frames, 1, T - CTX);
        const bool changed = std::fabs(tgt[0] - plan_tgt[0]) > cfg.replan_delta_speed || std::fabs(tgt[1] - plan_tgt[1]) > cfg.replan_delta_speed ||
                             std::fabs(tgt[2] - plan_tgt[2]) > cfg.replan_delta_yaw;
        if (plan.empty() || cursor >= T || since_replan >= every || (changed && since_replan >= std::max(1, cfg.min_replan_gap))) replan(cfg, changed);
        std::vector<float> feat(plan.begin() + (size_t)cursor * D, plan.begin() + (size_t)(cursor + 1) * D);
        if (have_old && xfade_i < xfade_n && old_cursor + xfade_i < T) {
            const float w = (float)(xfade_i + 1) / (float)(xfade_n + 1);
            const float* o = &old_plan[(size_t)(old_cursor + xfade_i) * D];
            for (int i = 0; i < D; ++i) feat[i] = o[i] + (feat[i] - o[i]) * w;
        }
        ++xfade_i;
        const float* r = &plan_root[(size_t)cursor * 4];
        // history for the next context
        std::memmove(hist.data(), hist.data() + D, sizeof(float) * (size_t)(CTX - 1) * D);
        std::memcpy(hist.data() + (size_t)(CTX - 1) * D, feat.data(), sizeof(float) * D);
        std::memmove(hist_root.data(), hist_root.data() + 4, sizeof(float) * (size_t)(CTX - 1) * 4);
        std::memcpy(hist_root.data() + (size_t)(CTX - 1) * 4, r, sizeof(float) * 4);
        // frame
        prev = cur;
        to_joints(feat.data(), cur.pos);
        const float low = lowest_paw(cur.pos);
        if (!ground_init) { ground_s = low; ground_init = true; }
        ground_s += (low - ground_s) * std::clamp(cfg.ground_smooth, 0.01f, 1.0f);
        cur.ground = ground_s;
        cur.x = x; cur.z = z; cur.yaw = yaw;
        for (int i = 0; i < 4; ++i) cur.root[i] = r[i];
        cur.contact.resize(C);
        for (int i = 0; i < C; ++i) cur.contact[i] = feat[D - C + i] * ps[D - C + i] + pm[D - C + i];
        // integrate the root with the root values the plan was made for (heading frame -> world)
        const double c = std::cos(yaw), s = std::sin(yaw), dt = 1.0 / fps;
        x += (c * r[0] + s * r[1]) * dt;
        z += (-s * r[0] + c * r[1]) * dt;
        yaw += r[2] * dt;
        ++cursor; ++since_replan; ++st.frames;
        st.frames_since_replan = since_replan;
        if (prev.pos.empty()) prev = cur;
    }

    void solve(const Config& cfg) {
        float a = cfg.interpolate ? (float)std::clamp(accum * fps, 0.0, 1.0) : 1.0f;
        std::vector<V3> p(J);
        for (int j = 0; j < J; ++j) p[j] = prev.pos[j] + (cur.pos[j] - prev.pos[j]) * a;
        fix_lengths(p, out_pos);
        if (cfg.grounding) {
            const float g = prev.ground + (cur.ground - prev.ground) * a;
            for (auto& v : out_pos) v.y -= g - cfg.ground_clearance;
        }
        out_x = prev.x + (cur.x - prev.x) * a;
        out_z = prev.z + (cur.z - prev.z) * a;
        out_yaw = prev.yaw + (cur.yaw - prev.yaw) * a;
        for (int i = 0; i < 4; ++i) out_root[i] = prev.root[i] + (cur.root[i] - prev.root[i]) * a;
        out_contact.resize(C);
        if ((int)prev.contact.size() == C && (int)cur.contact.size() == C)
            for (int i = 0; i < C; ++i) out_contact[i] = prev.contact[i] + (cur.contact[i] - prev.contact[i]) * a;
        // bone rotations: root = pelvis frame; every other bone = minimal rotation from the parent-rotated rest direction (hierarchical)
        out_rot.assign(J, Quat{});
        if (cfg.root_orient) {
            const V3 cx = norm(out_pos[o_left] - out_pos[o_right], {1, 0, 0});
            const V3 cy = norm(cross(out_pos[o_front] - out_pos[o_back], cx), {0, 1, 0});
            const V3 cz = cross(cx, cy);
            // R = Fcur * Frest^T ; columns of R = cx*prx.i + cy*pry.i + cz*prz.i
            const V3 rx = cx * prx.x + cy * pry.x + cz * prz.x;
            const V3 ry = cx * prx.y + cy * pry.y + cz * prz.y;
            const V3 rz = cx * prx.z + cy * pry.z + cz * prz.z;
            out_rot[0] = from_axes(rx, ry, rz);
        }
        for (int j = 1; j < J; ++j) {
            const int pa = parents[j];
            if (bone_len[j] <= 1e-9f) { out_rot[j] = out_rot[pa]; continue; }
            const V3 from = qrot(out_rot[pa], rest_dir[j]);
            const V3 to = norm(out_pos[j] - out_pos[pa], from);
            out_rot[j] = qnorm(qmul(swing(from, to), out_rot[pa]));
        }
    }

    void reset_state(bool with_root = false) {
        plan.clear(); old_plan.clear(); plan_root.clear(); plan_tok.clear();
        cursor = 0; since_replan = 0; have_old = false; xfade_i = xfade_n = 0;
        hist = ctx_start;
        hist_root.assign((size_t)CTX * 4, 0.0f);
        for (int i = 0; i < CTX; ++i) hist_root[i * 4 + 3] = height_of(0);
        if (with_root && root_start.size() == (size_t)CTX * 4) hist_root = root_start;
        x = z = yaw = accum = 0;
        ground_init = false;
        prev = Frame{}; cur = Frame{};
        for (int i = 0; i < 3; ++i) plan_tgt[i] = tgt[i];
        const Stats keep_load = st;
        st = Stats{};
        st.load_ms = keep_load.load_ms;
    }
};

Brain::Brain() : p_(new Impl) {}
Brain::~Brain() = default;
bool Brain::loaded() const { return p_->loaded; }
const std::string& Brain::error() const { return p_->error; }
const Stats& Brain::stats() const { return p_->st; }
int Brain::joint_count() const { return p_->J; }
const std::vector<std::string>& Brain::joint_names() const { return p_->names; }
const std::vector<int>& Brain::parents() const { return p_->parents; }
const std::vector<V3>& Brain::rest() const { return p_->rest; }
float Brain::rig_scale() const { return p_->rig_scale; }
float Brain::paw_rest_height() const { return p_->paw_rest; }
const std::vector<V3>& Brain::local_joints() const { return p_->out_pos; }
const std::vector<Quat>& Brain::rotations() const { return p_->out_rot; }
V3 Brain::root_position() const { return {(float)p_->out_x, 0, (float)p_->out_z}; }
float Brain::root_yaw() const { return (float)p_->out_yaw; }
const float* Brain::applied_root() const { return p_->out_root; }
const float* Brain::target() const { return p_->tgt; }
const float* Brain::contacts() const { return p_->out_contact.data(); }
int Brain::contact_count() const { return (int)p_->out_contact.size(); }
const std::vector<int>& Brain::feet() const { return p_->feet; }
int Brain::pose_dim() const { return p_->D; }
int Brain::plan_frames() const { return p_->T; }

std::vector<V3> Brain::world_joints() const {
    std::vector<V3> w(p_->out_pos.size());
    const float c = (float)std::cos(p_->out_yaw), s = (float)std::sin(p_->out_yaw);
    for (size_t j = 0; j < w.size(); ++j) {
        const V3 v = p_->out_pos[j];
        w[j] = {c * v.x + s * v.z + (float)p_->out_x, v.y, -s * v.x + c * v.z + (float)p_->out_z};
    }
    return w;
}

bool Brain::load(const std::string& dir_utf8, int threads) {
    Impl& p = *p_;
    p.loaded = false;
    p.error.clear();
    try {
        const auto t0 = std::chrono::steady_clock::now();
        const std::filesystem::path dir = std::filesystem::u8path(dir_utf8);
        std::ifstream in(dir / "dog_brain.json");
        if (!in) throw std::runtime_error("cannot open dog_brain.json in " + dir_utf8);
        const nlohmann::json j = nlohmann::json::parse(in);
        p.L = j.at("tokens"); p.FPT = j.at("frames_per_token"); p.T = j.at("frames"); p.CTX = j.at("context_frames");
        p.H = j.at("heads"); p.NC = j.at("codes"); p.mask_id = j.at("mask_id"); p.steps = j.at("decode_steps");
        p.D = j.at("pose_dim"); p.J = j.at("joints_n"); p.fps = j.at("fps"); p.rig_scale = j.at("rig_scale");
        p.keep = j.at("keep_schedule").get<std::vector<int>>();
        p.parents = j.at("parents").get<std::vector<int>>();
        p.feet = j.at("feet").get<std::vector<int>>();
        p.names = j.at("joints").get<std::vector<std::string>>();
        p.pm = j.at("pose_mean").get<std::vector<float>>(); p.ps = j.at("pose_std").get<std::vector<float>>();
        p.rm = j.at("root_mean").get<std::vector<float>>(); p.rs = j.at("root_std").get<std::vector<float>>();
        p.h_speed = j.at("height_speed").get<std::vector<float>>(); p.h_value = j.at("height_value").get<std::vector<float>>();
        p.ctx_start = j.at("ctx_start").get<std::vector<float>>();
        if (j.contains("test")) {
            p.test_root = j["test"].at("root").get<std::vector<float>>();
            p.test_pose = j["test"].at("pose").get<std::vector<float>>();
            p.test_tokens = j["test"].at("tokens").get<std::vector<long long>>();
        }
        p.rest.clear();
        for (const auto& r : j.at("rest_metres")) p.rest.push_back({r[0].get<float>(), r[1].get<float>(), r[2].get<float>()});
        if (p.J < 2 || (int)p.rest.size() != p.J || (int)p.parents.size() != p.J || (int)p.names.size() != p.J)
            throw std::runtime_error("spec: joints, parents and rest_metres must have joints_n entries");
        p.C = (int)p.feet.size();
        if ((int)p.pm.size() != p.D || (int)p.ps.size() != p.D || (int)p.ctx_start.size() != p.CTX * p.D || (int)p.keep.size() != p.steps ||
            p.T != p.L * p.FPT || p.C < 1 || p.D != 6 * p.J + p.C)
            throw std::runtime_error("spec: sizes do not match (pose_dim must be 6 * joints_n + number of feet)");
        for (int f : p.feet)
            if (f < 0 || f >= p.J) throw std::runtime_error("spec: foot joint out of range");
        p.o_left = 16; p.o_right = 20; p.o_front = 2; p.o_back = 24;
        if (j.contains("orient")) {   // body frame joints, rigid with the pelvis / cephalothorax
            p.o_left = j["orient"].at("left"); p.o_right = j["orient"].at("right"); p.o_front = j["orient"].at("front"); p.o_back = j["orient"].at("back");
        } else if (p.J != 27) {
            throw std::runtime_error("spec: block 'orient' {left, right, front, back} is needed for a skeleton that is not the 27-joint dog");
        }
        for (int o : {p.o_left, p.o_right, p.o_front, p.o_back})
            if (o < 0 || o >= p.J) throw std::runtime_error("spec: orient joint out of range");
        p.out_contact.assign(p.C, 0.0f);
        p.bone_len.assign(p.J, 0); p.rest_dir.assign(p.J, V3{0, 0, 1});
        for (int k = 1; k < p.J; ++k) {
            const V3 d = p.rest[k] - p.rest[p.parents[k]];
            p.bone_len[k] = len(d);
            p.rest_dir[k] = norm(d);
        }
        p.paw_rest = 1e9f;
        for (int f : p.feet) p.paw_rest = std::min(p.paw_rest, p.rest[f].y);
        cfg.ground_clearance = p.paw_rest;
        p.prx = norm(p.rest[p.o_left] - p.rest[p.o_right], {1, 0, 0});
        p.pry = norm(cross(p.rest[p.o_front] - p.rest[p.o_back], p.prx), {0, 1, 0});
        p.prz = cross(p.prx, p.pry);
        Ort::SessionOptions so;
        so.SetGraphOptimizationLevel(ORT_ENABLE_ALL);
        so.SetInterOpNumThreads(1);
        so.SetIntraOpNumThreads(std::max(1, threads));
        const std::filesystem::path f1 = dir / j.at("pose_step").get<std::string>(), f2 = dir / j.at("tok_dec").get<std::string>();
        p.step_s = std::make_unique<Ort::Session>(shared_env(), f1.c_str(), so);
        p.dec_s = std::make_unique<Ort::Session>(shared_env(), f2.c_str(), so);
        p.seam_carry = -1; p.seam_left = 0;
        if (j.contains("seam") && j["seam"].value("enabled", true)) {   // model v2 (optional): see MODEL_CONTRACT.md
            const auto& sm = j["seam"];
            p.seam_carry = sm.at("carry_tokens"); p.seam_left = sm.at("left_ctx");
            p.keep_seam = sm.at("keep_schedule").get<std::vector<int>>();
            if (sm.contains("test")) {
                p.st_init = sm["test"].at("init").get<std::vector<long long>>(); p.st_left = sm["test"].at("left").get<std::vector<long long>>();
                p.st_tokens = sm["test"].at("tokens").get<std::vector<long long>>(); p.st_pose = sm["test"].at("pose").get<std::vector<float>>();
            }
            if ((int)p.keep_seam.size() != p.steps || p.seam_carry < 0 || p.seam_left < 0) throw std::runtime_error("spec: seam sizes do not match");
        }
        p.root_s.reset();
        p.root_fut = 0;
        if (j.contains("root_net") && j["root_net"].value("enabled", true)) {   // model v2 (optional): see MODEL_CONTRACT.md
            const auto& rn = j["root_net"];
            p.root_fut = rn.at("frames");
            p.root_idle_eps = rn.value("idle_eps", 0.08f);
            p.root_start = rn.at("root_start").get<std::vector<float>>();
            if (rn.contains("test")) {
                p.rt_cmd = rn["test"].at("cmd").get<std::vector<float>>(); p.rt_hist = rn["test"].at("root_hist").get<std::vector<float>>();
                p.rt_ctx = rn["test"].at("ctx").get<std::vector<float>>(); p.rt_out = rn["test"].at("root").get<std::vector<float>>();
            }
            if (p.root_fut != p.T - p.CTX || p.root_start.size() != (size_t)p.CTX * 4) throw std::runtime_error("spec: root_net sizes do not match");
            const std::filesystem::path f3 = dir / rn.at("file").get<std::string>();
            p.root_s = std::make_unique<Ort::Session>(shared_env(), f3.c_str(), so);
        }
        p.out_pos = p.rest;
        p.out_rot.assign(p.J, Quat{});
        p.reset_state(p.root_on(cfg));
        p.st.load_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        p.loaded = true;
    } catch (const std::exception& e) {
        p.error = e.what();
    }
    return p.loaded;
}

void Brain::reset() {
    if (!p_->loaded) return;
    p_->reset_state(p_->root_on(cfg));
    p_->out_pos = p_->rest;
    p_->out_rot.assign(p_->J, Quat{});
    p_->out_x = p_->out_z = p_->out_yaw = 0;
}

void Brain::set_command(float forward, float side, float yaw_rate) {
    p_->tgt[0] = std::isfinite(forward) ? forward : 0;
    p_->tgt[1] = std::isfinite(side) ? side : 0;
    p_->tgt[2] = std::isfinite(yaw_rate) ? yaw_rate : 0;
}

int Brain::step(double delta) {
    Impl& p = *p_;
    if (!p.loaded || !(delta >= 0)) return 0;
    p.accum += std::min(delta, 0.25);
    const double dt = 1.0 / p.fps;
    int n = 0;
    try {
        if (p.cur.pos.empty()) { p.tick(cfg); ++n; }
        while (p.accum >= dt && n < 8) {
            p.tick(cfg);
            p.accum -= dt;
            ++n;
        }
        if (p.accum >= dt) p.accum = 0;   // too slow: drop time, do not spiral
        p.solve(cfg);
    } catch (const std::exception& e) {
        p.error = e.what();
        p.loaded = false;
    }
    return n;
}

bool Brain::plan_once(const float* ctx, const float* root_norm, float* pose_out, long long* tokens_out) {
    if (!p_->loaded) return false;
    try {
        return p_->run_plan(ctx, root_norm, pose_out, tokens_out);
    } catch (const std::exception& e) {
        p_->error = e.what();
        return false;
    }
}

bool Brain::has_root_net() const { return (bool)p_->root_s; }
bool Brain::has_seam() const { return p_->seam_carry >= 0; }

double Brain::self_test_seam(int* token_mismatches) {
    Impl& p = *p_;
    const size_t ni = (size_t)(1 + std::max(p.seam_carry, 0)) * p.H, nl = (size_t)p.seam_left * p.H;
    if (!p.loaded || p.seam_carry < 0 || p.st_init.size() != ni || p.st_left.size() != nl || p.st_pose.size() != (size_t)p.T * p.D ||
        p.test_root.size() != (size_t)p.T * 4)
        return -1;
    try {
        std::vector<float> out((size_t)p.T * p.D);
        std::vector<long long> tok((size_t)p.L * p.H);
        const std::vector<long long> keep_tok = p.plan_tok;
        p.run_plan(p.ctx_start.data(), p.test_root.data(), out.data(), tok.data(), p.st_init.data(), 1 + p.seam_carry, p.st_left.data(), p.seam_left);
        p.plan_tok = keep_tok;
        int mm = 0;
        for (size_t i = 0; i < tok.size() && i < p.st_tokens.size(); ++i) mm += tok[i] != p.st_tokens[i];
        if (token_mismatches) *token_mismatches = mm;
        double d = 0;
        for (size_t i = 0; i < out.size(); ++i) d = std::max(d, (double)std::fabs(out[i] - p.st_pose[i]));
        return d;
    } catch (const std::exception& e) {
        p.error = e.what();
        return -1;
    }
}

double Brain::self_test_root() {
    Impl& p = *p_;
    if (!p.loaded || !p.root_s || p.rt_cmd.size() != 3 || p.rt_hist.size() != (size_t)p.CTX * 4 || p.rt_ctx.size() != (size_t)p.CTX * p.D ||
        p.rt_out.size() != (size_t)p.root_fut * 4)
        return -1;
    try {
        std::vector<float> out(p.rt_out.size());
        p.run_root(p.rt_cmd.data(), p.rt_hist.data(), p.rt_ctx.data(), out.data());
        double d = 0;
        for (size_t i = 0; i < out.size(); ++i) d = std::max(d, (double)std::fabs(out[i] - p.rt_out[i]));
        return d;
    } catch (const std::exception& e) {
        p.error = e.what();
        return -1;
    }
}

double Brain::self_test(int* token_mismatches) {
    Impl& p = *p_;
    if (!p.loaded || p.test_pose.size() != (size_t)p.T * p.D || p.test_root.size() != (size_t)p.T * 4) return -1;
    std::vector<float> out((size_t)p.T * p.D);
    std::vector<long long> tok((size_t)p.L * p.H);
    const std::vector<long long> keep_tok = p.plan_tok;   // a test must not change the planner state
    const bool ok = plan_once(p.ctx_start.data(), p.test_root.data(), out.data(), tok.data());
    p.plan_tok = keep_tok;
    if (!ok) return -1;
    int mm = 0;
    for (size_t i = 0; i < tok.size() && i < p.test_tokens.size(); ++i) mm += tok[i] != p.test_tokens[i];
    if (token_mismatches) *token_mismatches = mm;
    double d = 0;
    for (size_t i = 0; i < out.size(); ++i) d = std::max(d, (double)std::fabs(out[i] - p.test_pose[i]));
    return d;
}

}  // namespace dog
