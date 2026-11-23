/* Creature runtime: plain C API over the C++ core (class dog::Brain).
 * Use it from C, C#, Python (ctypes), Rust, Unity, Unreal, or any engine that can call a C library.
 * Axes: Y up, forward +Z, left +X, metres. Joint order = the model spec (dog_brain.json).
 * All functions take the handle from creature_create(). One handle must be used from one thread at a time.
 */
#ifndef CREATURE_RUNTIME_H
#define CREATURE_RUNTIME_H

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32) && defined(CREATURE_RUNTIME_SHARED)
#  ifdef CREATURE_RUNTIME_BUILD
#    define CREATURE_API __declspec(dllexport)
#  else
#    define CREATURE_API __declspec(dllimport)
#  endif
#elif defined(__GNUC__)
#  define CREATURE_API __attribute__((visibility("default")))
#else
#  define CREATURE_API
#endif

typedef struct creature_runtime creature_runtime;

/* Create / destroy. creature_create returns NULL only when memory runs out. */
CREATURE_API creature_runtime* creature_create(void);
CREATURE_API void creature_destroy(creature_runtime* rt);

/* Load a model folder (dog_brain.json + the ONNX files it names). threads = CPU threads for inference (>= 1).
 * Returns 1 on success, 0 on failure (see creature_last_error). */
CREATURE_API int creature_load(creature_runtime* rt, const char* model_dir_utf8, int threads);

/* Back to the start pose, root at the origin, command zero. */
CREATURE_API void creature_reset(creature_runtime* rt);

/* Command: forward speed (m/s), side speed (m/s), yaw rate (rad/s, left turn = positive). */
CREATURE_API void creature_set_command(creature_runtime* rt, float forward, float side, float yaw_rate);

/* Advance the simulation by dt seconds (any dt; the model runs at 30 fps inside). Returns the number of model frames advanced, or -1 on error. */
CREATURE_API int creature_step(creature_runtime* rt, double dt);

/* Number of joints (0 when no model is loaded) and number of feet / contact values. */
CREATURE_API int creature_joint_count(const creature_runtime* rt);
CREATURE_API int creature_contact_count(const creature_runtime* rt);

/* Joint world positions: x, y, z per joint (3 * count floats). Writes at most max_joints joints and returns the number written. */
CREATURE_API int creature_get_world_joints(const creature_runtime* rt, float* out_xyz, int max_joints);

/* Root position in the world (x, y, z; y = 0) and heading yaw in radians about +Y (0 = facing +Z). Any pointer may be NULL. */
CREATURE_API void creature_get_root(const creature_runtime* rt, float* out_xyz, float* out_yaw);

/* Foot contact values (about 0..1), one per foot. Writes at most max_contacts values and returns the number written. */
CREATURE_API int creature_get_contacts(const creature_runtime* rt, float* out, int max_contacts);

/* Number of non-finite values the model produced so far (0 = healthy). */
CREATURE_API int creature_nan_events(const creature_runtime* rt);

/* Last error text of this handle ("" when none). The pointer stays valid until the next call on this handle. */
CREATURE_API const char* creature_last_error(const creature_runtime* rt);

#ifdef __cplusplus
}
#endif
#endif
