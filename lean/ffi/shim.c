/*
 * shim.c — Lean-ABI wrappers over engine/c_api/tropical_c.h.
 *
 * Phase 2 of the Lean port: the engine owns the native runtime and DAC
 * directly (no koffi, no bun on the control path). Opaque C handles
 * become Lean external objects whose finalizers call the matching
 * *_free — GC-driven cleanup, same shape as the TS FinalizationRegistry
 * it replaces.
 *
 * Conventions:
 *  - every IO-returning Lean extern gets a trailing `lean_obj_arg world`
 *  - handles are borrowed (`b_lean_obj_arg`) — Lean keeps ownership
 *  - errors surface as Bool/sentinel returns; Lean fetches
 *    tropical_last_error (thread-local; IO runs on the calling thread,
 *    so the read happens before any other FFI call can clobber it)
 */

#include <lean/lean.h>
#include "tropical_c.h"

/* leanc compiles against Lean's hermetic sysroot (no libc headers);
   declare the three libc functions we use. They resolve at link time. */
extern size_t strlen(const char *);
extern void *memcpy(void *, const void *, size_t);
extern void *memset(void *, int, size_t);

/* ── External classes ──────────────────────────────────────────────────────── */

static lean_external_class *g_runtime_class = NULL;
static lean_external_class *g_dac_class = NULL;
static lean_external_class *g_socket_class = NULL;

static void runtime_finalizer(void *h) { tropical_runtime_free(h); }
static void dac_finalizer(void *h) { tropical_dac_free(h); }
static void socket_finalizer(void *h) { tropical_socket_free(h); }
static void noop_foreach(void *mod, b_lean_obj_arg fn) { (void)mod; (void)fn; }

static lean_external_class *runtime_class(void) {
  if (!g_runtime_class)
    g_runtime_class = lean_register_external_class(runtime_finalizer, noop_foreach);
  return g_runtime_class;
}

static lean_external_class *dac_class(void) {
  if (!g_dac_class)
    g_dac_class = lean_register_external_class(dac_finalizer, noop_foreach);
  return g_dac_class;
}

static lean_external_class *socket_class(void) {
  if (!g_socket_class)
    g_socket_class = lean_register_external_class(socket_finalizer, noop_foreach);
  return g_socket_class;
}

static void *unwrap(b_lean_obj_arg o) { return lean_get_external_data(o); }

static lean_obj_res io_err(const char *msg) {
  return lean_io_result_mk_error(lean_mk_io_user_error(lean_mk_string(msg)));
}

/* ── Error channel ─────────────────────────────────────────────────────────── */

LEAN_EXPORT lean_obj_res shim_last_error(lean_obj_arg world) {
  (void)world;
  const char *e = tropical_last_error();
  return lean_io_result_mk_ok(lean_mk_string(e ? e : ""));
}

/* ── Runtime ───────────────────────────────────────────────────────────────── */

LEAN_EXPORT lean_obj_res shim_runtime_new(uint32_t buffer_length, lean_obj_arg world) {
  (void)world;
  tropical_runtime_t r = tropical_runtime_new(buffer_length);
  if (!r) return io_err(tropical_last_error());
  return lean_io_result_mk_ok(lean_alloc_external(runtime_class(), r));
}

LEAN_EXPORT lean_obj_res shim_runtime_load_ir(b_lean_obj_arg rt, b_lean_obj_arg ir,
                                              b_lean_obj_arg manifest, lean_obj_arg world) {
  (void)world;
  const char *i = lean_string_cstr(ir);
  const char *m = lean_string_cstr(manifest);
  bool ok = tropical_runtime_load_ir(unwrap(rt), i, strlen(i), m, strlen(m));
  return lean_io_result_mk_ok(lean_box(ok));
}

/* In-process IR→wasm (build-time tool). Empty ByteArray signals failure;
   Lean reads tropical_last_error. */
LEAN_EXPORT lean_obj_res shim_compile_ir_to_wasm(b_lean_obj_arg ir, lean_obj_arg world) {
  (void)world;
  const char *i = lean_string_cstr(ir);
  size_t out_len = 0;
  uint8_t *buf = tropical_compile_ir_to_wasm(i, strlen(i), &out_len);
  lean_obj_res arr;
  if (buf) {
    arr = lean_alloc_sarray(1, out_len, out_len);
    memcpy(lean_sarray_cptr(arr), buf, out_len);
    tropical_free_buffer(buf);
  } else {
    arr = lean_alloc_sarray(1, 0, 0);
  }
  return lean_io_result_mk_ok(arr);
}

LEAN_EXPORT lean_obj_res shim_runtime_process(b_lean_obj_arg rt, lean_obj_arg world) {
  (void)world;
  tropical_runtime_process(unwrap(rt));
  return lean_io_result_mk_ok(lean_box(0));
}

/* Copy the output buffer (buffer_length doubles) into a fresh ByteArray. */
LEAN_EXPORT lean_obj_res shim_runtime_output_bytes(b_lean_obj_arg rt, lean_obj_arg world) {
  (void)world;
  tropical_runtime_t r = unwrap(rt);
  const double *buf = tropical_runtime_output_buffer(r);
  unsigned int len = tropical_runtime_get_buffer_length(r);
  size_t bytes = (size_t)len * sizeof(double);
  lean_obj_res arr = lean_alloc_sarray(1, bytes, bytes);
  if (buf && bytes) memcpy(lean_sarray_cptr(arr), buf, bytes);
  return lean_io_result_mk_ok(arr);
}

LEAN_EXPORT lean_obj_res shim_runtime_get_buffer_length(b_lean_obj_arg rt, lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box_uint32(tropical_runtime_get_buffer_length(unwrap(rt))));
}

LEAN_EXPORT lean_obj_res shim_runtime_slot_index(b_lean_obj_arg rt, b_lean_obj_arg name,
                                                 lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box_uint32(
      tropical_runtime_slot_index(unwrap(rt), lean_string_cstr(name))));
}

LEAN_EXPORT lean_obj_res shim_runtime_set_slot(b_lean_obj_arg rt, uint32_t idx, double value,
                                               lean_obj_arg world) {
  (void)world;
  tropical_runtime_set_slot(unwrap(rt), idx, value);
  return lean_io_result_mk_ok(lean_box(0));
}

LEAN_EXPORT lean_obj_res shim_runtime_get_slot(b_lean_obj_arg rt, uint32_t idx,
                                               lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box_float(tropical_runtime_get_slot(unwrap(rt), idx)));
}

/* ── DAC ───────────────────────────────────────────────────────────────────── */

LEAN_EXPORT lean_obj_res shim_dac_new_runtime(b_lean_obj_arg rt, uint32_t sample_rate,
                                              uint32_t channels, lean_obj_arg world) {
  (void)world;
  tropical_dac_t d = tropical_dac_new_runtime(unwrap(rt), sample_rate, channels);
  if (!d) return io_err(tropical_last_error());
  return lean_io_result_mk_ok(lean_alloc_external(dac_class(), d));
}

LEAN_EXPORT lean_obj_res shim_dac_start(b_lean_obj_arg dac, lean_obj_arg world) {
  (void)world;
  tropical_dac_start(unwrap(dac));
  return lean_io_result_mk_ok(lean_box(0));
}

LEAN_EXPORT lean_obj_res shim_dac_stop(b_lean_obj_arg dac, lean_obj_arg world) {
  (void)world;
  tropical_dac_stop(unwrap(dac));
  return lean_io_result_mk_ok(lean_box(0));
}

LEAN_EXPORT lean_obj_res shim_dac_is_running(b_lean_obj_arg dac, lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box(tropical_dac_is_running(unwrap(dac))));
}

LEAN_EXPORT lean_obj_res shim_dac_is_reconnecting(b_lean_obj_arg dac, lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box(tropical_dac_is_reconnecting(unwrap(dac))));
}

LEAN_EXPORT lean_obj_res shim_dac_switch_device(b_lean_obj_arg dac, uint32_t device_id,
                                                lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box(tropical_dac_switch_device(unwrap(dac), device_id)));
}

/* Stats — one getter per field; each refetches the (cheap) struct. */

static tropical_dac_stats_t dac_stats(b_lean_obj_arg dac) {
  tropical_dac_stats_t s;
  memset(&s, 0, sizeof s);
  tropical_dac_get_stats(unwrap(dac), &s);
  return s;
}

LEAN_EXPORT lean_obj_res shim_dac_stat_callback_count(b_lean_obj_arg dac, lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box_uint64(dac_stats(dac).callback_count));
}

LEAN_EXPORT lean_obj_res shim_dac_stat_avg_ms(b_lean_obj_arg dac, lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box_float(dac_stats(dac).avg_callback_ms));
}

LEAN_EXPORT lean_obj_res shim_dac_stat_max_ms(b_lean_obj_arg dac, lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box_float(dac_stats(dac).max_callback_ms));
}

LEAN_EXPORT lean_obj_res shim_dac_stat_underruns(b_lean_obj_arg dac, lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box_uint64(dac_stats(dac).underrun_count));
}

LEAN_EXPORT lean_obj_res shim_dac_stat_overruns(b_lean_obj_arg dac, lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box_uint64(dac_stats(dac).overrun_count));
}

/* ── Device enumeration ────────────────────────────────────────────────────── */

LEAN_EXPORT lean_obj_res shim_audio_device_count(lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box_uint32(tropical_audio_device_count()));
}

/* Device IDs as a Lean Array UInt32. */
LEAN_EXPORT lean_obj_res shim_audio_device_ids(lean_obj_arg world) {
  (void)world;
  unsigned int count = tropical_audio_device_count();
  unsigned int ids[256];
  if (count > 256) count = 256;
  tropical_audio_get_device_ids(ids, count);
  lean_obj_res arr = lean_alloc_array(count, count);
  for (unsigned int i = 0; i < count; i++)
    lean_array_set_core(arr, i, lean_box_uint32(ids[i]));
  return lean_io_result_mk_ok(arr);
}

LEAN_EXPORT lean_obj_res shim_audio_device_name(uint32_t device_id, lean_obj_arg world) {
  (void)world;
  tropical_device_info_t info;
  memset(&info, 0, sizeof info);
  if (!tropical_audio_get_device_info(device_id, &info))
    return lean_io_result_mk_ok(lean_mk_string(""));
  info.name[255] = '\0';
  return lean_io_result_mk_ok(lean_mk_string(info.name));
}

LEAN_EXPORT lean_obj_res shim_audio_default_output_device(lean_obj_arg world) {
  (void)world;
  return lean_io_result_mk_ok(lean_box_uint32(tropical_audio_default_output_device()));
}

/* ── Socket endpoint ───────────────────────────────────────────────────────── */

LEAN_EXPORT lean_obj_res shim_socket_listen(b_lean_obj_arg rt, b_lean_obj_arg addr,
                                            lean_obj_arg world) {
  (void)world;
  tropical_socket_t sock = tropical_socket_listen(unwrap(rt), lean_string_cstr(addr));
  if (!sock) return io_err(tropical_last_error());
  return lean_io_result_mk_ok(lean_alloc_external(socket_class(), sock));
}

/* Returns `Option (UInt64 × String)`: none on shutdown, else (client_id, line). */
LEAN_EXPORT lean_obj_res shim_socket_next_control(b_lean_obj_arg s, lean_obj_arg world) {
  (void)world;
  uint64_t cid = 0;
  char *buf = NULL;
  size_t len = 0;
  if (!tropical_socket_next_control(unwrap(s), &cid, &buf, &len)) {
    return lean_io_result_mk_ok(lean_box(0));   /* Option.none */
  }
  lean_object *pair = lean_alloc_ctor(0, 2, 0); /* Prod.mk */
  lean_ctor_set(pair, 0, lean_box_uint64(cid));
  lean_ctor_set(pair, 1, lean_mk_string(buf ? buf : ""));
  tropical_free_buffer((uint8_t *)buf);
  lean_object *some = lean_alloc_ctor(1, 1, 0); /* Option.some */
  lean_ctor_set(some, 0, pair);
  return lean_io_result_mk_ok(some);
}

LEAN_EXPORT lean_obj_res shim_socket_send_response(b_lean_obj_arg s, uint64_t client_id,
                                                   b_lean_obj_arg bytes, lean_obj_arg world) {
  (void)world;
  const char *b = lean_string_cstr(bytes);
  tropical_socket_send_response(unwrap(s), client_id, b, strlen(b));
  return lean_io_result_mk_ok(lean_box(0));
}
