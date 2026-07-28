/**
 * Compatibility quarantine: direct-C tropical_plan_4 metadata loading.
 *
 * This is deliberately not a production-language test. Lean emits plan_5 and
 * owns LLVM generation. A direct C caller may still pair its own closed-form
 * LLVM kernel with a plan_4 manifest; obsolete state fields in that manifest
 * are ignored by the current runtime.
 */

#include "c_api/tropical_c.h"

#include <cmath>
#include <cstdio>
#include <cstring>

int main()
{
  constexpr unsigned int kFrames = 16;
  tropical_runtime_t runtime = tropical_runtime_new(kFrames);
  if (!runtime)
  {
    std::fprintf(stderr, "runtime_new failed: %s\n", tropical_last_error());
    return 1;
  }

  // Current 11-argument kernel ABI, but behavior is closed-form and ignores
  // every state-shaped pointer.
  const char * ir = R"(
define void @kernel(ptr %inputs, ptr %registers, ptr %arrays,
                    ptr %array_sizes, ptr %temps, double %sample_rate,
                    i64 %start_sample_index, ptr %param_ptrs,
                    ptr %output_buffer, i64 %buffer_length, ptr %slots) {
entry:
  br label %loop
loop:
  %i = phi i64 [ 0, %entry ], [ %next, %body ]
  %more = icmp ult i64 %i, %buffer_length
  br i1 %more, label %body, label %done
body:
  %out = getelementptr inbounds double, ptr %output_buffer, i64 %i
  store double 2.500000e-01, ptr %out, align 8
  %next = add i64 %i, 1
  br label %loop
done:
  ret void
}
)";

  // state_init/register_* are historical plan fields. Acceptance of this
  // manifest must not be read as activation: parse_plan4 does not consume them
  // and FlatRuntime allocates no persistent register backing store.
  const char * manifest = R"({
    "schema": "tropical_plan_4",
    "config": {"sampleRate": 44100},
    "register_count": 0,
    "array_slot_sizes": [],
    "output_targets": [],
    "outputs": [],
    "slot_count": 0,
    "slot_names": [],
    "slot_defaults": [],
    "instructions": [],
    "state_init": [123.0],
    "register_names": ["legacy_state"],
    "register_types": ["float"],
    "register_targets": [0],
    "state_reg_offset": 1
  })";

  const bool loaded = tropical_runtime_load_ir(
    runtime, ir, std::strlen(ir), manifest, std::strlen(manifest));
  if (!loaded)
  {
    std::fprintf(stderr, "legacy plan4 metadata load failed: %s\n",
                 tropical_last_error());
    tropical_runtime_free(runtime);
    return 1;
  }

  tropical_runtime_process(runtime);
  const double * output = tropical_runtime_output_buffer(runtime);
  if (!output)
  {
    std::fprintf(stderr, "output buffer is null\n");
    tropical_runtime_free(runtime);
    return 1;
  }
  for (unsigned int i = 0; i < kFrames; ++i)
  {
    if (std::fabs(output[i] - 0.25) > 1e-12)
    {
      std::fprintf(stderr, "sample %u = %.17g, expected 0.25\n", i, output[i]);
      tropical_runtime_free(runtime);
      return 1;
    }
  }

  tropical_runtime_free(runtime);
  std::printf("PASS compat_legacy_plan4_manifest: metadata accepted; "
              "obsolete state keys inert\n");
  return 0;
}
