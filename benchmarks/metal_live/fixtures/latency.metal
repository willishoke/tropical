#include <metal_stdlib>
using namespace metal;

struct TropicalKernelConsts {
    ulong start_sample_index;
    float sample_rate;
    uint buffer_length;
};

kernel void tropical_kernel(
    device float* output_buffer [[buffer(0)]],
    constant float* slots [[buffer(1)]],
    constant TropicalKernelConsts& k [[buffer(2)]],
    uint s [[thread_position_in_grid]])
{
    if (s >= k.buffer_length) { return; }
    output_buffer[s] = slots[0];
}
