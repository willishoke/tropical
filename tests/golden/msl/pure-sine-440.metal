#include <metal_stdlib>
using namespace metal;

struct TropicalKernelConsts {
    ulong start_sample_index;
    float sample_rate;
    uint  buffer_length;
};

kernel void tropical_kernel(
    device float*                  output_buffer [[buffer(0)]],
    constant float*                slots         [[buffer(1)]],
    constant TropicalKernelConsts& k             [[buffer(2)]],
    uint s [[thread_position_in_grid]])
{
    if (s >= k.buffer_length) { return; }
    const long current_idx = long(k.start_sample_index) + long(s);
    const long t0 = (current_idx << 32L);
    long ti50 = t0;
    long sli2 = ti50;
    const long t1 = (sli2 >> 32L);
    long ti5 = t1;
    const long t2 = (42852281L * ti5);
    long ti6 = t2;
    const long t3 = (sli2 & 4294967295L);
    long ti7 = t3;
    const long t4 = (42852281L * ti7);
    long ti8 = t4;
    const long t5 = (ti8 >> 32L);
    long ti9 = t5;
    const long t6 = (ti6 + ti9);
    long ti10 = t6;
    const long t7 = (ti10 + 0L);
    long ti13 = t7;
    const long t8 = (ti13 & 4294967295L);
    long ti14 = t8;
    const long t9 = (ti14 + 1073741824L);
    long ti15 = t9;
    const long t10 = (ti15 >> 31L);
    long ti16 = t10;
    const long t11 = (ti16 & 1L);
    long ti17 = t11;
    const long t12 = (2L * ti17);
    long ti18 = t12;
    const long t13 = (1L - ti18);
    long ti19 = t13;
    const long t14 = (ti16 << 31L);
    long ti20 = t14;
    const long t15 = (ti14 - ti20);
    long ti21 = t15;
    const long t16 = (ti21 * ti21);
    long ti22 = t16;
    const long t17 = (ti22 >> 30L);
    long ti23 = t17;
    const long t18 = (ti23 >> 30L);
    long ti24 = t18;
    const long t19 = (61L - ti24);
    long ti25 = t19;
    const long t20 = (ti25 * ti23);
    long ti26 = t20;
    const long t21 = (ti26 >> 30L);
    long ti27 = t21;
    const long t22 = (3864L - ti27);
    long ti28 = t22;
    const long t23 = (ti28 * ti23);
    long ti29 = t23;
    const long t24 = (ti29 >> 30L);
    long ti30 = t24;
    const long t25 = (172272L - ti30);
    long ti31 = t25;
    const long t26 = (ti31 * ti23);
    long ti32 = t26;
    const long t27 = (ti32 >> 30L);
    long ti33 = t27;
    const long t28 = (5026995L - ti33);
    long ti34 = t28;
    const long t29 = (ti34 * ti23);
    long ti35 = t29;
    const long t30 = (ti35 >> 30L);
    long ti36 = t30;
    const long t31 = (85569306L - ti36);
    long ti37 = t31;
    const long t32 = (ti37 * ti23);
    long ti38 = t32;
    const long t33 = (ti38 >> 30L);
    long ti39 = t33;
    const long t34 = (693598668L - ti39);
    long ti40 = t34;
    const long t35 = (ti40 * ti23);
    long ti41 = t35;
    const long t36 = (ti41 >> 30L);
    long ti42 = t36;
    const long t37 = (1686629713L - ti42);
    long ti43 = t37;
    const long t38 = (ti21 * ti43);
    long ti44 = t38;
    const long t39 = (ti44 >> 30L);
    long ti45 = t39;
    const long t40 = (ti19 * ti45);
    long ti46 = t40;
    const float t41 = float(ti46);
    float tf47 = t41;
    const float t42 = (as_type<float>(0x4e800000u) == 0.0f ? 0.0f : (tf47 / as_type<float>(0x4e800000u)));
    float tf48 = t42;
    const float t43 = (tf48 + as_type<float>(0x00000000u));
    float tf49 = t43;
    float slf0 = tf49;
    const float t44 = (as_type<float>(0x00000000u) + slf0);
    output_buffer[s] = t44 * as_type<float>(0x3f800000u);
}
