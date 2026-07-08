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
    long ti55 = t0;
    long sli2 = ti55;
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
    const float t9 = float(ti14);
    float tf15 = t9;
    const float t10 = (as_type<float>(0x4f800000u) == 0.0f ? 0.0f : (tf15 / as_type<float>(0x4f800000u)));
    float tf16 = t10;
    const float t11 = (tf16 > as_type<float>(0x00000000u) ? tf16 : as_type<float>(0x00000000u));
    const float t12 = (t11 < as_type<float>(0x3f800000u) ? t11 : as_type<float>(0x3f800000u));
    float tf17 = t12;
    const float t13 = (tf17 * as_type<float>(0x4f800000u));
    float tf18 = t13;
    const long t14 = long(tf18);
    long ti19 = t14;
    const long t15 = (ti19 + 1073741824L);
    long ti20 = t15;
    const long t16 = (ti20 >> 31L);
    long ti21 = t16;
    const long t17 = (ti21 & 1L);
    long ti22 = t17;
    const long t18 = (2L * ti22);
    long ti23 = t18;
    const long t19 = (1L - ti23);
    long ti24 = t19;
    const long t20 = (ti21 << 31L);
    long ti25 = t20;
    const long t21 = (ti19 - ti25);
    long ti26 = t21;
    const long t22 = (ti26 * ti26);
    long ti27 = t22;
    const long t23 = (ti27 >> 30L);
    long ti28 = t23;
    const long t24 = (ti28 >> 30L);
    long ti29 = t24;
    const long t25 = (61L - ti29);
    long ti30 = t25;
    const long t26 = (ti30 * ti28);
    long ti31 = t26;
    const long t27 = (ti31 >> 30L);
    long ti32 = t27;
    const long t28 = (3864L - ti32);
    long ti33 = t28;
    const long t29 = (ti33 * ti28);
    long ti34 = t29;
    const long t30 = (ti34 >> 30L);
    long ti35 = t30;
    const long t31 = (172272L - ti35);
    long ti36 = t31;
    const long t32 = (ti36 * ti28);
    long ti37 = t32;
    const long t33 = (ti37 >> 30L);
    long ti38 = t33;
    const long t34 = (5026995L - ti38);
    long ti39 = t34;
    const long t35 = (ti39 * ti28);
    long ti40 = t35;
    const long t36 = (ti40 >> 30L);
    long ti41 = t36;
    const long t37 = (85569306L - ti41);
    long ti42 = t37;
    const long t38 = (ti42 * ti28);
    long ti43 = t38;
    const long t39 = (ti43 >> 30L);
    long ti44 = t39;
    const long t40 = (693598668L - ti44);
    long ti45 = t40;
    const long t41 = (ti45 * ti28);
    long ti46 = t41;
    const long t42 = (ti46 >> 30L);
    long ti47 = t42;
    const long t43 = (1686629713L - ti47);
    long ti48 = t43;
    const long t44 = (ti26 * ti48);
    long ti49 = t44;
    const long t45 = (ti49 >> 30L);
    long ti50 = t45;
    const long t46 = (ti24 * ti50);
    long ti51 = t46;
    const float t47 = float(ti51);
    float tf52 = t47;
    const float t48 = (as_type<float>(0x4e800000u) == 0.0f ? 0.0f : (tf52 / as_type<float>(0x4e800000u)));
    float tf53 = t48;
    const float t49 = (tf53 + as_type<float>(0x00000000u));
    float tf54 = t49;
    float slf0 = tf54;
    const float t50 = (as_type<float>(0x00000000u) + slf0);
    output_buffer[s] = t50 * as_type<float>(0x3d4ccccdu);
}
