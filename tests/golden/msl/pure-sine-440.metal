#include <metal_stdlib>
using namespace metal;

kernel void tropical_kernel(
    device float*                  output_buffer [[buffer(0)]],
    constant float*                slots         [[buffer(1)]],
    constant long*                 tau           [[buffer(2)]],
    uint s [[thread_position_in_grid]])
{
    if (s >= 256u) { return; }
    const long tau_s = tau[s];
    long sli2 = tau_s;
    const long t0 = (sli2 >> 16L);
    long ti5 = t0;
    const long t1 = (42852281L * ti5);
    long ti6 = t1;
    const long t2 = (sli2 & 65535L);
    long ti7 = t2;
    const long t3 = (42852281L * ti7);
    long ti8 = t3;
    const long t4 = (ti8 >> 16L);
    long ti9 = t4;
    const long t5 = (ti6 + ti9);
    long ti10 = t5;
    const long t6 = (ti10 + 0L);
    long ti13 = t6;
    const long t7 = (ti13 & 4294967295L);
    long ti14 = t7;
    const long t8 = (ti14 + 1073741824L);
    long ti15 = t8;
    const long t9 = (ti15 >> 31L);
    long ti16 = t9;
    const long t10 = (ti16 & 1L);
    long ti17 = t10;
    const long t11 = (2L * ti17);
    long ti18 = t11;
    const long t12 = (1L - ti18);
    long ti19 = t12;
    const long t13 = (ti16 << 31L);
    long ti20 = t13;
    const long t14 = (ti14 - ti20);
    long ti21 = t14;
    const long t15 = (ti21 * ti21);
    long ti22 = t15;
    const long t16 = (ti22 >> 30L);
    long ti23 = t16;
    const long t17 = (ti23 >> 30L);
    long ti24 = t17;
    const long t18 = (61L - ti24);
    long ti25 = t18;
    const long t19 = (ti25 * ti23);
    long ti26 = t19;
    const long t20 = (ti26 >> 30L);
    long ti27 = t20;
    const long t21 = (3864L - ti27);
    long ti28 = t21;
    const long t22 = (ti28 * ti23);
    long ti29 = t22;
    const long t23 = (ti29 >> 30L);
    long ti30 = t23;
    const long t24 = (172272L - ti30);
    long ti31 = t24;
    const long t25 = (ti31 * ti23);
    long ti32 = t25;
    const long t26 = (ti32 >> 30L);
    long ti33 = t26;
    const long t27 = (5026995L - ti33);
    long ti34 = t27;
    const long t28 = (ti34 * ti23);
    long ti35 = t28;
    const long t29 = (ti35 >> 30L);
    long ti36 = t29;
    const long t30 = (85569306L - ti36);
    long ti37 = t30;
    const long t31 = (ti37 * ti23);
    long ti38 = t31;
    const long t32 = (ti38 >> 30L);
    long ti39 = t32;
    const long t33 = (693598668L - ti39);
    long ti40 = t33;
    const long t34 = (ti40 * ti23);
    long ti41 = t34;
    const long t35 = (ti41 >> 30L);
    long ti42 = t35;
    const long t36 = (1686629713L - ti42);
    long ti43 = t36;
    const long t37 = (ti21 * ti43);
    long ti44 = t37;
    const long t38 = (ti44 >> 30L);
    long ti45 = t38;
    const long t39 = (ti19 * ti45);
    long ti46 = t39;
    const float t40 = float(ti46);
    float tf47 = t40;
    const float t41 = (as_type<float>(0x4e800000u) == 0.0f ? 0.0f : (tf47 / as_type<float>(0x4e800000u)));
    float tf48 = t41;
    const float t42 = (tf48 + as_type<float>(0x00000000u));
    float tf49 = t42;
    float slf0 = tf49;
    const float t43 = (as_type<float>(0x00000000u) + slf0);
    output_buffer[s] = t43 * as_type<float>(0x3f800000u);
}
