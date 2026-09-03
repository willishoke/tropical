define void @tropical_kernel(ptr %inputs, ptr %registers, ptr %arrays, ptr %array_sizes, ptr %temps, double %sampleRate, i64 %start_sample_index, ptr %param_ptrs, ptr %output_buffer, i64 %buffer_length, ptr noalias nocapture %slots) {
entry:
  br label %loop_cond

loop_cond:
  %s = phi i64 [ 0, %entry ], [ %s_next, %loop_body ]
  %current_idx = add i64 %start_sample_index, %s
  %loopcond = icmp ult i64 %s, %buffer_length
  br i1 %loopcond, label %loop_body, label %loop_end

loop_body:
  %t0 = getelementptr inbounds double, ptr %slots, i64 1
  %t1 = load double, ptr %t0, align 8
  %t2 = fmul double %t1, %sampleRate
  %t3 = fmul double %t2, 0x41f0000000000000
  %t4 = fptosi double %t3 to i64
  %t5 = sitofp i64 %t4 to double
  %t6 = getelementptr inbounds double, ptr %slots, i64 6
  store double %t5, ptr %t6, align 8
  %t7 = getelementptr inbounds double, ptr %slots, i64 0
  %t8 = load double, ptr %t7, align 8
  %t9 = fmul double %t8, 0x41f0000000000000
  %t10 = fptosi double %t9 to i64
  %t11 = sitofp i64 %t10 to double
  %t12 = getelementptr inbounds double, ptr %slots, i64 7
  store double %t11, ptr %t12, align 8
  %t13 = getelementptr inbounds double, ptr %slots, i64 3
  %t14 = load double, ptr %t13, align 8
  %t15 = fmul double 0x3ff0000000000000, %t14
  %t16 = fmul double 0x401921fb54442d18, %t15
  %t17 = fdiv double %t16, 0x401921fb54442d18
  %t18 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t19 = select i1 %t18, double 0x0000000000000000, double %t17
  %t20 = fmul double %t19, 0x41f0000000000000
  %t21 = fdiv double %t20, %sampleRate
  %t22 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t23 = select i1 %t22, double 0x0000000000000000, double %t21
  %t24 = getelementptr inbounds double, ptr %slots, i64 3
  %t25 = load double, ptr %t24, align 8
  %t26 = fmul double 0x4000000000000000, %t25
  %t27 = fmul double 0x401921fb54442d18, %t26
  %t28 = fdiv double %t27, 0x401921fb54442d18
  %t29 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t30 = select i1 %t29, double 0x0000000000000000, double %t28
  %t31 = fmul double %t30, 0x41f0000000000000
  %t32 = fdiv double %t31, %sampleRate
  %t33 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t34 = select i1 %t33, double 0x0000000000000000, double %t32
  %t35 = getelementptr inbounds double, ptr %slots, i64 3
  %t36 = load double, ptr %t35, align 8
  %t37 = fmul double 0x4008000000000000, %t36
  %t38 = fmul double 0x401921fb54442d18, %t37
  %t39 = fdiv double %t38, 0x401921fb54442d18
  %t40 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t41 = select i1 %t40, double 0x0000000000000000, double %t39
  %t42 = fmul double %t41, 0x41f0000000000000
  %t43 = fdiv double %t42, %sampleRate
  %t44 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t45 = select i1 %t44, double 0x0000000000000000, double %t43
  %t46 = getelementptr inbounds double, ptr %slots, i64 3
  %t47 = load double, ptr %t46, align 8
  %t48 = fmul double 0x4010000000000000, %t47
  %t49 = fmul double 0x401921fb54442d18, %t48
  %t50 = fdiv double %t49, 0x401921fb54442d18
  %t51 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t52 = select i1 %t51, double 0x0000000000000000, double %t50
  %t53 = fmul double %t52, 0x41f0000000000000
  %t54 = fdiv double %t53, %sampleRate
  %t55 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t56 = select i1 %t55, double 0x0000000000000000, double %t54
  %t57 = getelementptr inbounds double, ptr %slots, i64 3
  %t58 = load double, ptr %t57, align 8
  %t59 = fmul double 0x4014000000000000, %t58
  %t60 = fmul double 0x401921fb54442d18, %t59
  %t61 = fdiv double %t60, 0x401921fb54442d18
  %t62 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t63 = select i1 %t62, double 0x0000000000000000, double %t61
  %t64 = fmul double %t63, 0x41f0000000000000
  %t65 = fdiv double %t64, %sampleRate
  %t66 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t67 = select i1 %t66, double 0x0000000000000000, double %t65
  %t68 = getelementptr inbounds double, ptr %slots, i64 3
  %t69 = load double, ptr %t68, align 8
  %t70 = fmul double 0x4018000000000000, %t69
  %t71 = fmul double 0x401921fb54442d18, %t70
  %t72 = fdiv double %t71, 0x401921fb54442d18
  %t73 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t74 = select i1 %t73, double 0x0000000000000000, double %t72
  %t75 = fmul double %t74, 0x41f0000000000000
  %t76 = fdiv double %t75, %sampleRate
  %t77 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t78 = select i1 %t77, double 0x0000000000000000, double %t76
  %t79 = getelementptr inbounds double, ptr %slots, i64 3
  %t80 = load double, ptr %t79, align 8
  %t81 = fmul double 0x401c000000000000, %t80
  %t82 = fmul double 0x401921fb54442d18, %t81
  %t83 = fdiv double %t82, 0x401921fb54442d18
  %t84 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t85 = select i1 %t84, double 0x0000000000000000, double %t83
  %t86 = fmul double %t85, 0x41f0000000000000
  %t87 = fdiv double %t86, %sampleRate
  %t88 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t89 = select i1 %t88, double 0x0000000000000000, double %t87
  %t90 = getelementptr inbounds double, ptr %slots, i64 3
  %t91 = load double, ptr %t90, align 8
  %t92 = fmul double 0x4020000000000000, %t91
  %t93 = fmul double 0x401921fb54442d18, %t92
  %t94 = fdiv double %t93, 0x401921fb54442d18
  %t95 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t96 = select i1 %t95, double 0x0000000000000000, double %t94
  %t97 = fmul double %t96, 0x41f0000000000000
  %t98 = fdiv double %t97, %sampleRate
  %t99 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t100 = select i1 %t99, double 0x0000000000000000, double %t98
  %t101 = getelementptr inbounds double, ptr %slots, i64 3
  %t102 = load double, ptr %t101, align 8
  %t103 = fmul double 0x4022000000000000, %t102
  %t104 = fmul double 0x401921fb54442d18, %t103
  %t105 = fdiv double %t104, 0x401921fb54442d18
  %t106 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t107 = select i1 %t106, double 0x0000000000000000, double %t105
  %t108 = fmul double %t107, 0x41f0000000000000
  %t109 = fdiv double %t108, %sampleRate
  %t110 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t111 = select i1 %t110, double 0x0000000000000000, double %t109
  %t112 = getelementptr inbounds double, ptr %slots, i64 3
  %t113 = load double, ptr %t112, align 8
  %t114 = fmul double 0x4024000000000000, %t113
  %t115 = fmul double 0x401921fb54442d18, %t114
  %t116 = fdiv double %t115, 0x401921fb54442d18
  %t117 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t118 = select i1 %t117, double 0x0000000000000000, double %t116
  %t119 = fmul double %t118, 0x41f0000000000000
  %t120 = fdiv double %t119, %sampleRate
  %t121 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t122 = select i1 %t121, double 0x0000000000000000, double %t120
  %t123 = getelementptr inbounds double, ptr %slots, i64 3
  %t124 = load double, ptr %t123, align 8
  %t125 = fmul double 0x4026000000000000, %t124
  %t126 = fmul double 0x401921fb54442d18, %t125
  %t127 = fdiv double %t126, 0x401921fb54442d18
  %t128 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t129 = select i1 %t128, double 0x0000000000000000, double %t127
  %t130 = fmul double %t129, 0x41f0000000000000
  %t131 = fdiv double %t130, %sampleRate
  %t132 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t133 = select i1 %t132, double 0x0000000000000000, double %t131
  %t134 = getelementptr inbounds double, ptr %slots, i64 3
  %t135 = load double, ptr %t134, align 8
  %t136 = fmul double 0x4028000000000000, %t135
  %t137 = fmul double 0x401921fb54442d18, %t136
  %t138 = fdiv double %t137, 0x401921fb54442d18
  %t139 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t140 = select i1 %t139, double 0x0000000000000000, double %t138
  %t141 = fmul double %t140, 0x41f0000000000000
  %t142 = fdiv double %t141, %sampleRate
  %t143 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t144 = select i1 %t143, double 0x0000000000000000, double %t142
  %t145 = getelementptr inbounds double, ptr %slots, i64 3
  %t146 = load double, ptr %t145, align 8
  %t147 = fmul double 0x402a000000000000, %t146
  %t148 = fmul double 0x401921fb54442d18, %t147
  %t149 = fdiv double %t148, 0x401921fb54442d18
  %t150 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t151 = select i1 %t150, double 0x0000000000000000, double %t149
  %t152 = fmul double %t151, 0x41f0000000000000
  %t153 = fdiv double %t152, %sampleRate
  %t154 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t155 = select i1 %t154, double 0x0000000000000000, double %t153
  %t156 = getelementptr inbounds double, ptr %slots, i64 3
  %t157 = load double, ptr %t156, align 8
  %t158 = fmul double 0x402c000000000000, %t157
  %t159 = fmul double 0x401921fb54442d18, %t158
  %t160 = fdiv double %t159, 0x401921fb54442d18
  %t161 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t162 = select i1 %t161, double 0x0000000000000000, double %t160
  %t163 = fmul double %t162, 0x41f0000000000000
  %t164 = fdiv double %t163, %sampleRate
  %t165 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t166 = select i1 %t165, double 0x0000000000000000, double %t164
  %t167 = getelementptr inbounds double, ptr %slots, i64 3
  %t168 = load double, ptr %t167, align 8
  %t169 = fmul double 0x402e000000000000, %t168
  %t170 = fmul double 0x401921fb54442d18, %t169
  %t171 = fdiv double %t170, 0x401921fb54442d18
  %t172 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t173 = select i1 %t172, double 0x0000000000000000, double %t171
  %t174 = fmul double %t173, 0x41f0000000000000
  %t175 = fdiv double %t174, %sampleRate
  %t176 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t177 = select i1 %t176, double 0x0000000000000000, double %t175
  %t178 = getelementptr inbounds double, ptr %slots, i64 3
  %t179 = load double, ptr %t178, align 8
  %t180 = fmul double 0x4030000000000000, %t179
  %t181 = fmul double 0x401921fb54442d18, %t180
  %t182 = fdiv double %t181, 0x401921fb54442d18
  %t183 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t184 = select i1 %t183, double 0x0000000000000000, double %t182
  %t185 = fmul double %t184, 0x41f0000000000000
  %t186 = fdiv double %t185, %sampleRate
  %t187 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t188 = select i1 %t187, double 0x0000000000000000, double %t186
  %t189 = getelementptr inbounds ptr, ptr %arrays, i64 0
  %t190 = load ptr, ptr %t189, align 8
  %t191 = bitcast double %t23 to i64
  %t192 = getelementptr inbounds i64, ptr %t190, i64 0
  store i64 %t191, ptr %t192, align 8
  %t193 = bitcast double %t34 to i64
  %t194 = getelementptr inbounds i64, ptr %t190, i64 1
  store i64 %t193, ptr %t194, align 8
  %t195 = bitcast double %t45 to i64
  %t196 = getelementptr inbounds i64, ptr %t190, i64 2
  store i64 %t195, ptr %t196, align 8
  %t197 = bitcast double %t56 to i64
  %t198 = getelementptr inbounds i64, ptr %t190, i64 3
  store i64 %t197, ptr %t198, align 8
  %t199 = bitcast double %t67 to i64
  %t200 = getelementptr inbounds i64, ptr %t190, i64 4
  store i64 %t199, ptr %t200, align 8
  %t201 = bitcast double %t78 to i64
  %t202 = getelementptr inbounds i64, ptr %t190, i64 5
  store i64 %t201, ptr %t202, align 8
  %t203 = bitcast double %t89 to i64
  %t204 = getelementptr inbounds i64, ptr %t190, i64 6
  store i64 %t203, ptr %t204, align 8
  %t205 = bitcast double %t100 to i64
  %t206 = getelementptr inbounds i64, ptr %t190, i64 7
  store i64 %t205, ptr %t206, align 8
  %t207 = bitcast double %t111 to i64
  %t208 = getelementptr inbounds i64, ptr %t190, i64 8
  store i64 %t207, ptr %t208, align 8
  %t209 = bitcast double %t122 to i64
  %t210 = getelementptr inbounds i64, ptr %t190, i64 9
  store i64 %t209, ptr %t210, align 8
  %t211 = bitcast double %t133 to i64
  %t212 = getelementptr inbounds i64, ptr %t190, i64 10
  store i64 %t211, ptr %t212, align 8
  %t213 = bitcast double %t144 to i64
  %t214 = getelementptr inbounds i64, ptr %t190, i64 11
  store i64 %t213, ptr %t214, align 8
  %t215 = bitcast double %t155 to i64
  %t216 = getelementptr inbounds i64, ptr %t190, i64 12
  store i64 %t215, ptr %t216, align 8
  %t217 = bitcast double %t166 to i64
  %t218 = getelementptr inbounds i64, ptr %t190, i64 13
  store i64 %t217, ptr %t218, align 8
  %t219 = bitcast double %t177 to i64
  %t220 = getelementptr inbounds i64, ptr %t190, i64 14
  store i64 %t219, ptr %t220, align 8
  %t221 = bitcast double %t188 to i64
  %t222 = getelementptr inbounds i64, ptr %t190, i64 15
  store i64 %t221, ptr %t222, align 8
  %t223 = getelementptr inbounds double, ptr %slots, i64 4
  %t224 = load double, ptr %t223, align 8
  %t225 = fmul double %t224, 0x3ff6666666666666
  %t226 = getelementptr inbounds double, ptr %slots, i64 4
  %t227 = load double, ptr %t226, align 8
  %t228 = fmul double %t227, 0x3ffccccccccccccd
  %t229 = getelementptr inbounds double, ptr %slots, i64 4
  %t230 = load double, ptr %t229, align 8
  %t231 = fmul double %t230, 0x400199999999999a
  %t232 = getelementptr inbounds double, ptr %slots, i64 4
  %t233 = load double, ptr %t232, align 8
  %t234 = fmul double %t233, 0x4004cccccccccccd
  %t235 = getelementptr inbounds double, ptr %slots, i64 4
  %t236 = load double, ptr %t235, align 8
  %t237 = fmul double %t236, 0x4008000000000000
  %t238 = getelementptr inbounds double, ptr %slots, i64 4
  %t239 = load double, ptr %t238, align 8
  %t240 = fmul double %t239, 0x400b333333333333
  %t241 = getelementptr inbounds double, ptr %slots, i64 4
  %t242 = load double, ptr %t241, align 8
  %t243 = fmul double %t242, 0x400e666666666666
  %t244 = getelementptr inbounds double, ptr %slots, i64 4
  %t245 = load double, ptr %t244, align 8
  %t246 = fmul double %t245, 0x4010cccccccccccd
  %t247 = getelementptr inbounds double, ptr %slots, i64 4
  %t248 = load double, ptr %t247, align 8
  %t249 = fmul double %t248, 0x4012666666666666
  %t250 = getelementptr inbounds double, ptr %slots, i64 4
  %t251 = load double, ptr %t250, align 8
  %t252 = fmul double %t251, 0x4014000000000000
  %t253 = getelementptr inbounds double, ptr %slots, i64 4
  %t254 = load double, ptr %t253, align 8
  %t255 = fmul double %t254, 0x401599999999999a
  %t256 = getelementptr inbounds double, ptr %slots, i64 4
  %t257 = load double, ptr %t256, align 8
  %t258 = fmul double %t257, 0x4017333333333333
  %t259 = getelementptr inbounds double, ptr %slots, i64 4
  %t260 = load double, ptr %t259, align 8
  %t261 = fmul double %t260, 0x4018cccccccccccd
  %t262 = getelementptr inbounds double, ptr %slots, i64 4
  %t263 = load double, ptr %t262, align 8
  %t264 = fmul double %t263, 0x401a666666666666
  %t265 = getelementptr inbounds double, ptr %slots, i64 4
  %t266 = load double, ptr %t265, align 8
  %t267 = fmul double %t266, 0x401c000000000000
  %t268 = getelementptr inbounds double, ptr %slots, i64 4
  %t269 = load double, ptr %t268, align 8
  %t270 = fmul double %t269, 0x401d99999999999a
  %t271 = getelementptr inbounds ptr, ptr %arrays, i64 1
  %t272 = load ptr, ptr %t271, align 8
  %t273 = bitcast double %t225 to i64
  %t274 = getelementptr inbounds i64, ptr %t272, i64 0
  store i64 %t273, ptr %t274, align 8
  %t275 = bitcast double %t228 to i64
  %t276 = getelementptr inbounds i64, ptr %t272, i64 1
  store i64 %t275, ptr %t276, align 8
  %t277 = bitcast double %t231 to i64
  %t278 = getelementptr inbounds i64, ptr %t272, i64 2
  store i64 %t277, ptr %t278, align 8
  %t279 = bitcast double %t234 to i64
  %t280 = getelementptr inbounds i64, ptr %t272, i64 3
  store i64 %t279, ptr %t280, align 8
  %t281 = bitcast double %t237 to i64
  %t282 = getelementptr inbounds i64, ptr %t272, i64 4
  store i64 %t281, ptr %t282, align 8
  %t283 = bitcast double %t240 to i64
  %t284 = getelementptr inbounds i64, ptr %t272, i64 5
  store i64 %t283, ptr %t284, align 8
  %t285 = bitcast double %t243 to i64
  %t286 = getelementptr inbounds i64, ptr %t272, i64 6
  store i64 %t285, ptr %t286, align 8
  %t287 = bitcast double %t246 to i64
  %t288 = getelementptr inbounds i64, ptr %t272, i64 7
  store i64 %t287, ptr %t288, align 8
  %t289 = bitcast double %t249 to i64
  %t290 = getelementptr inbounds i64, ptr %t272, i64 8
  store i64 %t289, ptr %t290, align 8
  %t291 = bitcast double %t252 to i64
  %t292 = getelementptr inbounds i64, ptr %t272, i64 9
  store i64 %t291, ptr %t292, align 8
  %t293 = bitcast double %t255 to i64
  %t294 = getelementptr inbounds i64, ptr %t272, i64 10
  store i64 %t293, ptr %t294, align 8
  %t295 = bitcast double %t258 to i64
  %t296 = getelementptr inbounds i64, ptr %t272, i64 11
  store i64 %t295, ptr %t296, align 8
  %t297 = bitcast double %t261 to i64
  %t298 = getelementptr inbounds i64, ptr %t272, i64 12
  store i64 %t297, ptr %t298, align 8
  %t299 = bitcast double %t264 to i64
  %t300 = getelementptr inbounds i64, ptr %t272, i64 13
  store i64 %t299, ptr %t300, align 8
  %t301 = bitcast double %t267 to i64
  %t302 = getelementptr inbounds i64, ptr %t272, i64 14
  store i64 %t301, ptr %t302, align 8
  %t303 = bitcast double %t270 to i64
  %t304 = getelementptr inbounds i64, ptr %t272, i64 15
  store i64 %t303, ptr %t304, align 8
  %t305 = fptosi double %t3 to i64
  %t306 = sitofp i64 %t305 to double
  %t307 = getelementptr inbounds double, ptr %slots, i64 8
  store double %t306, ptr %t307, align 8
  %t308 = fptosi double %t9 to i64
  %t309 = sitofp i64 %t308 to double
  %t310 = getelementptr inbounds double, ptr %slots, i64 9
  store double %t309, ptr %t310, align 8
  %t311 = getelementptr inbounds double, ptr %output_buffer, i64 %s
  store double 0x0000000000000000, ptr %t311, align 8
  %s_next = add i64 %s, 1
  br label %loop_cond

loop_end:
  ret void
}

declare double @llvm.sqrt.f64(double)
declare double @llvm.floor.f64(double)
declare double @llvm.ceil.f64(double)
declare double @llvm.round.f64(double)
declare double @llvm.fabs.f64(double)
