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
  %t6 = getelementptr inbounds double, ptr %slots, i64 20
  store double %t5, ptr %t6, align 8
  %t7 = getelementptr inbounds double, ptr %slots, i64 0
  %t8 = load double, ptr %t7, align 8
  %t9 = fmul double %t8, 0x41f0000000000000
  %t10 = fptosi double %t9 to i64
  %t11 = sitofp i64 %t10 to double
  %t12 = getelementptr inbounds double, ptr %slots, i64 21
  store double %t11, ptr %t12, align 8
  %t13 = getelementptr inbounds double, ptr %slots, i64 6
  %t14 = load double, ptr %t13, align 8
  %t15 = getelementptr inbounds double, ptr %slots, i64 5
  %t16 = load double, ptr %t15, align 8
  %t17 = fsub double %t14, %t16
  %t18 = getelementptr inbounds double, ptr %slots, i64 22
  store double %t17, ptr %t18, align 8
  %t19 = getelementptr inbounds double, ptr %slots, i64 13
  %t20 = load double, ptr %t19, align 8
  %t21 = getelementptr inbounds double, ptr %slots, i64 12
  %t22 = load double, ptr %t21, align 8
  %t23 = fsub double %t20, %t22
  %t24 = getelementptr inbounds double, ptr %slots, i64 23
  store double %t23, ptr %t24, align 8
  %t25 = getelementptr inbounds double, ptr %slots, i64 6
  %t26 = load double, ptr %t25, align 8
  %t27 = getelementptr inbounds double, ptr %slots, i64 5
  %t28 = load double, ptr %t27, align 8
  %t29 = fsub double %t26, %t28
  %t30 = getelementptr inbounds double, ptr %slots, i64 24
  store double %t29, ptr %t30, align 8
  %t31 = getelementptr inbounds double, ptr %slots, i64 3
  %t32 = load double, ptr %t31, align 8
  %t33 = fmul double 0x3ff0000000000000, %t32
  %t34 = fmul double 0x401921fb54442d18, %t33
  %t35 = fdiv double %t34, 0x401921fb54442d18
  %t36 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t37 = select i1 %t36, double 0x0000000000000000, double %t35
  %t38 = fmul double %t37, 0x41f0000000000000
  %t39 = fdiv double %t38, %sampleRate
  %t40 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t41 = select i1 %t40, double 0x0000000000000000, double %t39
  %t42 = getelementptr inbounds double, ptr %slots, i64 25
  store double %t41, ptr %t42, align 8
  %t43 = getelementptr inbounds double, ptr %slots, i64 3
  %t44 = load double, ptr %t43, align 8
  %t45 = fmul double 0x4000000000000000, %t44
  %t46 = fmul double 0x401921fb54442d18, %t45
  %t47 = fdiv double %t46, 0x401921fb54442d18
  %t48 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t49 = select i1 %t48, double 0x0000000000000000, double %t47
  %t50 = fmul double %t49, 0x41f0000000000000
  %t51 = fdiv double %t50, %sampleRate
  %t52 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t53 = select i1 %t52, double 0x0000000000000000, double %t51
  %t54 = getelementptr inbounds double, ptr %slots, i64 26
  store double %t53, ptr %t54, align 8
  %t55 = getelementptr inbounds double, ptr %slots, i64 3
  %t56 = load double, ptr %t55, align 8
  %t57 = fmul double 0x4008000000000000, %t56
  %t58 = fmul double 0x401921fb54442d18, %t57
  %t59 = fdiv double %t58, 0x401921fb54442d18
  %t60 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t61 = select i1 %t60, double 0x0000000000000000, double %t59
  %t62 = fmul double %t61, 0x41f0000000000000
  %t63 = fdiv double %t62, %sampleRate
  %t64 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t65 = select i1 %t64, double 0x0000000000000000, double %t63
  %t66 = getelementptr inbounds double, ptr %slots, i64 27
  store double %t65, ptr %t66, align 8
  %t67 = getelementptr inbounds double, ptr %slots, i64 3
  %t68 = load double, ptr %t67, align 8
  %t69 = fmul double 0x4010000000000000, %t68
  %t70 = fmul double 0x401921fb54442d18, %t69
  %t71 = fdiv double %t70, 0x401921fb54442d18
  %t72 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t73 = select i1 %t72, double 0x0000000000000000, double %t71
  %t74 = fmul double %t73, 0x41f0000000000000
  %t75 = fdiv double %t74, %sampleRate
  %t76 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t77 = select i1 %t76, double 0x0000000000000000, double %t75
  %t78 = getelementptr inbounds double, ptr %slots, i64 28
  store double %t77, ptr %t78, align 8
  %t79 = getelementptr inbounds double, ptr %slots, i64 3
  %t80 = load double, ptr %t79, align 8
  %t81 = fmul double 0x4014000000000000, %t80
  %t82 = fmul double 0x401921fb54442d18, %t81
  %t83 = fdiv double %t82, 0x401921fb54442d18
  %t84 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t85 = select i1 %t84, double 0x0000000000000000, double %t83
  %t86 = fmul double %t85, 0x41f0000000000000
  %t87 = fdiv double %t86, %sampleRate
  %t88 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t89 = select i1 %t88, double 0x0000000000000000, double %t87
  %t90 = getelementptr inbounds double, ptr %slots, i64 29
  store double %t89, ptr %t90, align 8
  %t91 = getelementptr inbounds double, ptr %slots, i64 3
  %t92 = load double, ptr %t91, align 8
  %t93 = fmul double 0x4018000000000000, %t92
  %t94 = fmul double 0x401921fb54442d18, %t93
  %t95 = fdiv double %t94, 0x401921fb54442d18
  %t96 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t97 = select i1 %t96, double 0x0000000000000000, double %t95
  %t98 = fmul double %t97, 0x41f0000000000000
  %t99 = fdiv double %t98, %sampleRate
  %t100 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t101 = select i1 %t100, double 0x0000000000000000, double %t99
  %t102 = getelementptr inbounds double, ptr %slots, i64 30
  store double %t101, ptr %t102, align 8
  %t103 = getelementptr inbounds double, ptr %slots, i64 3
  %t104 = load double, ptr %t103, align 8
  %t105 = fmul double 0x401c000000000000, %t104
  %t106 = fmul double 0x401921fb54442d18, %t105
  %t107 = fdiv double %t106, 0x401921fb54442d18
  %t108 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t109 = select i1 %t108, double 0x0000000000000000, double %t107
  %t110 = fmul double %t109, 0x41f0000000000000
  %t111 = fdiv double %t110, %sampleRate
  %t112 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t113 = select i1 %t112, double 0x0000000000000000, double %t111
  %t114 = getelementptr inbounds double, ptr %slots, i64 31
  store double %t113, ptr %t114, align 8
  %t115 = getelementptr inbounds double, ptr %slots, i64 3
  %t116 = load double, ptr %t115, align 8
  %t117 = fmul double 0x4020000000000000, %t116
  %t118 = fmul double 0x401921fb54442d18, %t117
  %t119 = fdiv double %t118, 0x401921fb54442d18
  %t120 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t121 = select i1 %t120, double 0x0000000000000000, double %t119
  %t122 = fmul double %t121, 0x41f0000000000000
  %t123 = fdiv double %t122, %sampleRate
  %t124 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t125 = select i1 %t124, double 0x0000000000000000, double %t123
  %t126 = getelementptr inbounds double, ptr %slots, i64 32
  store double %t125, ptr %t126, align 8
  %t127 = getelementptr inbounds double, ptr %slots, i64 3
  %t128 = load double, ptr %t127, align 8
  %t129 = fmul double 0x4022000000000000, %t128
  %t130 = fmul double 0x401921fb54442d18, %t129
  %t131 = fdiv double %t130, 0x401921fb54442d18
  %t132 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t133 = select i1 %t132, double 0x0000000000000000, double %t131
  %t134 = fmul double %t133, 0x41f0000000000000
  %t135 = fdiv double %t134, %sampleRate
  %t136 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t137 = select i1 %t136, double 0x0000000000000000, double %t135
  %t138 = getelementptr inbounds double, ptr %slots, i64 33
  store double %t137, ptr %t138, align 8
  %t139 = getelementptr inbounds double, ptr %slots, i64 3
  %t140 = load double, ptr %t139, align 8
  %t141 = fmul double 0x4024000000000000, %t140
  %t142 = fmul double 0x401921fb54442d18, %t141
  %t143 = fdiv double %t142, 0x401921fb54442d18
  %t144 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t145 = select i1 %t144, double 0x0000000000000000, double %t143
  %t146 = fmul double %t145, 0x41f0000000000000
  %t147 = fdiv double %t146, %sampleRate
  %t148 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t149 = select i1 %t148, double 0x0000000000000000, double %t147
  %t150 = getelementptr inbounds double, ptr %slots, i64 34
  store double %t149, ptr %t150, align 8
  %t151 = getelementptr inbounds double, ptr %slots, i64 3
  %t152 = load double, ptr %t151, align 8
  %t153 = fmul double 0x4026000000000000, %t152
  %t154 = fmul double 0x401921fb54442d18, %t153
  %t155 = fdiv double %t154, 0x401921fb54442d18
  %t156 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t157 = select i1 %t156, double 0x0000000000000000, double %t155
  %t158 = fmul double %t157, 0x41f0000000000000
  %t159 = fdiv double %t158, %sampleRate
  %t160 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t161 = select i1 %t160, double 0x0000000000000000, double %t159
  %t162 = getelementptr inbounds double, ptr %slots, i64 35
  store double %t161, ptr %t162, align 8
  %t163 = getelementptr inbounds double, ptr %slots, i64 3
  %t164 = load double, ptr %t163, align 8
  %t165 = fmul double 0x4028000000000000, %t164
  %t166 = fmul double 0x401921fb54442d18, %t165
  %t167 = fdiv double %t166, 0x401921fb54442d18
  %t168 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t169 = select i1 %t168, double 0x0000000000000000, double %t167
  %t170 = fmul double %t169, 0x41f0000000000000
  %t171 = fdiv double %t170, %sampleRate
  %t172 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t173 = select i1 %t172, double 0x0000000000000000, double %t171
  %t174 = getelementptr inbounds double, ptr %slots, i64 36
  store double %t173, ptr %t174, align 8
  %t175 = getelementptr inbounds double, ptr %slots, i64 3
  %t176 = load double, ptr %t175, align 8
  %t177 = fmul double 0x402a000000000000, %t176
  %t178 = fmul double 0x401921fb54442d18, %t177
  %t179 = fdiv double %t178, 0x401921fb54442d18
  %t180 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t181 = select i1 %t180, double 0x0000000000000000, double %t179
  %t182 = fmul double %t181, 0x41f0000000000000
  %t183 = fdiv double %t182, %sampleRate
  %t184 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t185 = select i1 %t184, double 0x0000000000000000, double %t183
  %t186 = getelementptr inbounds double, ptr %slots, i64 37
  store double %t185, ptr %t186, align 8
  %t187 = getelementptr inbounds double, ptr %slots, i64 3
  %t188 = load double, ptr %t187, align 8
  %t189 = fmul double 0x402c000000000000, %t188
  %t190 = fmul double 0x401921fb54442d18, %t189
  %t191 = fdiv double %t190, 0x401921fb54442d18
  %t192 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t193 = select i1 %t192, double 0x0000000000000000, double %t191
  %t194 = fmul double %t193, 0x41f0000000000000
  %t195 = fdiv double %t194, %sampleRate
  %t196 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t197 = select i1 %t196, double 0x0000000000000000, double %t195
  %t198 = getelementptr inbounds double, ptr %slots, i64 38
  store double %t197, ptr %t198, align 8
  %t199 = getelementptr inbounds double, ptr %slots, i64 3
  %t200 = load double, ptr %t199, align 8
  %t201 = fmul double 0x402e000000000000, %t200
  %t202 = fmul double 0x401921fb54442d18, %t201
  %t203 = fdiv double %t202, 0x401921fb54442d18
  %t204 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t205 = select i1 %t204, double 0x0000000000000000, double %t203
  %t206 = fmul double %t205, 0x41f0000000000000
  %t207 = fdiv double %t206, %sampleRate
  %t208 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t209 = select i1 %t208, double 0x0000000000000000, double %t207
  %t210 = getelementptr inbounds double, ptr %slots, i64 39
  store double %t209, ptr %t210, align 8
  %t211 = getelementptr inbounds double, ptr %slots, i64 3
  %t212 = load double, ptr %t211, align 8
  %t213 = fmul double 0x4030000000000000, %t212
  %t214 = fmul double 0x401921fb54442d18, %t213
  %t215 = fdiv double %t214, 0x401921fb54442d18
  %t216 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t217 = select i1 %t216, double 0x0000000000000000, double %t215
  %t218 = fmul double %t217, 0x41f0000000000000
  %t219 = fdiv double %t218, %sampleRate
  %t220 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t221 = select i1 %t220, double 0x0000000000000000, double %t219
  %t222 = getelementptr inbounds double, ptr %slots, i64 40
  store double %t221, ptr %t222, align 8
  %t223 = getelementptr inbounds double, ptr %slots, i64 4
  %t224 = load double, ptr %t223, align 8
  %t225 = fmul double %t224, 0x3ff6666666666666
  %t226 = fneg double %t225
  %t227 = fneg double %t226
  %t228 = getelementptr inbounds double, ptr %slots, i64 41
  store double %t227, ptr %t228, align 8
  %t229 = getelementptr inbounds double, ptr %slots, i64 4
  %t230 = load double, ptr %t229, align 8
  %t231 = fmul double %t230, 0x3ffccccccccccccd
  %t232 = fneg double %t231
  %t233 = fneg double %t232
  %t234 = getelementptr inbounds double, ptr %slots, i64 42
  store double %t233, ptr %t234, align 8
  %t235 = getelementptr inbounds double, ptr %slots, i64 4
  %t236 = load double, ptr %t235, align 8
  %t237 = fmul double %t236, 0x400199999999999a
  %t238 = fneg double %t237
  %t239 = fneg double %t238
  %t240 = getelementptr inbounds double, ptr %slots, i64 43
  store double %t239, ptr %t240, align 8
  %t241 = getelementptr inbounds double, ptr %slots, i64 4
  %t242 = load double, ptr %t241, align 8
  %t243 = fmul double %t242, 0x4004cccccccccccd
  %t244 = fneg double %t243
  %t245 = fneg double %t244
  %t246 = getelementptr inbounds double, ptr %slots, i64 44
  store double %t245, ptr %t246, align 8
  %t247 = getelementptr inbounds double, ptr %slots, i64 4
  %t248 = load double, ptr %t247, align 8
  %t249 = fmul double %t248, 0x4008000000000000
  %t250 = fneg double %t249
  %t251 = fneg double %t250
  %t252 = getelementptr inbounds double, ptr %slots, i64 45
  store double %t251, ptr %t252, align 8
  %t253 = getelementptr inbounds double, ptr %slots, i64 4
  %t254 = load double, ptr %t253, align 8
  %t255 = fmul double %t254, 0x400b333333333333
  %t256 = fneg double %t255
  %t257 = fneg double %t256
  %t258 = getelementptr inbounds double, ptr %slots, i64 46
  store double %t257, ptr %t258, align 8
  %t259 = getelementptr inbounds double, ptr %slots, i64 4
  %t260 = load double, ptr %t259, align 8
  %t261 = fmul double %t260, 0x400e666666666666
  %t262 = fneg double %t261
  %t263 = fneg double %t262
  %t264 = getelementptr inbounds double, ptr %slots, i64 47
  store double %t263, ptr %t264, align 8
  %t265 = getelementptr inbounds double, ptr %slots, i64 4
  %t266 = load double, ptr %t265, align 8
  %t267 = fmul double %t266, 0x4010cccccccccccd
  %t268 = fneg double %t267
  %t269 = fneg double %t268
  %t270 = getelementptr inbounds double, ptr %slots, i64 48
  store double %t269, ptr %t270, align 8
  %t271 = getelementptr inbounds double, ptr %slots, i64 4
  %t272 = load double, ptr %t271, align 8
  %t273 = fmul double %t272, 0x4012666666666666
  %t274 = fneg double %t273
  %t275 = fneg double %t274
  %t276 = getelementptr inbounds double, ptr %slots, i64 49
  store double %t275, ptr %t276, align 8
  %t277 = getelementptr inbounds double, ptr %slots, i64 4
  %t278 = load double, ptr %t277, align 8
  %t279 = fmul double %t278, 0x4014000000000000
  %t280 = fneg double %t279
  %t281 = fneg double %t280
  %t282 = getelementptr inbounds double, ptr %slots, i64 50
  store double %t281, ptr %t282, align 8
  %t283 = getelementptr inbounds double, ptr %slots, i64 4
  %t284 = load double, ptr %t283, align 8
  %t285 = fmul double %t284, 0x401599999999999a
  %t286 = fneg double %t285
  %t287 = fneg double %t286
  %t288 = getelementptr inbounds double, ptr %slots, i64 51
  store double %t287, ptr %t288, align 8
  %t289 = getelementptr inbounds double, ptr %slots, i64 4
  %t290 = load double, ptr %t289, align 8
  %t291 = fmul double %t290, 0x4017333333333333
  %t292 = fneg double %t291
  %t293 = fneg double %t292
  %t294 = getelementptr inbounds double, ptr %slots, i64 52
  store double %t293, ptr %t294, align 8
  %t295 = getelementptr inbounds double, ptr %slots, i64 4
  %t296 = load double, ptr %t295, align 8
  %t297 = fmul double %t296, 0x4018cccccccccccd
  %t298 = fneg double %t297
  %t299 = fneg double %t298
  %t300 = getelementptr inbounds double, ptr %slots, i64 53
  store double %t299, ptr %t300, align 8
  %t301 = getelementptr inbounds double, ptr %slots, i64 4
  %t302 = load double, ptr %t301, align 8
  %t303 = fmul double %t302, 0x401a666666666666
  %t304 = fneg double %t303
  %t305 = fneg double %t304
  %t306 = getelementptr inbounds double, ptr %slots, i64 54
  store double %t305, ptr %t306, align 8
  %t307 = getelementptr inbounds double, ptr %slots, i64 4
  %t308 = load double, ptr %t307, align 8
  %t309 = fmul double %t308, 0x401c000000000000
  %t310 = fneg double %t309
  %t311 = fneg double %t310
  %t312 = getelementptr inbounds double, ptr %slots, i64 55
  store double %t311, ptr %t312, align 8
  %t313 = getelementptr inbounds double, ptr %slots, i64 4
  %t314 = load double, ptr %t313, align 8
  %t315 = fmul double %t314, 0x401d99999999999a
  %t316 = fneg double %t315
  %t317 = fneg double %t316
  %t318 = getelementptr inbounds double, ptr %slots, i64 56
  store double %t317, ptr %t318, align 8
  %t319 = getelementptr inbounds ptr, ptr %arrays, i64 41
  %t320 = load ptr, ptr %t319, align 8
  %t321 = bitcast double %t226 to i64
  %t322 = getelementptr inbounds i64, ptr %t320, i64 0
  store i64 %t321, ptr %t322, align 8
  %t323 = bitcast double %t232 to i64
  %t324 = getelementptr inbounds i64, ptr %t320, i64 1
  store i64 %t323, ptr %t324, align 8
  %t325 = bitcast double %t238 to i64
  %t326 = getelementptr inbounds i64, ptr %t320, i64 2
  store i64 %t325, ptr %t326, align 8
  %t327 = bitcast double %t244 to i64
  %t328 = getelementptr inbounds i64, ptr %t320, i64 3
  store i64 %t327, ptr %t328, align 8
  %t329 = bitcast double %t250 to i64
  %t330 = getelementptr inbounds i64, ptr %t320, i64 4
  store i64 %t329, ptr %t330, align 8
  %t331 = bitcast double %t256 to i64
  %t332 = getelementptr inbounds i64, ptr %t320, i64 5
  store i64 %t331, ptr %t332, align 8
  %t333 = bitcast double %t262 to i64
  %t334 = getelementptr inbounds i64, ptr %t320, i64 6
  store i64 %t333, ptr %t334, align 8
  %t335 = bitcast double %t268 to i64
  %t336 = getelementptr inbounds i64, ptr %t320, i64 7
  store i64 %t335, ptr %t336, align 8
  %t337 = bitcast double %t274 to i64
  %t338 = getelementptr inbounds i64, ptr %t320, i64 8
  store i64 %t337, ptr %t338, align 8
  %t339 = bitcast double %t280 to i64
  %t340 = getelementptr inbounds i64, ptr %t320, i64 9
  store i64 %t339, ptr %t340, align 8
  %t341 = bitcast double %t286 to i64
  %t342 = getelementptr inbounds i64, ptr %t320, i64 10
  store i64 %t341, ptr %t342, align 8
  %t343 = bitcast double %t292 to i64
  %t344 = getelementptr inbounds i64, ptr %t320, i64 11
  store i64 %t343, ptr %t344, align 8
  %t345 = bitcast double %t298 to i64
  %t346 = getelementptr inbounds i64, ptr %t320, i64 12
  store i64 %t345, ptr %t346, align 8
  %t347 = bitcast double %t304 to i64
  %t348 = getelementptr inbounds i64, ptr %t320, i64 13
  store i64 %t347, ptr %t348, align 8
  %t349 = bitcast double %t310 to i64
  %t350 = getelementptr inbounds i64, ptr %t320, i64 14
  store i64 %t349, ptr %t350, align 8
  %t351 = bitcast double %t316 to i64
  %t352 = getelementptr inbounds i64, ptr %t320, i64 15
  store i64 %t351, ptr %t352, align 8
  %t353 = getelementptr inbounds ptr, ptr %arrays, i64 42
  %t354 = load ptr, ptr %t353, align 8
  %t355 = bitcast double %t34 to i64
  %t356 = getelementptr inbounds i64, ptr %t354, i64 0
  store i64 %t355, ptr %t356, align 8
  %t357 = bitcast double %t46 to i64
  %t358 = getelementptr inbounds i64, ptr %t354, i64 1
  store i64 %t357, ptr %t358, align 8
  %t359 = bitcast double %t58 to i64
  %t360 = getelementptr inbounds i64, ptr %t354, i64 2
  store i64 %t359, ptr %t360, align 8
  %t361 = bitcast double %t70 to i64
  %t362 = getelementptr inbounds i64, ptr %t354, i64 3
  store i64 %t361, ptr %t362, align 8
  %t363 = bitcast double %t82 to i64
  %t364 = getelementptr inbounds i64, ptr %t354, i64 4
  store i64 %t363, ptr %t364, align 8
  %t365 = bitcast double %t94 to i64
  %t366 = getelementptr inbounds i64, ptr %t354, i64 5
  store i64 %t365, ptr %t366, align 8
  %t367 = bitcast double %t106 to i64
  %t368 = getelementptr inbounds i64, ptr %t354, i64 6
  store i64 %t367, ptr %t368, align 8
  %t369 = bitcast double %t118 to i64
  %t370 = getelementptr inbounds i64, ptr %t354, i64 7
  store i64 %t369, ptr %t370, align 8
  %t371 = bitcast double %t130 to i64
  %t372 = getelementptr inbounds i64, ptr %t354, i64 8
  store i64 %t371, ptr %t372, align 8
  %t373 = bitcast double %t142 to i64
  %t374 = getelementptr inbounds i64, ptr %t354, i64 9
  store i64 %t373, ptr %t374, align 8
  %t375 = bitcast double %t154 to i64
  %t376 = getelementptr inbounds i64, ptr %t354, i64 10
  store i64 %t375, ptr %t376, align 8
  %t377 = bitcast double %t166 to i64
  %t378 = getelementptr inbounds i64, ptr %t354, i64 11
  store i64 %t377, ptr %t378, align 8
  %t379 = bitcast double %t178 to i64
  %t380 = getelementptr inbounds i64, ptr %t354, i64 12
  store i64 %t379, ptr %t380, align 8
  %t381 = bitcast double %t190 to i64
  %t382 = getelementptr inbounds i64, ptr %t354, i64 13
  store i64 %t381, ptr %t382, align 8
  %t383 = bitcast double %t202 to i64
  %t384 = getelementptr inbounds i64, ptr %t354, i64 14
  store i64 %t383, ptr %t384, align 8
  %t385 = bitcast double %t214 to i64
  %t386 = getelementptr inbounds i64, ptr %t354, i64 15
  store i64 %t385, ptr %t386, align 8
  %t387 = fptosi double %t3 to i64
  %t388 = sitofp i64 %t387 to double
  %t389 = getelementptr inbounds double, ptr %slots, i64 57
  store double %t388, ptr %t389, align 8
  %t390 = fptosi double %t9 to i64
  %t391 = sitofp i64 %t390 to double
  %t392 = getelementptr inbounds double, ptr %slots, i64 58
  store double %t391, ptr %t392, align 8
  %t393 = fptosi double %t3 to i64
  %t394 = sitofp i64 %t393 to double
  %t395 = getelementptr inbounds double, ptr %slots, i64 59
  store double %t394, ptr %t395, align 8
  %t396 = fptosi double %t9 to i64
  %t397 = sitofp i64 %t396 to double
  %t398 = getelementptr inbounds double, ptr %slots, i64 60
  store double %t397, ptr %t398, align 8
  %t399 = getelementptr inbounds double, ptr %output_buffer, i64 %s
  store double 0x0000000000000000, ptr %t399, align 8
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
