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
  %t223 = getelementptr inbounds double, ptr %slots, i64 3
  %t224 = load double, ptr %t223, align 8
  %t225 = fmul double 0x4031000000000000, %t224
  %t226 = fmul double 0x401921fb54442d18, %t225
  %t227 = fdiv double %t226, 0x401921fb54442d18
  %t228 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t229 = select i1 %t228, double 0x0000000000000000, double %t227
  %t230 = fmul double %t229, 0x41f0000000000000
  %t231 = fdiv double %t230, %sampleRate
  %t232 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t233 = select i1 %t232, double 0x0000000000000000, double %t231
  %t234 = getelementptr inbounds double, ptr %slots, i64 41
  store double %t233, ptr %t234, align 8
  %t235 = getelementptr inbounds double, ptr %slots, i64 3
  %t236 = load double, ptr %t235, align 8
  %t237 = fmul double 0x4032000000000000, %t236
  %t238 = fmul double 0x401921fb54442d18, %t237
  %t239 = fdiv double %t238, 0x401921fb54442d18
  %t240 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t241 = select i1 %t240, double 0x0000000000000000, double %t239
  %t242 = fmul double %t241, 0x41f0000000000000
  %t243 = fdiv double %t242, %sampleRate
  %t244 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t245 = select i1 %t244, double 0x0000000000000000, double %t243
  %t246 = getelementptr inbounds double, ptr %slots, i64 42
  store double %t245, ptr %t246, align 8
  %t247 = getelementptr inbounds double, ptr %slots, i64 3
  %t248 = load double, ptr %t247, align 8
  %t249 = fmul double 0x4033000000000000, %t248
  %t250 = fmul double 0x401921fb54442d18, %t249
  %t251 = fdiv double %t250, 0x401921fb54442d18
  %t252 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t253 = select i1 %t252, double 0x0000000000000000, double %t251
  %t254 = fmul double %t253, 0x41f0000000000000
  %t255 = fdiv double %t254, %sampleRate
  %t256 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t257 = select i1 %t256, double 0x0000000000000000, double %t255
  %t258 = getelementptr inbounds double, ptr %slots, i64 43
  store double %t257, ptr %t258, align 8
  %t259 = getelementptr inbounds double, ptr %slots, i64 3
  %t260 = load double, ptr %t259, align 8
  %t261 = fmul double 0x4034000000000000, %t260
  %t262 = fmul double 0x401921fb54442d18, %t261
  %t263 = fdiv double %t262, 0x401921fb54442d18
  %t264 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t265 = select i1 %t264, double 0x0000000000000000, double %t263
  %t266 = fmul double %t265, 0x41f0000000000000
  %t267 = fdiv double %t266, %sampleRate
  %t268 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t269 = select i1 %t268, double 0x0000000000000000, double %t267
  %t270 = getelementptr inbounds double, ptr %slots, i64 44
  store double %t269, ptr %t270, align 8
  %t271 = getelementptr inbounds double, ptr %slots, i64 3
  %t272 = load double, ptr %t271, align 8
  %t273 = fmul double 0x4035000000000000, %t272
  %t274 = fmul double 0x401921fb54442d18, %t273
  %t275 = fdiv double %t274, 0x401921fb54442d18
  %t276 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t277 = select i1 %t276, double 0x0000000000000000, double %t275
  %t278 = fmul double %t277, 0x41f0000000000000
  %t279 = fdiv double %t278, %sampleRate
  %t280 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t281 = select i1 %t280, double 0x0000000000000000, double %t279
  %t282 = getelementptr inbounds double, ptr %slots, i64 45
  store double %t281, ptr %t282, align 8
  %t283 = getelementptr inbounds double, ptr %slots, i64 3
  %t284 = load double, ptr %t283, align 8
  %t285 = fmul double 0x4036000000000000, %t284
  %t286 = fmul double 0x401921fb54442d18, %t285
  %t287 = fdiv double %t286, 0x401921fb54442d18
  %t288 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t289 = select i1 %t288, double 0x0000000000000000, double %t287
  %t290 = fmul double %t289, 0x41f0000000000000
  %t291 = fdiv double %t290, %sampleRate
  %t292 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t293 = select i1 %t292, double 0x0000000000000000, double %t291
  %t294 = getelementptr inbounds double, ptr %slots, i64 46
  store double %t293, ptr %t294, align 8
  %t295 = getelementptr inbounds double, ptr %slots, i64 3
  %t296 = load double, ptr %t295, align 8
  %t297 = fmul double 0x4037000000000000, %t296
  %t298 = fmul double 0x401921fb54442d18, %t297
  %t299 = fdiv double %t298, 0x401921fb54442d18
  %t300 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t301 = select i1 %t300, double 0x0000000000000000, double %t299
  %t302 = fmul double %t301, 0x41f0000000000000
  %t303 = fdiv double %t302, %sampleRate
  %t304 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t305 = select i1 %t304, double 0x0000000000000000, double %t303
  %t306 = getelementptr inbounds double, ptr %slots, i64 47
  store double %t305, ptr %t306, align 8
  %t307 = getelementptr inbounds double, ptr %slots, i64 3
  %t308 = load double, ptr %t307, align 8
  %t309 = fmul double 0x4038000000000000, %t308
  %t310 = fmul double 0x401921fb54442d18, %t309
  %t311 = fdiv double %t310, 0x401921fb54442d18
  %t312 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t313 = select i1 %t312, double 0x0000000000000000, double %t311
  %t314 = fmul double %t313, 0x41f0000000000000
  %t315 = fdiv double %t314, %sampleRate
  %t316 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t317 = select i1 %t316, double 0x0000000000000000, double %t315
  %t318 = getelementptr inbounds double, ptr %slots, i64 48
  store double %t317, ptr %t318, align 8
  %t319 = getelementptr inbounds double, ptr %slots, i64 3
  %t320 = load double, ptr %t319, align 8
  %t321 = fmul double 0x4039000000000000, %t320
  %t322 = fmul double 0x401921fb54442d18, %t321
  %t323 = fdiv double %t322, 0x401921fb54442d18
  %t324 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t325 = select i1 %t324, double 0x0000000000000000, double %t323
  %t326 = fmul double %t325, 0x41f0000000000000
  %t327 = fdiv double %t326, %sampleRate
  %t328 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t329 = select i1 %t328, double 0x0000000000000000, double %t327
  %t330 = getelementptr inbounds double, ptr %slots, i64 49
  store double %t329, ptr %t330, align 8
  %t331 = getelementptr inbounds double, ptr %slots, i64 3
  %t332 = load double, ptr %t331, align 8
  %t333 = fmul double 0x403a000000000000, %t332
  %t334 = fmul double 0x401921fb54442d18, %t333
  %t335 = fdiv double %t334, 0x401921fb54442d18
  %t336 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t337 = select i1 %t336, double 0x0000000000000000, double %t335
  %t338 = fmul double %t337, 0x41f0000000000000
  %t339 = fdiv double %t338, %sampleRate
  %t340 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t341 = select i1 %t340, double 0x0000000000000000, double %t339
  %t342 = getelementptr inbounds double, ptr %slots, i64 50
  store double %t341, ptr %t342, align 8
  %t343 = getelementptr inbounds double, ptr %slots, i64 3
  %t344 = load double, ptr %t343, align 8
  %t345 = fmul double 0x403b000000000000, %t344
  %t346 = fmul double 0x401921fb54442d18, %t345
  %t347 = fdiv double %t346, 0x401921fb54442d18
  %t348 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t349 = select i1 %t348, double 0x0000000000000000, double %t347
  %t350 = fmul double %t349, 0x41f0000000000000
  %t351 = fdiv double %t350, %sampleRate
  %t352 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t353 = select i1 %t352, double 0x0000000000000000, double %t351
  %t354 = getelementptr inbounds double, ptr %slots, i64 51
  store double %t353, ptr %t354, align 8
  %t355 = getelementptr inbounds double, ptr %slots, i64 3
  %t356 = load double, ptr %t355, align 8
  %t357 = fmul double 0x403c000000000000, %t356
  %t358 = fmul double 0x401921fb54442d18, %t357
  %t359 = fdiv double %t358, 0x401921fb54442d18
  %t360 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t361 = select i1 %t360, double 0x0000000000000000, double %t359
  %t362 = fmul double %t361, 0x41f0000000000000
  %t363 = fdiv double %t362, %sampleRate
  %t364 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t365 = select i1 %t364, double 0x0000000000000000, double %t363
  %t366 = getelementptr inbounds double, ptr %slots, i64 52
  store double %t365, ptr %t366, align 8
  %t367 = getelementptr inbounds double, ptr %slots, i64 3
  %t368 = load double, ptr %t367, align 8
  %t369 = fmul double 0x403d000000000000, %t368
  %t370 = fmul double 0x401921fb54442d18, %t369
  %t371 = fdiv double %t370, 0x401921fb54442d18
  %t372 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t373 = select i1 %t372, double 0x0000000000000000, double %t371
  %t374 = fmul double %t373, 0x41f0000000000000
  %t375 = fdiv double %t374, %sampleRate
  %t376 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t377 = select i1 %t376, double 0x0000000000000000, double %t375
  %t378 = getelementptr inbounds double, ptr %slots, i64 53
  store double %t377, ptr %t378, align 8
  %t379 = getelementptr inbounds double, ptr %slots, i64 3
  %t380 = load double, ptr %t379, align 8
  %t381 = fmul double 0x403e000000000000, %t380
  %t382 = fmul double 0x401921fb54442d18, %t381
  %t383 = fdiv double %t382, 0x401921fb54442d18
  %t384 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t385 = select i1 %t384, double 0x0000000000000000, double %t383
  %t386 = fmul double %t385, 0x41f0000000000000
  %t387 = fdiv double %t386, %sampleRate
  %t388 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t389 = select i1 %t388, double 0x0000000000000000, double %t387
  %t390 = getelementptr inbounds double, ptr %slots, i64 54
  store double %t389, ptr %t390, align 8
  %t391 = getelementptr inbounds double, ptr %slots, i64 3
  %t392 = load double, ptr %t391, align 8
  %t393 = fmul double 0x403f000000000000, %t392
  %t394 = fmul double 0x401921fb54442d18, %t393
  %t395 = fdiv double %t394, 0x401921fb54442d18
  %t396 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t397 = select i1 %t396, double 0x0000000000000000, double %t395
  %t398 = fmul double %t397, 0x41f0000000000000
  %t399 = fdiv double %t398, %sampleRate
  %t400 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t401 = select i1 %t400, double 0x0000000000000000, double %t399
  %t402 = getelementptr inbounds double, ptr %slots, i64 55
  store double %t401, ptr %t402, align 8
  %t403 = getelementptr inbounds double, ptr %slots, i64 3
  %t404 = load double, ptr %t403, align 8
  %t405 = fmul double 0x4040000000000000, %t404
  %t406 = fmul double 0x401921fb54442d18, %t405
  %t407 = fdiv double %t406, 0x401921fb54442d18
  %t408 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t409 = select i1 %t408, double 0x0000000000000000, double %t407
  %t410 = fmul double %t409, 0x41f0000000000000
  %t411 = fdiv double %t410, %sampleRate
  %t412 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t413 = select i1 %t412, double 0x0000000000000000, double %t411
  %t414 = getelementptr inbounds double, ptr %slots, i64 56
  store double %t413, ptr %t414, align 8
  %t415 = getelementptr inbounds double, ptr %slots, i64 3
  %t416 = load double, ptr %t415, align 8
  %t417 = fmul double 0x4040800000000000, %t416
  %t418 = fmul double 0x401921fb54442d18, %t417
  %t419 = fdiv double %t418, 0x401921fb54442d18
  %t420 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t421 = select i1 %t420, double 0x0000000000000000, double %t419
  %t422 = fmul double %t421, 0x41f0000000000000
  %t423 = fdiv double %t422, %sampleRate
  %t424 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t425 = select i1 %t424, double 0x0000000000000000, double %t423
  %t426 = getelementptr inbounds double, ptr %slots, i64 57
  store double %t425, ptr %t426, align 8
  %t427 = getelementptr inbounds double, ptr %slots, i64 3
  %t428 = load double, ptr %t427, align 8
  %t429 = fmul double 0x4041000000000000, %t428
  %t430 = fmul double 0x401921fb54442d18, %t429
  %t431 = fdiv double %t430, 0x401921fb54442d18
  %t432 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t433 = select i1 %t432, double 0x0000000000000000, double %t431
  %t434 = fmul double %t433, 0x41f0000000000000
  %t435 = fdiv double %t434, %sampleRate
  %t436 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t437 = select i1 %t436, double 0x0000000000000000, double %t435
  %t438 = getelementptr inbounds double, ptr %slots, i64 58
  store double %t437, ptr %t438, align 8
  %t439 = getelementptr inbounds double, ptr %slots, i64 3
  %t440 = load double, ptr %t439, align 8
  %t441 = fmul double 0x4041800000000000, %t440
  %t442 = fmul double 0x401921fb54442d18, %t441
  %t443 = fdiv double %t442, 0x401921fb54442d18
  %t444 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t445 = select i1 %t444, double 0x0000000000000000, double %t443
  %t446 = fmul double %t445, 0x41f0000000000000
  %t447 = fdiv double %t446, %sampleRate
  %t448 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t449 = select i1 %t448, double 0x0000000000000000, double %t447
  %t450 = getelementptr inbounds double, ptr %slots, i64 59
  store double %t449, ptr %t450, align 8
  %t451 = getelementptr inbounds double, ptr %slots, i64 3
  %t452 = load double, ptr %t451, align 8
  %t453 = fmul double 0x4042000000000000, %t452
  %t454 = fmul double 0x401921fb54442d18, %t453
  %t455 = fdiv double %t454, 0x401921fb54442d18
  %t456 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t457 = select i1 %t456, double 0x0000000000000000, double %t455
  %t458 = fmul double %t457, 0x41f0000000000000
  %t459 = fdiv double %t458, %sampleRate
  %t460 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t461 = select i1 %t460, double 0x0000000000000000, double %t459
  %t462 = getelementptr inbounds double, ptr %slots, i64 60
  store double %t461, ptr %t462, align 8
  %t463 = getelementptr inbounds double, ptr %slots, i64 3
  %t464 = load double, ptr %t463, align 8
  %t465 = fmul double 0x4042800000000000, %t464
  %t466 = fmul double 0x401921fb54442d18, %t465
  %t467 = fdiv double %t466, 0x401921fb54442d18
  %t468 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t469 = select i1 %t468, double 0x0000000000000000, double %t467
  %t470 = fmul double %t469, 0x41f0000000000000
  %t471 = fdiv double %t470, %sampleRate
  %t472 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t473 = select i1 %t472, double 0x0000000000000000, double %t471
  %t474 = getelementptr inbounds double, ptr %slots, i64 61
  store double %t473, ptr %t474, align 8
  %t475 = getelementptr inbounds double, ptr %slots, i64 3
  %t476 = load double, ptr %t475, align 8
  %t477 = fmul double 0x4043000000000000, %t476
  %t478 = fmul double 0x401921fb54442d18, %t477
  %t479 = fdiv double %t478, 0x401921fb54442d18
  %t480 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t481 = select i1 %t480, double 0x0000000000000000, double %t479
  %t482 = fmul double %t481, 0x41f0000000000000
  %t483 = fdiv double %t482, %sampleRate
  %t484 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t485 = select i1 %t484, double 0x0000000000000000, double %t483
  %t486 = getelementptr inbounds double, ptr %slots, i64 62
  store double %t485, ptr %t486, align 8
  %t487 = getelementptr inbounds double, ptr %slots, i64 3
  %t488 = load double, ptr %t487, align 8
  %t489 = fmul double 0x4043800000000000, %t488
  %t490 = fmul double 0x401921fb54442d18, %t489
  %t491 = fdiv double %t490, 0x401921fb54442d18
  %t492 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t493 = select i1 %t492, double 0x0000000000000000, double %t491
  %t494 = fmul double %t493, 0x41f0000000000000
  %t495 = fdiv double %t494, %sampleRate
  %t496 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t497 = select i1 %t496, double 0x0000000000000000, double %t495
  %t498 = getelementptr inbounds double, ptr %slots, i64 63
  store double %t497, ptr %t498, align 8
  %t499 = getelementptr inbounds double, ptr %slots, i64 3
  %t500 = load double, ptr %t499, align 8
  %t501 = fmul double 0x4044000000000000, %t500
  %t502 = fmul double 0x401921fb54442d18, %t501
  %t503 = fdiv double %t502, 0x401921fb54442d18
  %t504 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t505 = select i1 %t504, double 0x0000000000000000, double %t503
  %t506 = fmul double %t505, 0x41f0000000000000
  %t507 = fdiv double %t506, %sampleRate
  %t508 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t509 = select i1 %t508, double 0x0000000000000000, double %t507
  %t510 = getelementptr inbounds double, ptr %slots, i64 64
  store double %t509, ptr %t510, align 8
  %t511 = getelementptr inbounds double, ptr %slots, i64 3
  %t512 = load double, ptr %t511, align 8
  %t513 = fmul double 0x4044800000000000, %t512
  %t514 = fmul double 0x401921fb54442d18, %t513
  %t515 = fdiv double %t514, 0x401921fb54442d18
  %t516 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t517 = select i1 %t516, double 0x0000000000000000, double %t515
  %t518 = fmul double %t517, 0x41f0000000000000
  %t519 = fdiv double %t518, %sampleRate
  %t520 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t521 = select i1 %t520, double 0x0000000000000000, double %t519
  %t522 = getelementptr inbounds double, ptr %slots, i64 65
  store double %t521, ptr %t522, align 8
  %t523 = getelementptr inbounds double, ptr %slots, i64 3
  %t524 = load double, ptr %t523, align 8
  %t525 = fmul double 0x4045000000000000, %t524
  %t526 = fmul double 0x401921fb54442d18, %t525
  %t527 = fdiv double %t526, 0x401921fb54442d18
  %t528 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t529 = select i1 %t528, double 0x0000000000000000, double %t527
  %t530 = fmul double %t529, 0x41f0000000000000
  %t531 = fdiv double %t530, %sampleRate
  %t532 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t533 = select i1 %t532, double 0x0000000000000000, double %t531
  %t534 = getelementptr inbounds double, ptr %slots, i64 66
  store double %t533, ptr %t534, align 8
  %t535 = getelementptr inbounds double, ptr %slots, i64 3
  %t536 = load double, ptr %t535, align 8
  %t537 = fmul double 0x4045800000000000, %t536
  %t538 = fmul double 0x401921fb54442d18, %t537
  %t539 = fdiv double %t538, 0x401921fb54442d18
  %t540 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t541 = select i1 %t540, double 0x0000000000000000, double %t539
  %t542 = fmul double %t541, 0x41f0000000000000
  %t543 = fdiv double %t542, %sampleRate
  %t544 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t545 = select i1 %t544, double 0x0000000000000000, double %t543
  %t546 = getelementptr inbounds double, ptr %slots, i64 67
  store double %t545, ptr %t546, align 8
  %t547 = getelementptr inbounds double, ptr %slots, i64 3
  %t548 = load double, ptr %t547, align 8
  %t549 = fmul double 0x4046000000000000, %t548
  %t550 = fmul double 0x401921fb54442d18, %t549
  %t551 = fdiv double %t550, 0x401921fb54442d18
  %t552 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t553 = select i1 %t552, double 0x0000000000000000, double %t551
  %t554 = fmul double %t553, 0x41f0000000000000
  %t555 = fdiv double %t554, %sampleRate
  %t556 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t557 = select i1 %t556, double 0x0000000000000000, double %t555
  %t558 = getelementptr inbounds double, ptr %slots, i64 68
  store double %t557, ptr %t558, align 8
  %t559 = getelementptr inbounds double, ptr %slots, i64 3
  %t560 = load double, ptr %t559, align 8
  %t561 = fmul double 0x4046800000000000, %t560
  %t562 = fmul double 0x401921fb54442d18, %t561
  %t563 = fdiv double %t562, 0x401921fb54442d18
  %t564 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t565 = select i1 %t564, double 0x0000000000000000, double %t563
  %t566 = fmul double %t565, 0x41f0000000000000
  %t567 = fdiv double %t566, %sampleRate
  %t568 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t569 = select i1 %t568, double 0x0000000000000000, double %t567
  %t570 = getelementptr inbounds double, ptr %slots, i64 69
  store double %t569, ptr %t570, align 8
  %t571 = getelementptr inbounds double, ptr %slots, i64 3
  %t572 = load double, ptr %t571, align 8
  %t573 = fmul double 0x4047000000000000, %t572
  %t574 = fmul double 0x401921fb54442d18, %t573
  %t575 = fdiv double %t574, 0x401921fb54442d18
  %t576 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t577 = select i1 %t576, double 0x0000000000000000, double %t575
  %t578 = fmul double %t577, 0x41f0000000000000
  %t579 = fdiv double %t578, %sampleRate
  %t580 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t581 = select i1 %t580, double 0x0000000000000000, double %t579
  %t582 = getelementptr inbounds double, ptr %slots, i64 70
  store double %t581, ptr %t582, align 8
  %t583 = getelementptr inbounds double, ptr %slots, i64 3
  %t584 = load double, ptr %t583, align 8
  %t585 = fmul double 0x4047800000000000, %t584
  %t586 = fmul double 0x401921fb54442d18, %t585
  %t587 = fdiv double %t586, 0x401921fb54442d18
  %t588 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t589 = select i1 %t588, double 0x0000000000000000, double %t587
  %t590 = fmul double %t589, 0x41f0000000000000
  %t591 = fdiv double %t590, %sampleRate
  %t592 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t593 = select i1 %t592, double 0x0000000000000000, double %t591
  %t594 = getelementptr inbounds double, ptr %slots, i64 71
  store double %t593, ptr %t594, align 8
  %t595 = getelementptr inbounds double, ptr %slots, i64 3
  %t596 = load double, ptr %t595, align 8
  %t597 = fmul double 0x4048000000000000, %t596
  %t598 = fmul double 0x401921fb54442d18, %t597
  %t599 = fdiv double %t598, 0x401921fb54442d18
  %t600 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t601 = select i1 %t600, double 0x0000000000000000, double %t599
  %t602 = fmul double %t601, 0x41f0000000000000
  %t603 = fdiv double %t602, %sampleRate
  %t604 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t605 = select i1 %t604, double 0x0000000000000000, double %t603
  %t606 = getelementptr inbounds double, ptr %slots, i64 72
  store double %t605, ptr %t606, align 8
  %t607 = getelementptr inbounds double, ptr %slots, i64 3
  %t608 = load double, ptr %t607, align 8
  %t609 = fmul double 0x4048800000000000, %t608
  %t610 = fmul double 0x401921fb54442d18, %t609
  %t611 = fdiv double %t610, 0x401921fb54442d18
  %t612 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t613 = select i1 %t612, double 0x0000000000000000, double %t611
  %t614 = fmul double %t613, 0x41f0000000000000
  %t615 = fdiv double %t614, %sampleRate
  %t616 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t617 = select i1 %t616, double 0x0000000000000000, double %t615
  %t618 = getelementptr inbounds double, ptr %slots, i64 73
  store double %t617, ptr %t618, align 8
  %t619 = getelementptr inbounds double, ptr %slots, i64 3
  %t620 = load double, ptr %t619, align 8
  %t621 = fmul double 0x4049000000000000, %t620
  %t622 = fmul double 0x401921fb54442d18, %t621
  %t623 = fdiv double %t622, 0x401921fb54442d18
  %t624 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t625 = select i1 %t624, double 0x0000000000000000, double %t623
  %t626 = fmul double %t625, 0x41f0000000000000
  %t627 = fdiv double %t626, %sampleRate
  %t628 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t629 = select i1 %t628, double 0x0000000000000000, double %t627
  %t630 = getelementptr inbounds double, ptr %slots, i64 74
  store double %t629, ptr %t630, align 8
  %t631 = getelementptr inbounds double, ptr %slots, i64 3
  %t632 = load double, ptr %t631, align 8
  %t633 = fmul double 0x4049800000000000, %t632
  %t634 = fmul double 0x401921fb54442d18, %t633
  %t635 = fdiv double %t634, 0x401921fb54442d18
  %t636 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t637 = select i1 %t636, double 0x0000000000000000, double %t635
  %t638 = fmul double %t637, 0x41f0000000000000
  %t639 = fdiv double %t638, %sampleRate
  %t640 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t641 = select i1 %t640, double 0x0000000000000000, double %t639
  %t642 = getelementptr inbounds double, ptr %slots, i64 75
  store double %t641, ptr %t642, align 8
  %t643 = getelementptr inbounds double, ptr %slots, i64 3
  %t644 = load double, ptr %t643, align 8
  %t645 = fmul double 0x404a000000000000, %t644
  %t646 = fmul double 0x401921fb54442d18, %t645
  %t647 = fdiv double %t646, 0x401921fb54442d18
  %t648 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t649 = select i1 %t648, double 0x0000000000000000, double %t647
  %t650 = fmul double %t649, 0x41f0000000000000
  %t651 = fdiv double %t650, %sampleRate
  %t652 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t653 = select i1 %t652, double 0x0000000000000000, double %t651
  %t654 = getelementptr inbounds double, ptr %slots, i64 76
  store double %t653, ptr %t654, align 8
  %t655 = getelementptr inbounds double, ptr %slots, i64 3
  %t656 = load double, ptr %t655, align 8
  %t657 = fmul double 0x404a800000000000, %t656
  %t658 = fmul double 0x401921fb54442d18, %t657
  %t659 = fdiv double %t658, 0x401921fb54442d18
  %t660 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t661 = select i1 %t660, double 0x0000000000000000, double %t659
  %t662 = fmul double %t661, 0x41f0000000000000
  %t663 = fdiv double %t662, %sampleRate
  %t664 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t665 = select i1 %t664, double 0x0000000000000000, double %t663
  %t666 = getelementptr inbounds double, ptr %slots, i64 77
  store double %t665, ptr %t666, align 8
  %t667 = getelementptr inbounds double, ptr %slots, i64 3
  %t668 = load double, ptr %t667, align 8
  %t669 = fmul double 0x404b000000000000, %t668
  %t670 = fmul double 0x401921fb54442d18, %t669
  %t671 = fdiv double %t670, 0x401921fb54442d18
  %t672 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t673 = select i1 %t672, double 0x0000000000000000, double %t671
  %t674 = fmul double %t673, 0x41f0000000000000
  %t675 = fdiv double %t674, %sampleRate
  %t676 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t677 = select i1 %t676, double 0x0000000000000000, double %t675
  %t678 = getelementptr inbounds double, ptr %slots, i64 78
  store double %t677, ptr %t678, align 8
  %t679 = getelementptr inbounds double, ptr %slots, i64 3
  %t680 = load double, ptr %t679, align 8
  %t681 = fmul double 0x404b800000000000, %t680
  %t682 = fmul double 0x401921fb54442d18, %t681
  %t683 = fdiv double %t682, 0x401921fb54442d18
  %t684 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t685 = select i1 %t684, double 0x0000000000000000, double %t683
  %t686 = fmul double %t685, 0x41f0000000000000
  %t687 = fdiv double %t686, %sampleRate
  %t688 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t689 = select i1 %t688, double 0x0000000000000000, double %t687
  %t690 = getelementptr inbounds double, ptr %slots, i64 79
  store double %t689, ptr %t690, align 8
  %t691 = getelementptr inbounds double, ptr %slots, i64 3
  %t692 = load double, ptr %t691, align 8
  %t693 = fmul double 0x404c000000000000, %t692
  %t694 = fmul double 0x401921fb54442d18, %t693
  %t695 = fdiv double %t694, 0x401921fb54442d18
  %t696 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t697 = select i1 %t696, double 0x0000000000000000, double %t695
  %t698 = fmul double %t697, 0x41f0000000000000
  %t699 = fdiv double %t698, %sampleRate
  %t700 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t701 = select i1 %t700, double 0x0000000000000000, double %t699
  %t702 = getelementptr inbounds double, ptr %slots, i64 80
  store double %t701, ptr %t702, align 8
  %t703 = getelementptr inbounds double, ptr %slots, i64 3
  %t704 = load double, ptr %t703, align 8
  %t705 = fmul double 0x404c800000000000, %t704
  %t706 = fmul double 0x401921fb54442d18, %t705
  %t707 = fdiv double %t706, 0x401921fb54442d18
  %t708 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t709 = select i1 %t708, double 0x0000000000000000, double %t707
  %t710 = fmul double %t709, 0x41f0000000000000
  %t711 = fdiv double %t710, %sampleRate
  %t712 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t713 = select i1 %t712, double 0x0000000000000000, double %t711
  %t714 = getelementptr inbounds double, ptr %slots, i64 81
  store double %t713, ptr %t714, align 8
  %t715 = getelementptr inbounds double, ptr %slots, i64 3
  %t716 = load double, ptr %t715, align 8
  %t717 = fmul double 0x404d000000000000, %t716
  %t718 = fmul double 0x401921fb54442d18, %t717
  %t719 = fdiv double %t718, 0x401921fb54442d18
  %t720 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t721 = select i1 %t720, double 0x0000000000000000, double %t719
  %t722 = fmul double %t721, 0x41f0000000000000
  %t723 = fdiv double %t722, %sampleRate
  %t724 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t725 = select i1 %t724, double 0x0000000000000000, double %t723
  %t726 = getelementptr inbounds double, ptr %slots, i64 82
  store double %t725, ptr %t726, align 8
  %t727 = getelementptr inbounds double, ptr %slots, i64 3
  %t728 = load double, ptr %t727, align 8
  %t729 = fmul double 0x404d800000000000, %t728
  %t730 = fmul double 0x401921fb54442d18, %t729
  %t731 = fdiv double %t730, 0x401921fb54442d18
  %t732 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t733 = select i1 %t732, double 0x0000000000000000, double %t731
  %t734 = fmul double %t733, 0x41f0000000000000
  %t735 = fdiv double %t734, %sampleRate
  %t736 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t737 = select i1 %t736, double 0x0000000000000000, double %t735
  %t738 = getelementptr inbounds double, ptr %slots, i64 83
  store double %t737, ptr %t738, align 8
  %t739 = getelementptr inbounds double, ptr %slots, i64 3
  %t740 = load double, ptr %t739, align 8
  %t741 = fmul double 0x404e000000000000, %t740
  %t742 = fmul double 0x401921fb54442d18, %t741
  %t743 = fdiv double %t742, 0x401921fb54442d18
  %t744 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t745 = select i1 %t744, double 0x0000000000000000, double %t743
  %t746 = fmul double %t745, 0x41f0000000000000
  %t747 = fdiv double %t746, %sampleRate
  %t748 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t749 = select i1 %t748, double 0x0000000000000000, double %t747
  %t750 = getelementptr inbounds double, ptr %slots, i64 84
  store double %t749, ptr %t750, align 8
  %t751 = getelementptr inbounds double, ptr %slots, i64 3
  %t752 = load double, ptr %t751, align 8
  %t753 = fmul double 0x404e800000000000, %t752
  %t754 = fmul double 0x401921fb54442d18, %t753
  %t755 = fdiv double %t754, 0x401921fb54442d18
  %t756 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t757 = select i1 %t756, double 0x0000000000000000, double %t755
  %t758 = fmul double %t757, 0x41f0000000000000
  %t759 = fdiv double %t758, %sampleRate
  %t760 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t761 = select i1 %t760, double 0x0000000000000000, double %t759
  %t762 = getelementptr inbounds double, ptr %slots, i64 85
  store double %t761, ptr %t762, align 8
  %t763 = getelementptr inbounds double, ptr %slots, i64 3
  %t764 = load double, ptr %t763, align 8
  %t765 = fmul double 0x404f000000000000, %t764
  %t766 = fmul double 0x401921fb54442d18, %t765
  %t767 = fdiv double %t766, 0x401921fb54442d18
  %t768 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t769 = select i1 %t768, double 0x0000000000000000, double %t767
  %t770 = fmul double %t769, 0x41f0000000000000
  %t771 = fdiv double %t770, %sampleRate
  %t772 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t773 = select i1 %t772, double 0x0000000000000000, double %t771
  %t774 = getelementptr inbounds double, ptr %slots, i64 86
  store double %t773, ptr %t774, align 8
  %t775 = getelementptr inbounds double, ptr %slots, i64 3
  %t776 = load double, ptr %t775, align 8
  %t777 = fmul double 0x404f800000000000, %t776
  %t778 = fmul double 0x401921fb54442d18, %t777
  %t779 = fdiv double %t778, 0x401921fb54442d18
  %t780 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t781 = select i1 %t780, double 0x0000000000000000, double %t779
  %t782 = fmul double %t781, 0x41f0000000000000
  %t783 = fdiv double %t782, %sampleRate
  %t784 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t785 = select i1 %t784, double 0x0000000000000000, double %t783
  %t786 = getelementptr inbounds double, ptr %slots, i64 87
  store double %t785, ptr %t786, align 8
  %t787 = getelementptr inbounds double, ptr %slots, i64 3
  %t788 = load double, ptr %t787, align 8
  %t789 = fmul double 0x4050000000000000, %t788
  %t790 = fmul double 0x401921fb54442d18, %t789
  %t791 = fdiv double %t790, 0x401921fb54442d18
  %t792 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t793 = select i1 %t792, double 0x0000000000000000, double %t791
  %t794 = fmul double %t793, 0x41f0000000000000
  %t795 = fdiv double %t794, %sampleRate
  %t796 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t797 = select i1 %t796, double 0x0000000000000000, double %t795
  %t798 = getelementptr inbounds double, ptr %slots, i64 88
  store double %t797, ptr %t798, align 8
  %t799 = getelementptr inbounds double, ptr %slots, i64 4
  %t800 = load double, ptr %t799, align 8
  %t801 = fmul double %t800, 0x3ff6666666666666
  %t802 = fneg double %t801
  %t803 = fneg double %t802
  %t804 = getelementptr inbounds double, ptr %slots, i64 89
  store double %t803, ptr %t804, align 8
  %t805 = getelementptr inbounds double, ptr %slots, i64 4
  %t806 = load double, ptr %t805, align 8
  %t807 = fmul double %t806, 0x3ffccccccccccccd
  %t808 = fneg double %t807
  %t809 = fneg double %t808
  %t810 = getelementptr inbounds double, ptr %slots, i64 90
  store double %t809, ptr %t810, align 8
  %t811 = getelementptr inbounds double, ptr %slots, i64 4
  %t812 = load double, ptr %t811, align 8
  %t813 = fmul double %t812, 0x400199999999999a
  %t814 = fneg double %t813
  %t815 = fneg double %t814
  %t816 = getelementptr inbounds double, ptr %slots, i64 91
  store double %t815, ptr %t816, align 8
  %t817 = getelementptr inbounds double, ptr %slots, i64 4
  %t818 = load double, ptr %t817, align 8
  %t819 = fmul double %t818, 0x4004cccccccccccd
  %t820 = fneg double %t819
  %t821 = fneg double %t820
  %t822 = getelementptr inbounds double, ptr %slots, i64 92
  store double %t821, ptr %t822, align 8
  %t823 = getelementptr inbounds double, ptr %slots, i64 4
  %t824 = load double, ptr %t823, align 8
  %t825 = fmul double %t824, 0x4008000000000000
  %t826 = fneg double %t825
  %t827 = fneg double %t826
  %t828 = getelementptr inbounds double, ptr %slots, i64 93
  store double %t827, ptr %t828, align 8
  %t829 = getelementptr inbounds double, ptr %slots, i64 4
  %t830 = load double, ptr %t829, align 8
  %t831 = fmul double %t830, 0x400b333333333333
  %t832 = fneg double %t831
  %t833 = fneg double %t832
  %t834 = getelementptr inbounds double, ptr %slots, i64 94
  store double %t833, ptr %t834, align 8
  %t835 = getelementptr inbounds double, ptr %slots, i64 4
  %t836 = load double, ptr %t835, align 8
  %t837 = fmul double %t836, 0x400e666666666666
  %t838 = fneg double %t837
  %t839 = fneg double %t838
  %t840 = getelementptr inbounds double, ptr %slots, i64 95
  store double %t839, ptr %t840, align 8
  %t841 = getelementptr inbounds double, ptr %slots, i64 4
  %t842 = load double, ptr %t841, align 8
  %t843 = fmul double %t842, 0x4010cccccccccccd
  %t844 = fneg double %t843
  %t845 = fneg double %t844
  %t846 = getelementptr inbounds double, ptr %slots, i64 96
  store double %t845, ptr %t846, align 8
  %t847 = getelementptr inbounds double, ptr %slots, i64 4
  %t848 = load double, ptr %t847, align 8
  %t849 = fmul double %t848, 0x4012666666666666
  %t850 = fneg double %t849
  %t851 = fneg double %t850
  %t852 = getelementptr inbounds double, ptr %slots, i64 97
  store double %t851, ptr %t852, align 8
  %t853 = getelementptr inbounds double, ptr %slots, i64 4
  %t854 = load double, ptr %t853, align 8
  %t855 = fmul double %t854, 0x4014000000000000
  %t856 = fneg double %t855
  %t857 = fneg double %t856
  %t858 = getelementptr inbounds double, ptr %slots, i64 98
  store double %t857, ptr %t858, align 8
  %t859 = getelementptr inbounds double, ptr %slots, i64 4
  %t860 = load double, ptr %t859, align 8
  %t861 = fmul double %t860, 0x401599999999999a
  %t862 = fneg double %t861
  %t863 = fneg double %t862
  %t864 = getelementptr inbounds double, ptr %slots, i64 99
  store double %t863, ptr %t864, align 8
  %t865 = getelementptr inbounds double, ptr %slots, i64 4
  %t866 = load double, ptr %t865, align 8
  %t867 = fmul double %t866, 0x4017333333333333
  %t868 = fneg double %t867
  %t869 = fneg double %t868
  %t870 = getelementptr inbounds double, ptr %slots, i64 100
  store double %t869, ptr %t870, align 8
  %t871 = getelementptr inbounds double, ptr %slots, i64 4
  %t872 = load double, ptr %t871, align 8
  %t873 = fmul double %t872, 0x4018cccccccccccd
  %t874 = fneg double %t873
  %t875 = fneg double %t874
  %t876 = getelementptr inbounds double, ptr %slots, i64 101
  store double %t875, ptr %t876, align 8
  %t877 = getelementptr inbounds double, ptr %slots, i64 4
  %t878 = load double, ptr %t877, align 8
  %t879 = fmul double %t878, 0x401a666666666666
  %t880 = fneg double %t879
  %t881 = fneg double %t880
  %t882 = getelementptr inbounds double, ptr %slots, i64 102
  store double %t881, ptr %t882, align 8
  %t883 = getelementptr inbounds double, ptr %slots, i64 4
  %t884 = load double, ptr %t883, align 8
  %t885 = fmul double %t884, 0x401c000000000000
  %t886 = fneg double %t885
  %t887 = fneg double %t886
  %t888 = getelementptr inbounds double, ptr %slots, i64 103
  store double %t887, ptr %t888, align 8
  %t889 = getelementptr inbounds double, ptr %slots, i64 4
  %t890 = load double, ptr %t889, align 8
  %t891 = fmul double %t890, 0x401d99999999999a
  %t892 = fneg double %t891
  %t893 = fneg double %t892
  %t894 = getelementptr inbounds double, ptr %slots, i64 104
  store double %t893, ptr %t894, align 8
  %t895 = getelementptr inbounds double, ptr %slots, i64 4
  %t896 = load double, ptr %t895, align 8
  %t897 = fmul double %t896, 0x401f333333333333
  %t898 = fneg double %t897
  %t899 = fneg double %t898
  %t900 = getelementptr inbounds double, ptr %slots, i64 105
  store double %t899, ptr %t900, align 8
  %t901 = getelementptr inbounds double, ptr %slots, i64 4
  %t902 = load double, ptr %t901, align 8
  %t903 = fmul double %t902, 0x4020666666666666
  %t904 = fneg double %t903
  %t905 = fneg double %t904
  %t906 = getelementptr inbounds double, ptr %slots, i64 106
  store double %t905, ptr %t906, align 8
  %t907 = getelementptr inbounds double, ptr %slots, i64 4
  %t908 = load double, ptr %t907, align 8
  %t909 = fmul double %t908, 0x4021333333333333
  %t910 = fneg double %t909
  %t911 = fneg double %t910
  %t912 = getelementptr inbounds double, ptr %slots, i64 107
  store double %t911, ptr %t912, align 8
  %t913 = getelementptr inbounds double, ptr %slots, i64 4
  %t914 = load double, ptr %t913, align 8
  %t915 = fmul double %t914, 0x4022000000000000
  %t916 = fneg double %t915
  %t917 = fneg double %t916
  %t918 = getelementptr inbounds double, ptr %slots, i64 108
  store double %t917, ptr %t918, align 8
  %t919 = getelementptr inbounds double, ptr %slots, i64 4
  %t920 = load double, ptr %t919, align 8
  %t921 = fmul double %t920, 0x4022cccccccccccd
  %t922 = fneg double %t921
  %t923 = fneg double %t922
  %t924 = getelementptr inbounds double, ptr %slots, i64 109
  store double %t923, ptr %t924, align 8
  %t925 = getelementptr inbounds double, ptr %slots, i64 4
  %t926 = load double, ptr %t925, align 8
  %t927 = fmul double %t926, 0x402399999999999a
  %t928 = fneg double %t927
  %t929 = fneg double %t928
  %t930 = getelementptr inbounds double, ptr %slots, i64 110
  store double %t929, ptr %t930, align 8
  %t931 = getelementptr inbounds double, ptr %slots, i64 4
  %t932 = load double, ptr %t931, align 8
  %t933 = fmul double %t932, 0x4024666666666666
  %t934 = fneg double %t933
  %t935 = fneg double %t934
  %t936 = getelementptr inbounds double, ptr %slots, i64 111
  store double %t935, ptr %t936, align 8
  %t937 = getelementptr inbounds double, ptr %slots, i64 4
  %t938 = load double, ptr %t937, align 8
  %t939 = fmul double %t938, 0x4025333333333333
  %t940 = fneg double %t939
  %t941 = fneg double %t940
  %t942 = getelementptr inbounds double, ptr %slots, i64 112
  store double %t941, ptr %t942, align 8
  %t943 = getelementptr inbounds double, ptr %slots, i64 4
  %t944 = load double, ptr %t943, align 8
  %t945 = fmul double %t944, 0x4026000000000000
  %t946 = fneg double %t945
  %t947 = fneg double %t946
  %t948 = getelementptr inbounds double, ptr %slots, i64 113
  store double %t947, ptr %t948, align 8
  %t949 = getelementptr inbounds double, ptr %slots, i64 4
  %t950 = load double, ptr %t949, align 8
  %t951 = fmul double %t950, 0x4026cccccccccccd
  %t952 = fneg double %t951
  %t953 = fneg double %t952
  %t954 = getelementptr inbounds double, ptr %slots, i64 114
  store double %t953, ptr %t954, align 8
  %t955 = getelementptr inbounds double, ptr %slots, i64 4
  %t956 = load double, ptr %t955, align 8
  %t957 = fmul double %t956, 0x402799999999999a
  %t958 = fneg double %t957
  %t959 = fneg double %t958
  %t960 = getelementptr inbounds double, ptr %slots, i64 115
  store double %t959, ptr %t960, align 8
  %t961 = getelementptr inbounds double, ptr %slots, i64 4
  %t962 = load double, ptr %t961, align 8
  %t963 = fmul double %t962, 0x4028666666666666
  %t964 = fneg double %t963
  %t965 = fneg double %t964
  %t966 = getelementptr inbounds double, ptr %slots, i64 116
  store double %t965, ptr %t966, align 8
  %t967 = getelementptr inbounds double, ptr %slots, i64 4
  %t968 = load double, ptr %t967, align 8
  %t969 = fmul double %t968, 0x4029333333333333
  %t970 = fneg double %t969
  %t971 = fneg double %t970
  %t972 = getelementptr inbounds double, ptr %slots, i64 117
  store double %t971, ptr %t972, align 8
  %t973 = getelementptr inbounds double, ptr %slots, i64 4
  %t974 = load double, ptr %t973, align 8
  %t975 = fmul double %t974, 0x402a000000000000
  %t976 = fneg double %t975
  %t977 = fneg double %t976
  %t978 = getelementptr inbounds double, ptr %slots, i64 118
  store double %t977, ptr %t978, align 8
  %t979 = getelementptr inbounds double, ptr %slots, i64 4
  %t980 = load double, ptr %t979, align 8
  %t981 = fmul double %t980, 0x402acccccccccccd
  %t982 = fneg double %t981
  %t983 = fneg double %t982
  %t984 = getelementptr inbounds double, ptr %slots, i64 119
  store double %t983, ptr %t984, align 8
  %t985 = getelementptr inbounds double, ptr %slots, i64 4
  %t986 = load double, ptr %t985, align 8
  %t987 = fmul double %t986, 0x402b99999999999a
  %t988 = fneg double %t987
  %t989 = fneg double %t988
  %t990 = getelementptr inbounds double, ptr %slots, i64 120
  store double %t989, ptr %t990, align 8
  %t991 = getelementptr inbounds double, ptr %slots, i64 4
  %t992 = load double, ptr %t991, align 8
  %t993 = fmul double %t992, 0x402c666666666666
  %t994 = fneg double %t993
  %t995 = fneg double %t994
  %t996 = getelementptr inbounds double, ptr %slots, i64 121
  store double %t995, ptr %t996, align 8
  %t997 = getelementptr inbounds double, ptr %slots, i64 4
  %t998 = load double, ptr %t997, align 8
  %t999 = fmul double %t998, 0x402d333333333333
  %t1000 = fneg double %t999
  %t1001 = fneg double %t1000
  %t1002 = getelementptr inbounds double, ptr %slots, i64 122
  store double %t1001, ptr %t1002, align 8
  %t1003 = getelementptr inbounds double, ptr %slots, i64 4
  %t1004 = load double, ptr %t1003, align 8
  %t1005 = fmul double %t1004, 0x402e000000000000
  %t1006 = fneg double %t1005
  %t1007 = fneg double %t1006
  %t1008 = getelementptr inbounds double, ptr %slots, i64 123
  store double %t1007, ptr %t1008, align 8
  %t1009 = getelementptr inbounds double, ptr %slots, i64 4
  %t1010 = load double, ptr %t1009, align 8
  %t1011 = fmul double %t1010, 0x402ecccccccccccd
  %t1012 = fneg double %t1011
  %t1013 = fneg double %t1012
  %t1014 = getelementptr inbounds double, ptr %slots, i64 124
  store double %t1013, ptr %t1014, align 8
  %t1015 = getelementptr inbounds double, ptr %slots, i64 4
  %t1016 = load double, ptr %t1015, align 8
  %t1017 = fmul double %t1016, 0x402f99999999999a
  %t1018 = fneg double %t1017
  %t1019 = fneg double %t1018
  %t1020 = getelementptr inbounds double, ptr %slots, i64 125
  store double %t1019, ptr %t1020, align 8
  %t1021 = getelementptr inbounds double, ptr %slots, i64 4
  %t1022 = load double, ptr %t1021, align 8
  %t1023 = fmul double %t1022, 0x4030333333333333
  %t1024 = fneg double %t1023
  %t1025 = fneg double %t1024
  %t1026 = getelementptr inbounds double, ptr %slots, i64 126
  store double %t1025, ptr %t1026, align 8
  %t1027 = getelementptr inbounds double, ptr %slots, i64 4
  %t1028 = load double, ptr %t1027, align 8
  %t1029 = fmul double %t1028, 0x403099999999999a
  %t1030 = fneg double %t1029
  %t1031 = fneg double %t1030
  %t1032 = getelementptr inbounds double, ptr %slots, i64 127
  store double %t1031, ptr %t1032, align 8
  %t1033 = getelementptr inbounds double, ptr %slots, i64 4
  %t1034 = load double, ptr %t1033, align 8
  %t1035 = fmul double %t1034, 0x4031000000000000
  %t1036 = fneg double %t1035
  %t1037 = fneg double %t1036
  %t1038 = getelementptr inbounds double, ptr %slots, i64 128
  store double %t1037, ptr %t1038, align 8
  %t1039 = getelementptr inbounds double, ptr %slots, i64 4
  %t1040 = load double, ptr %t1039, align 8
  %t1041 = fmul double %t1040, 0x4031666666666666
  %t1042 = fneg double %t1041
  %t1043 = fneg double %t1042
  %t1044 = getelementptr inbounds double, ptr %slots, i64 129
  store double %t1043, ptr %t1044, align 8
  %t1045 = getelementptr inbounds double, ptr %slots, i64 4
  %t1046 = load double, ptr %t1045, align 8
  %t1047 = fmul double %t1046, 0x4031cccccccccccd
  %t1048 = fneg double %t1047
  %t1049 = fneg double %t1048
  %t1050 = getelementptr inbounds double, ptr %slots, i64 130
  store double %t1049, ptr %t1050, align 8
  %t1051 = getelementptr inbounds double, ptr %slots, i64 4
  %t1052 = load double, ptr %t1051, align 8
  %t1053 = fmul double %t1052, 0x4032333333333333
  %t1054 = fneg double %t1053
  %t1055 = fneg double %t1054
  %t1056 = getelementptr inbounds double, ptr %slots, i64 131
  store double %t1055, ptr %t1056, align 8
  %t1057 = getelementptr inbounds double, ptr %slots, i64 4
  %t1058 = load double, ptr %t1057, align 8
  %t1059 = fmul double %t1058, 0x403299999999999a
  %t1060 = fneg double %t1059
  %t1061 = fneg double %t1060
  %t1062 = getelementptr inbounds double, ptr %slots, i64 132
  store double %t1061, ptr %t1062, align 8
  %t1063 = getelementptr inbounds double, ptr %slots, i64 4
  %t1064 = load double, ptr %t1063, align 8
  %t1065 = fmul double %t1064, 0x4033000000000000
  %t1066 = fneg double %t1065
  %t1067 = fneg double %t1066
  %t1068 = getelementptr inbounds double, ptr %slots, i64 133
  store double %t1067, ptr %t1068, align 8
  %t1069 = getelementptr inbounds double, ptr %slots, i64 4
  %t1070 = load double, ptr %t1069, align 8
  %t1071 = fmul double %t1070, 0x4033666666666666
  %t1072 = fneg double %t1071
  %t1073 = fneg double %t1072
  %t1074 = getelementptr inbounds double, ptr %slots, i64 134
  store double %t1073, ptr %t1074, align 8
  %t1075 = getelementptr inbounds double, ptr %slots, i64 4
  %t1076 = load double, ptr %t1075, align 8
  %t1077 = fmul double %t1076, 0x4033cccccccccccd
  %t1078 = fneg double %t1077
  %t1079 = fneg double %t1078
  %t1080 = getelementptr inbounds double, ptr %slots, i64 135
  store double %t1079, ptr %t1080, align 8
  %t1081 = getelementptr inbounds double, ptr %slots, i64 4
  %t1082 = load double, ptr %t1081, align 8
  %t1083 = fmul double %t1082, 0x4034333333333333
  %t1084 = fneg double %t1083
  %t1085 = fneg double %t1084
  %t1086 = getelementptr inbounds double, ptr %slots, i64 136
  store double %t1085, ptr %t1086, align 8
  %t1087 = getelementptr inbounds double, ptr %slots, i64 4
  %t1088 = load double, ptr %t1087, align 8
  %t1089 = fmul double %t1088, 0x403499999999999a
  %t1090 = fneg double %t1089
  %t1091 = fneg double %t1090
  %t1092 = getelementptr inbounds double, ptr %slots, i64 137
  store double %t1091, ptr %t1092, align 8
  %t1093 = getelementptr inbounds double, ptr %slots, i64 4
  %t1094 = load double, ptr %t1093, align 8
  %t1095 = fmul double %t1094, 0x4035000000000000
  %t1096 = fneg double %t1095
  %t1097 = fneg double %t1096
  %t1098 = getelementptr inbounds double, ptr %slots, i64 138
  store double %t1097, ptr %t1098, align 8
  %t1099 = getelementptr inbounds double, ptr %slots, i64 4
  %t1100 = load double, ptr %t1099, align 8
  %t1101 = fmul double %t1100, 0x4035666666666666
  %t1102 = fneg double %t1101
  %t1103 = fneg double %t1102
  %t1104 = getelementptr inbounds double, ptr %slots, i64 139
  store double %t1103, ptr %t1104, align 8
  %t1105 = getelementptr inbounds double, ptr %slots, i64 4
  %t1106 = load double, ptr %t1105, align 8
  %t1107 = fmul double %t1106, 0x4035cccccccccccd
  %t1108 = fneg double %t1107
  %t1109 = fneg double %t1108
  %t1110 = getelementptr inbounds double, ptr %slots, i64 140
  store double %t1109, ptr %t1110, align 8
  %t1111 = getelementptr inbounds double, ptr %slots, i64 4
  %t1112 = load double, ptr %t1111, align 8
  %t1113 = fmul double %t1112, 0x4036333333333333
  %t1114 = fneg double %t1113
  %t1115 = fneg double %t1114
  %t1116 = getelementptr inbounds double, ptr %slots, i64 141
  store double %t1115, ptr %t1116, align 8
  %t1117 = getelementptr inbounds double, ptr %slots, i64 4
  %t1118 = load double, ptr %t1117, align 8
  %t1119 = fmul double %t1118, 0x403699999999999a
  %t1120 = fneg double %t1119
  %t1121 = fneg double %t1120
  %t1122 = getelementptr inbounds double, ptr %slots, i64 142
  store double %t1121, ptr %t1122, align 8
  %t1123 = getelementptr inbounds double, ptr %slots, i64 4
  %t1124 = load double, ptr %t1123, align 8
  %t1125 = fmul double %t1124, 0x4037000000000000
  %t1126 = fneg double %t1125
  %t1127 = fneg double %t1126
  %t1128 = getelementptr inbounds double, ptr %slots, i64 143
  store double %t1127, ptr %t1128, align 8
  %t1129 = getelementptr inbounds double, ptr %slots, i64 4
  %t1130 = load double, ptr %t1129, align 8
  %t1131 = fmul double %t1130, 0x4037666666666666
  %t1132 = fneg double %t1131
  %t1133 = fneg double %t1132
  %t1134 = getelementptr inbounds double, ptr %slots, i64 144
  store double %t1133, ptr %t1134, align 8
  %t1135 = getelementptr inbounds double, ptr %slots, i64 4
  %t1136 = load double, ptr %t1135, align 8
  %t1137 = fmul double %t1136, 0x4037cccccccccccd
  %t1138 = fneg double %t1137
  %t1139 = fneg double %t1138
  %t1140 = getelementptr inbounds double, ptr %slots, i64 145
  store double %t1139, ptr %t1140, align 8
  %t1141 = getelementptr inbounds double, ptr %slots, i64 4
  %t1142 = load double, ptr %t1141, align 8
  %t1143 = fmul double %t1142, 0x4038333333333333
  %t1144 = fneg double %t1143
  %t1145 = fneg double %t1144
  %t1146 = getelementptr inbounds double, ptr %slots, i64 146
  store double %t1145, ptr %t1146, align 8
  %t1147 = getelementptr inbounds double, ptr %slots, i64 4
  %t1148 = load double, ptr %t1147, align 8
  %t1149 = fmul double %t1148, 0x403899999999999a
  %t1150 = fneg double %t1149
  %t1151 = fneg double %t1150
  %t1152 = getelementptr inbounds double, ptr %slots, i64 147
  store double %t1151, ptr %t1152, align 8
  %t1153 = getelementptr inbounds double, ptr %slots, i64 4
  %t1154 = load double, ptr %t1153, align 8
  %t1155 = fmul double %t1154, 0x4039000000000000
  %t1156 = fneg double %t1155
  %t1157 = fneg double %t1156
  %t1158 = getelementptr inbounds double, ptr %slots, i64 148
  store double %t1157, ptr %t1158, align 8
  %t1159 = getelementptr inbounds double, ptr %slots, i64 4
  %t1160 = load double, ptr %t1159, align 8
  %t1161 = fmul double %t1160, 0x4039666666666666
  %t1162 = fneg double %t1161
  %t1163 = fneg double %t1162
  %t1164 = getelementptr inbounds double, ptr %slots, i64 149
  store double %t1163, ptr %t1164, align 8
  %t1165 = getelementptr inbounds double, ptr %slots, i64 4
  %t1166 = load double, ptr %t1165, align 8
  %t1167 = fmul double %t1166, 0x4039cccccccccccd
  %t1168 = fneg double %t1167
  %t1169 = fneg double %t1168
  %t1170 = getelementptr inbounds double, ptr %slots, i64 150
  store double %t1169, ptr %t1170, align 8
  %t1171 = getelementptr inbounds double, ptr %slots, i64 4
  %t1172 = load double, ptr %t1171, align 8
  %t1173 = fmul double %t1172, 0x403a333333333333
  %t1174 = fneg double %t1173
  %t1175 = fneg double %t1174
  %t1176 = getelementptr inbounds double, ptr %slots, i64 151
  store double %t1175, ptr %t1176, align 8
  %t1177 = getelementptr inbounds double, ptr %slots, i64 4
  %t1178 = load double, ptr %t1177, align 8
  %t1179 = fmul double %t1178, 0x403a99999999999a
  %t1180 = fneg double %t1179
  %t1181 = fneg double %t1180
  %t1182 = getelementptr inbounds double, ptr %slots, i64 152
  store double %t1181, ptr %t1182, align 8
  %t1183 = getelementptr inbounds ptr, ptr %arrays, i64 137
  %t1184 = load ptr, ptr %t1183, align 8
  %t1185 = bitcast double %t802 to i64
  %t1186 = getelementptr inbounds i64, ptr %t1184, i64 0
  store i64 %t1185, ptr %t1186, align 8
  %t1187 = bitcast double %t808 to i64
  %t1188 = getelementptr inbounds i64, ptr %t1184, i64 1
  store i64 %t1187, ptr %t1188, align 8
  %t1189 = bitcast double %t814 to i64
  %t1190 = getelementptr inbounds i64, ptr %t1184, i64 2
  store i64 %t1189, ptr %t1190, align 8
  %t1191 = bitcast double %t820 to i64
  %t1192 = getelementptr inbounds i64, ptr %t1184, i64 3
  store i64 %t1191, ptr %t1192, align 8
  %t1193 = bitcast double %t826 to i64
  %t1194 = getelementptr inbounds i64, ptr %t1184, i64 4
  store i64 %t1193, ptr %t1194, align 8
  %t1195 = bitcast double %t832 to i64
  %t1196 = getelementptr inbounds i64, ptr %t1184, i64 5
  store i64 %t1195, ptr %t1196, align 8
  %t1197 = bitcast double %t838 to i64
  %t1198 = getelementptr inbounds i64, ptr %t1184, i64 6
  store i64 %t1197, ptr %t1198, align 8
  %t1199 = bitcast double %t844 to i64
  %t1200 = getelementptr inbounds i64, ptr %t1184, i64 7
  store i64 %t1199, ptr %t1200, align 8
  %t1201 = bitcast double %t850 to i64
  %t1202 = getelementptr inbounds i64, ptr %t1184, i64 8
  store i64 %t1201, ptr %t1202, align 8
  %t1203 = bitcast double %t856 to i64
  %t1204 = getelementptr inbounds i64, ptr %t1184, i64 9
  store i64 %t1203, ptr %t1204, align 8
  %t1205 = bitcast double %t862 to i64
  %t1206 = getelementptr inbounds i64, ptr %t1184, i64 10
  store i64 %t1205, ptr %t1206, align 8
  %t1207 = bitcast double %t868 to i64
  %t1208 = getelementptr inbounds i64, ptr %t1184, i64 11
  store i64 %t1207, ptr %t1208, align 8
  %t1209 = bitcast double %t874 to i64
  %t1210 = getelementptr inbounds i64, ptr %t1184, i64 12
  store i64 %t1209, ptr %t1210, align 8
  %t1211 = bitcast double %t880 to i64
  %t1212 = getelementptr inbounds i64, ptr %t1184, i64 13
  store i64 %t1211, ptr %t1212, align 8
  %t1213 = bitcast double %t886 to i64
  %t1214 = getelementptr inbounds i64, ptr %t1184, i64 14
  store i64 %t1213, ptr %t1214, align 8
  %t1215 = bitcast double %t892 to i64
  %t1216 = getelementptr inbounds i64, ptr %t1184, i64 15
  store i64 %t1215, ptr %t1216, align 8
  %t1217 = bitcast double %t898 to i64
  %t1218 = getelementptr inbounds i64, ptr %t1184, i64 16
  store i64 %t1217, ptr %t1218, align 8
  %t1219 = bitcast double %t904 to i64
  %t1220 = getelementptr inbounds i64, ptr %t1184, i64 17
  store i64 %t1219, ptr %t1220, align 8
  %t1221 = bitcast double %t910 to i64
  %t1222 = getelementptr inbounds i64, ptr %t1184, i64 18
  store i64 %t1221, ptr %t1222, align 8
  %t1223 = bitcast double %t916 to i64
  %t1224 = getelementptr inbounds i64, ptr %t1184, i64 19
  store i64 %t1223, ptr %t1224, align 8
  %t1225 = bitcast double %t922 to i64
  %t1226 = getelementptr inbounds i64, ptr %t1184, i64 20
  store i64 %t1225, ptr %t1226, align 8
  %t1227 = bitcast double %t928 to i64
  %t1228 = getelementptr inbounds i64, ptr %t1184, i64 21
  store i64 %t1227, ptr %t1228, align 8
  %t1229 = bitcast double %t934 to i64
  %t1230 = getelementptr inbounds i64, ptr %t1184, i64 22
  store i64 %t1229, ptr %t1230, align 8
  %t1231 = bitcast double %t940 to i64
  %t1232 = getelementptr inbounds i64, ptr %t1184, i64 23
  store i64 %t1231, ptr %t1232, align 8
  %t1233 = bitcast double %t946 to i64
  %t1234 = getelementptr inbounds i64, ptr %t1184, i64 24
  store i64 %t1233, ptr %t1234, align 8
  %t1235 = bitcast double %t952 to i64
  %t1236 = getelementptr inbounds i64, ptr %t1184, i64 25
  store i64 %t1235, ptr %t1236, align 8
  %t1237 = bitcast double %t958 to i64
  %t1238 = getelementptr inbounds i64, ptr %t1184, i64 26
  store i64 %t1237, ptr %t1238, align 8
  %t1239 = bitcast double %t964 to i64
  %t1240 = getelementptr inbounds i64, ptr %t1184, i64 27
  store i64 %t1239, ptr %t1240, align 8
  %t1241 = bitcast double %t970 to i64
  %t1242 = getelementptr inbounds i64, ptr %t1184, i64 28
  store i64 %t1241, ptr %t1242, align 8
  %t1243 = bitcast double %t976 to i64
  %t1244 = getelementptr inbounds i64, ptr %t1184, i64 29
  store i64 %t1243, ptr %t1244, align 8
  %t1245 = bitcast double %t982 to i64
  %t1246 = getelementptr inbounds i64, ptr %t1184, i64 30
  store i64 %t1245, ptr %t1246, align 8
  %t1247 = bitcast double %t988 to i64
  %t1248 = getelementptr inbounds i64, ptr %t1184, i64 31
  store i64 %t1247, ptr %t1248, align 8
  %t1249 = bitcast double %t994 to i64
  %t1250 = getelementptr inbounds i64, ptr %t1184, i64 32
  store i64 %t1249, ptr %t1250, align 8
  %t1251 = bitcast double %t1000 to i64
  %t1252 = getelementptr inbounds i64, ptr %t1184, i64 33
  store i64 %t1251, ptr %t1252, align 8
  %t1253 = bitcast double %t1006 to i64
  %t1254 = getelementptr inbounds i64, ptr %t1184, i64 34
  store i64 %t1253, ptr %t1254, align 8
  %t1255 = bitcast double %t1012 to i64
  %t1256 = getelementptr inbounds i64, ptr %t1184, i64 35
  store i64 %t1255, ptr %t1256, align 8
  %t1257 = bitcast double %t1018 to i64
  %t1258 = getelementptr inbounds i64, ptr %t1184, i64 36
  store i64 %t1257, ptr %t1258, align 8
  %t1259 = bitcast double %t1024 to i64
  %t1260 = getelementptr inbounds i64, ptr %t1184, i64 37
  store i64 %t1259, ptr %t1260, align 8
  %t1261 = bitcast double %t1030 to i64
  %t1262 = getelementptr inbounds i64, ptr %t1184, i64 38
  store i64 %t1261, ptr %t1262, align 8
  %t1263 = bitcast double %t1036 to i64
  %t1264 = getelementptr inbounds i64, ptr %t1184, i64 39
  store i64 %t1263, ptr %t1264, align 8
  %t1265 = bitcast double %t1042 to i64
  %t1266 = getelementptr inbounds i64, ptr %t1184, i64 40
  store i64 %t1265, ptr %t1266, align 8
  %t1267 = bitcast double %t1048 to i64
  %t1268 = getelementptr inbounds i64, ptr %t1184, i64 41
  store i64 %t1267, ptr %t1268, align 8
  %t1269 = bitcast double %t1054 to i64
  %t1270 = getelementptr inbounds i64, ptr %t1184, i64 42
  store i64 %t1269, ptr %t1270, align 8
  %t1271 = bitcast double %t1060 to i64
  %t1272 = getelementptr inbounds i64, ptr %t1184, i64 43
  store i64 %t1271, ptr %t1272, align 8
  %t1273 = bitcast double %t1066 to i64
  %t1274 = getelementptr inbounds i64, ptr %t1184, i64 44
  store i64 %t1273, ptr %t1274, align 8
  %t1275 = bitcast double %t1072 to i64
  %t1276 = getelementptr inbounds i64, ptr %t1184, i64 45
  store i64 %t1275, ptr %t1276, align 8
  %t1277 = bitcast double %t1078 to i64
  %t1278 = getelementptr inbounds i64, ptr %t1184, i64 46
  store i64 %t1277, ptr %t1278, align 8
  %t1279 = bitcast double %t1084 to i64
  %t1280 = getelementptr inbounds i64, ptr %t1184, i64 47
  store i64 %t1279, ptr %t1280, align 8
  %t1281 = bitcast double %t1090 to i64
  %t1282 = getelementptr inbounds i64, ptr %t1184, i64 48
  store i64 %t1281, ptr %t1282, align 8
  %t1283 = bitcast double %t1096 to i64
  %t1284 = getelementptr inbounds i64, ptr %t1184, i64 49
  store i64 %t1283, ptr %t1284, align 8
  %t1285 = bitcast double %t1102 to i64
  %t1286 = getelementptr inbounds i64, ptr %t1184, i64 50
  store i64 %t1285, ptr %t1286, align 8
  %t1287 = bitcast double %t1108 to i64
  %t1288 = getelementptr inbounds i64, ptr %t1184, i64 51
  store i64 %t1287, ptr %t1288, align 8
  %t1289 = bitcast double %t1114 to i64
  %t1290 = getelementptr inbounds i64, ptr %t1184, i64 52
  store i64 %t1289, ptr %t1290, align 8
  %t1291 = bitcast double %t1120 to i64
  %t1292 = getelementptr inbounds i64, ptr %t1184, i64 53
  store i64 %t1291, ptr %t1292, align 8
  %t1293 = bitcast double %t1126 to i64
  %t1294 = getelementptr inbounds i64, ptr %t1184, i64 54
  store i64 %t1293, ptr %t1294, align 8
  %t1295 = bitcast double %t1132 to i64
  %t1296 = getelementptr inbounds i64, ptr %t1184, i64 55
  store i64 %t1295, ptr %t1296, align 8
  %t1297 = bitcast double %t1138 to i64
  %t1298 = getelementptr inbounds i64, ptr %t1184, i64 56
  store i64 %t1297, ptr %t1298, align 8
  %t1299 = bitcast double %t1144 to i64
  %t1300 = getelementptr inbounds i64, ptr %t1184, i64 57
  store i64 %t1299, ptr %t1300, align 8
  %t1301 = bitcast double %t1150 to i64
  %t1302 = getelementptr inbounds i64, ptr %t1184, i64 58
  store i64 %t1301, ptr %t1302, align 8
  %t1303 = bitcast double %t1156 to i64
  %t1304 = getelementptr inbounds i64, ptr %t1184, i64 59
  store i64 %t1303, ptr %t1304, align 8
  %t1305 = bitcast double %t1162 to i64
  %t1306 = getelementptr inbounds i64, ptr %t1184, i64 60
  store i64 %t1305, ptr %t1306, align 8
  %t1307 = bitcast double %t1168 to i64
  %t1308 = getelementptr inbounds i64, ptr %t1184, i64 61
  store i64 %t1307, ptr %t1308, align 8
  %t1309 = bitcast double %t1174 to i64
  %t1310 = getelementptr inbounds i64, ptr %t1184, i64 62
  store i64 %t1309, ptr %t1310, align 8
  %t1311 = bitcast double %t1180 to i64
  %t1312 = getelementptr inbounds i64, ptr %t1184, i64 63
  store i64 %t1311, ptr %t1312, align 8
  %t1313 = getelementptr inbounds ptr, ptr %arrays, i64 138
  %t1314 = load ptr, ptr %t1313, align 8
  %t1315 = bitcast double %t34 to i64
  %t1316 = getelementptr inbounds i64, ptr %t1314, i64 0
  store i64 %t1315, ptr %t1316, align 8
  %t1317 = bitcast double %t46 to i64
  %t1318 = getelementptr inbounds i64, ptr %t1314, i64 1
  store i64 %t1317, ptr %t1318, align 8
  %t1319 = bitcast double %t58 to i64
  %t1320 = getelementptr inbounds i64, ptr %t1314, i64 2
  store i64 %t1319, ptr %t1320, align 8
  %t1321 = bitcast double %t70 to i64
  %t1322 = getelementptr inbounds i64, ptr %t1314, i64 3
  store i64 %t1321, ptr %t1322, align 8
  %t1323 = bitcast double %t82 to i64
  %t1324 = getelementptr inbounds i64, ptr %t1314, i64 4
  store i64 %t1323, ptr %t1324, align 8
  %t1325 = bitcast double %t94 to i64
  %t1326 = getelementptr inbounds i64, ptr %t1314, i64 5
  store i64 %t1325, ptr %t1326, align 8
  %t1327 = bitcast double %t106 to i64
  %t1328 = getelementptr inbounds i64, ptr %t1314, i64 6
  store i64 %t1327, ptr %t1328, align 8
  %t1329 = bitcast double %t118 to i64
  %t1330 = getelementptr inbounds i64, ptr %t1314, i64 7
  store i64 %t1329, ptr %t1330, align 8
  %t1331 = bitcast double %t130 to i64
  %t1332 = getelementptr inbounds i64, ptr %t1314, i64 8
  store i64 %t1331, ptr %t1332, align 8
  %t1333 = bitcast double %t142 to i64
  %t1334 = getelementptr inbounds i64, ptr %t1314, i64 9
  store i64 %t1333, ptr %t1334, align 8
  %t1335 = bitcast double %t154 to i64
  %t1336 = getelementptr inbounds i64, ptr %t1314, i64 10
  store i64 %t1335, ptr %t1336, align 8
  %t1337 = bitcast double %t166 to i64
  %t1338 = getelementptr inbounds i64, ptr %t1314, i64 11
  store i64 %t1337, ptr %t1338, align 8
  %t1339 = bitcast double %t178 to i64
  %t1340 = getelementptr inbounds i64, ptr %t1314, i64 12
  store i64 %t1339, ptr %t1340, align 8
  %t1341 = bitcast double %t190 to i64
  %t1342 = getelementptr inbounds i64, ptr %t1314, i64 13
  store i64 %t1341, ptr %t1342, align 8
  %t1343 = bitcast double %t202 to i64
  %t1344 = getelementptr inbounds i64, ptr %t1314, i64 14
  store i64 %t1343, ptr %t1344, align 8
  %t1345 = bitcast double %t214 to i64
  %t1346 = getelementptr inbounds i64, ptr %t1314, i64 15
  store i64 %t1345, ptr %t1346, align 8
  %t1347 = bitcast double %t226 to i64
  %t1348 = getelementptr inbounds i64, ptr %t1314, i64 16
  store i64 %t1347, ptr %t1348, align 8
  %t1349 = bitcast double %t238 to i64
  %t1350 = getelementptr inbounds i64, ptr %t1314, i64 17
  store i64 %t1349, ptr %t1350, align 8
  %t1351 = bitcast double %t250 to i64
  %t1352 = getelementptr inbounds i64, ptr %t1314, i64 18
  store i64 %t1351, ptr %t1352, align 8
  %t1353 = bitcast double %t262 to i64
  %t1354 = getelementptr inbounds i64, ptr %t1314, i64 19
  store i64 %t1353, ptr %t1354, align 8
  %t1355 = bitcast double %t274 to i64
  %t1356 = getelementptr inbounds i64, ptr %t1314, i64 20
  store i64 %t1355, ptr %t1356, align 8
  %t1357 = bitcast double %t286 to i64
  %t1358 = getelementptr inbounds i64, ptr %t1314, i64 21
  store i64 %t1357, ptr %t1358, align 8
  %t1359 = bitcast double %t298 to i64
  %t1360 = getelementptr inbounds i64, ptr %t1314, i64 22
  store i64 %t1359, ptr %t1360, align 8
  %t1361 = bitcast double %t310 to i64
  %t1362 = getelementptr inbounds i64, ptr %t1314, i64 23
  store i64 %t1361, ptr %t1362, align 8
  %t1363 = bitcast double %t322 to i64
  %t1364 = getelementptr inbounds i64, ptr %t1314, i64 24
  store i64 %t1363, ptr %t1364, align 8
  %t1365 = bitcast double %t334 to i64
  %t1366 = getelementptr inbounds i64, ptr %t1314, i64 25
  store i64 %t1365, ptr %t1366, align 8
  %t1367 = bitcast double %t346 to i64
  %t1368 = getelementptr inbounds i64, ptr %t1314, i64 26
  store i64 %t1367, ptr %t1368, align 8
  %t1369 = bitcast double %t358 to i64
  %t1370 = getelementptr inbounds i64, ptr %t1314, i64 27
  store i64 %t1369, ptr %t1370, align 8
  %t1371 = bitcast double %t370 to i64
  %t1372 = getelementptr inbounds i64, ptr %t1314, i64 28
  store i64 %t1371, ptr %t1372, align 8
  %t1373 = bitcast double %t382 to i64
  %t1374 = getelementptr inbounds i64, ptr %t1314, i64 29
  store i64 %t1373, ptr %t1374, align 8
  %t1375 = bitcast double %t394 to i64
  %t1376 = getelementptr inbounds i64, ptr %t1314, i64 30
  store i64 %t1375, ptr %t1376, align 8
  %t1377 = bitcast double %t406 to i64
  %t1378 = getelementptr inbounds i64, ptr %t1314, i64 31
  store i64 %t1377, ptr %t1378, align 8
  %t1379 = bitcast double %t418 to i64
  %t1380 = getelementptr inbounds i64, ptr %t1314, i64 32
  store i64 %t1379, ptr %t1380, align 8
  %t1381 = bitcast double %t430 to i64
  %t1382 = getelementptr inbounds i64, ptr %t1314, i64 33
  store i64 %t1381, ptr %t1382, align 8
  %t1383 = bitcast double %t442 to i64
  %t1384 = getelementptr inbounds i64, ptr %t1314, i64 34
  store i64 %t1383, ptr %t1384, align 8
  %t1385 = bitcast double %t454 to i64
  %t1386 = getelementptr inbounds i64, ptr %t1314, i64 35
  store i64 %t1385, ptr %t1386, align 8
  %t1387 = bitcast double %t466 to i64
  %t1388 = getelementptr inbounds i64, ptr %t1314, i64 36
  store i64 %t1387, ptr %t1388, align 8
  %t1389 = bitcast double %t478 to i64
  %t1390 = getelementptr inbounds i64, ptr %t1314, i64 37
  store i64 %t1389, ptr %t1390, align 8
  %t1391 = bitcast double %t490 to i64
  %t1392 = getelementptr inbounds i64, ptr %t1314, i64 38
  store i64 %t1391, ptr %t1392, align 8
  %t1393 = bitcast double %t502 to i64
  %t1394 = getelementptr inbounds i64, ptr %t1314, i64 39
  store i64 %t1393, ptr %t1394, align 8
  %t1395 = bitcast double %t514 to i64
  %t1396 = getelementptr inbounds i64, ptr %t1314, i64 40
  store i64 %t1395, ptr %t1396, align 8
  %t1397 = bitcast double %t526 to i64
  %t1398 = getelementptr inbounds i64, ptr %t1314, i64 41
  store i64 %t1397, ptr %t1398, align 8
  %t1399 = bitcast double %t538 to i64
  %t1400 = getelementptr inbounds i64, ptr %t1314, i64 42
  store i64 %t1399, ptr %t1400, align 8
  %t1401 = bitcast double %t550 to i64
  %t1402 = getelementptr inbounds i64, ptr %t1314, i64 43
  store i64 %t1401, ptr %t1402, align 8
  %t1403 = bitcast double %t562 to i64
  %t1404 = getelementptr inbounds i64, ptr %t1314, i64 44
  store i64 %t1403, ptr %t1404, align 8
  %t1405 = bitcast double %t574 to i64
  %t1406 = getelementptr inbounds i64, ptr %t1314, i64 45
  store i64 %t1405, ptr %t1406, align 8
  %t1407 = bitcast double %t586 to i64
  %t1408 = getelementptr inbounds i64, ptr %t1314, i64 46
  store i64 %t1407, ptr %t1408, align 8
  %t1409 = bitcast double %t598 to i64
  %t1410 = getelementptr inbounds i64, ptr %t1314, i64 47
  store i64 %t1409, ptr %t1410, align 8
  %t1411 = bitcast double %t610 to i64
  %t1412 = getelementptr inbounds i64, ptr %t1314, i64 48
  store i64 %t1411, ptr %t1412, align 8
  %t1413 = bitcast double %t622 to i64
  %t1414 = getelementptr inbounds i64, ptr %t1314, i64 49
  store i64 %t1413, ptr %t1414, align 8
  %t1415 = bitcast double %t634 to i64
  %t1416 = getelementptr inbounds i64, ptr %t1314, i64 50
  store i64 %t1415, ptr %t1416, align 8
  %t1417 = bitcast double %t646 to i64
  %t1418 = getelementptr inbounds i64, ptr %t1314, i64 51
  store i64 %t1417, ptr %t1418, align 8
  %t1419 = bitcast double %t658 to i64
  %t1420 = getelementptr inbounds i64, ptr %t1314, i64 52
  store i64 %t1419, ptr %t1420, align 8
  %t1421 = bitcast double %t670 to i64
  %t1422 = getelementptr inbounds i64, ptr %t1314, i64 53
  store i64 %t1421, ptr %t1422, align 8
  %t1423 = bitcast double %t682 to i64
  %t1424 = getelementptr inbounds i64, ptr %t1314, i64 54
  store i64 %t1423, ptr %t1424, align 8
  %t1425 = bitcast double %t694 to i64
  %t1426 = getelementptr inbounds i64, ptr %t1314, i64 55
  store i64 %t1425, ptr %t1426, align 8
  %t1427 = bitcast double %t706 to i64
  %t1428 = getelementptr inbounds i64, ptr %t1314, i64 56
  store i64 %t1427, ptr %t1428, align 8
  %t1429 = bitcast double %t718 to i64
  %t1430 = getelementptr inbounds i64, ptr %t1314, i64 57
  store i64 %t1429, ptr %t1430, align 8
  %t1431 = bitcast double %t730 to i64
  %t1432 = getelementptr inbounds i64, ptr %t1314, i64 58
  store i64 %t1431, ptr %t1432, align 8
  %t1433 = bitcast double %t742 to i64
  %t1434 = getelementptr inbounds i64, ptr %t1314, i64 59
  store i64 %t1433, ptr %t1434, align 8
  %t1435 = bitcast double %t754 to i64
  %t1436 = getelementptr inbounds i64, ptr %t1314, i64 60
  store i64 %t1435, ptr %t1436, align 8
  %t1437 = bitcast double %t766 to i64
  %t1438 = getelementptr inbounds i64, ptr %t1314, i64 61
  store i64 %t1437, ptr %t1438, align 8
  %t1439 = bitcast double %t778 to i64
  %t1440 = getelementptr inbounds i64, ptr %t1314, i64 62
  store i64 %t1439, ptr %t1440, align 8
  %t1441 = bitcast double %t790 to i64
  %t1442 = getelementptr inbounds i64, ptr %t1314, i64 63
  store i64 %t1441, ptr %t1442, align 8
  %t1443 = fptosi double %t3 to i64
  %t1444 = sitofp i64 %t1443 to double
  %t1445 = getelementptr inbounds double, ptr %slots, i64 153
  store double %t1444, ptr %t1445, align 8
  %t1446 = fptosi double %t9 to i64
  %t1447 = sitofp i64 %t1446 to double
  %t1448 = getelementptr inbounds double, ptr %slots, i64 154
  store double %t1447, ptr %t1448, align 8
  %t1449 = fptosi double %t3 to i64
  %t1450 = sitofp i64 %t1449 to double
  %t1451 = getelementptr inbounds double, ptr %slots, i64 155
  store double %t1450, ptr %t1451, align 8
  %t1452 = fptosi double %t9 to i64
  %t1453 = sitofp i64 %t1452 to double
  %t1454 = getelementptr inbounds double, ptr %slots, i64 156
  store double %t1453, ptr %t1454, align 8
  %t1455 = getelementptr inbounds double, ptr %output_buffer, i64 %s
  store double 0x0000000000000000, ptr %t1455, align 8
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
