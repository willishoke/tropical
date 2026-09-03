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
  %t189 = getelementptr inbounds double, ptr %slots, i64 3
  %t190 = load double, ptr %t189, align 8
  %t191 = fmul double 0x4031000000000000, %t190
  %t192 = fmul double 0x401921fb54442d18, %t191
  %t193 = fdiv double %t192, 0x401921fb54442d18
  %t194 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t195 = select i1 %t194, double 0x0000000000000000, double %t193
  %t196 = fmul double %t195, 0x41f0000000000000
  %t197 = fdiv double %t196, %sampleRate
  %t198 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t199 = select i1 %t198, double 0x0000000000000000, double %t197
  %t200 = getelementptr inbounds double, ptr %slots, i64 3
  %t201 = load double, ptr %t200, align 8
  %t202 = fmul double 0x4032000000000000, %t201
  %t203 = fmul double 0x401921fb54442d18, %t202
  %t204 = fdiv double %t203, 0x401921fb54442d18
  %t205 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t206 = select i1 %t205, double 0x0000000000000000, double %t204
  %t207 = fmul double %t206, 0x41f0000000000000
  %t208 = fdiv double %t207, %sampleRate
  %t209 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t210 = select i1 %t209, double 0x0000000000000000, double %t208
  %t211 = getelementptr inbounds double, ptr %slots, i64 3
  %t212 = load double, ptr %t211, align 8
  %t213 = fmul double 0x4033000000000000, %t212
  %t214 = fmul double 0x401921fb54442d18, %t213
  %t215 = fdiv double %t214, 0x401921fb54442d18
  %t216 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t217 = select i1 %t216, double 0x0000000000000000, double %t215
  %t218 = fmul double %t217, 0x41f0000000000000
  %t219 = fdiv double %t218, %sampleRate
  %t220 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t221 = select i1 %t220, double 0x0000000000000000, double %t219
  %t222 = getelementptr inbounds double, ptr %slots, i64 3
  %t223 = load double, ptr %t222, align 8
  %t224 = fmul double 0x4034000000000000, %t223
  %t225 = fmul double 0x401921fb54442d18, %t224
  %t226 = fdiv double %t225, 0x401921fb54442d18
  %t227 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t228 = select i1 %t227, double 0x0000000000000000, double %t226
  %t229 = fmul double %t228, 0x41f0000000000000
  %t230 = fdiv double %t229, %sampleRate
  %t231 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t232 = select i1 %t231, double 0x0000000000000000, double %t230
  %t233 = getelementptr inbounds double, ptr %slots, i64 3
  %t234 = load double, ptr %t233, align 8
  %t235 = fmul double 0x4035000000000000, %t234
  %t236 = fmul double 0x401921fb54442d18, %t235
  %t237 = fdiv double %t236, 0x401921fb54442d18
  %t238 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t239 = select i1 %t238, double 0x0000000000000000, double %t237
  %t240 = fmul double %t239, 0x41f0000000000000
  %t241 = fdiv double %t240, %sampleRate
  %t242 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t243 = select i1 %t242, double 0x0000000000000000, double %t241
  %t244 = getelementptr inbounds double, ptr %slots, i64 3
  %t245 = load double, ptr %t244, align 8
  %t246 = fmul double 0x4036000000000000, %t245
  %t247 = fmul double 0x401921fb54442d18, %t246
  %t248 = fdiv double %t247, 0x401921fb54442d18
  %t249 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t250 = select i1 %t249, double 0x0000000000000000, double %t248
  %t251 = fmul double %t250, 0x41f0000000000000
  %t252 = fdiv double %t251, %sampleRate
  %t253 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t254 = select i1 %t253, double 0x0000000000000000, double %t252
  %t255 = getelementptr inbounds double, ptr %slots, i64 3
  %t256 = load double, ptr %t255, align 8
  %t257 = fmul double 0x4037000000000000, %t256
  %t258 = fmul double 0x401921fb54442d18, %t257
  %t259 = fdiv double %t258, 0x401921fb54442d18
  %t260 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t261 = select i1 %t260, double 0x0000000000000000, double %t259
  %t262 = fmul double %t261, 0x41f0000000000000
  %t263 = fdiv double %t262, %sampleRate
  %t264 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t265 = select i1 %t264, double 0x0000000000000000, double %t263
  %t266 = getelementptr inbounds double, ptr %slots, i64 3
  %t267 = load double, ptr %t266, align 8
  %t268 = fmul double 0x4038000000000000, %t267
  %t269 = fmul double 0x401921fb54442d18, %t268
  %t270 = fdiv double %t269, 0x401921fb54442d18
  %t271 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t272 = select i1 %t271, double 0x0000000000000000, double %t270
  %t273 = fmul double %t272, 0x41f0000000000000
  %t274 = fdiv double %t273, %sampleRate
  %t275 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t276 = select i1 %t275, double 0x0000000000000000, double %t274
  %t277 = getelementptr inbounds double, ptr %slots, i64 3
  %t278 = load double, ptr %t277, align 8
  %t279 = fmul double 0x4039000000000000, %t278
  %t280 = fmul double 0x401921fb54442d18, %t279
  %t281 = fdiv double %t280, 0x401921fb54442d18
  %t282 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t283 = select i1 %t282, double 0x0000000000000000, double %t281
  %t284 = fmul double %t283, 0x41f0000000000000
  %t285 = fdiv double %t284, %sampleRate
  %t286 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t287 = select i1 %t286, double 0x0000000000000000, double %t285
  %t288 = getelementptr inbounds double, ptr %slots, i64 3
  %t289 = load double, ptr %t288, align 8
  %t290 = fmul double 0x403a000000000000, %t289
  %t291 = fmul double 0x401921fb54442d18, %t290
  %t292 = fdiv double %t291, 0x401921fb54442d18
  %t293 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t294 = select i1 %t293, double 0x0000000000000000, double %t292
  %t295 = fmul double %t294, 0x41f0000000000000
  %t296 = fdiv double %t295, %sampleRate
  %t297 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t298 = select i1 %t297, double 0x0000000000000000, double %t296
  %t299 = getelementptr inbounds double, ptr %slots, i64 3
  %t300 = load double, ptr %t299, align 8
  %t301 = fmul double 0x403b000000000000, %t300
  %t302 = fmul double 0x401921fb54442d18, %t301
  %t303 = fdiv double %t302, 0x401921fb54442d18
  %t304 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t305 = select i1 %t304, double 0x0000000000000000, double %t303
  %t306 = fmul double %t305, 0x41f0000000000000
  %t307 = fdiv double %t306, %sampleRate
  %t308 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t309 = select i1 %t308, double 0x0000000000000000, double %t307
  %t310 = getelementptr inbounds double, ptr %slots, i64 3
  %t311 = load double, ptr %t310, align 8
  %t312 = fmul double 0x403c000000000000, %t311
  %t313 = fmul double 0x401921fb54442d18, %t312
  %t314 = fdiv double %t313, 0x401921fb54442d18
  %t315 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t316 = select i1 %t315, double 0x0000000000000000, double %t314
  %t317 = fmul double %t316, 0x41f0000000000000
  %t318 = fdiv double %t317, %sampleRate
  %t319 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t320 = select i1 %t319, double 0x0000000000000000, double %t318
  %t321 = getelementptr inbounds double, ptr %slots, i64 3
  %t322 = load double, ptr %t321, align 8
  %t323 = fmul double 0x403d000000000000, %t322
  %t324 = fmul double 0x401921fb54442d18, %t323
  %t325 = fdiv double %t324, 0x401921fb54442d18
  %t326 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t327 = select i1 %t326, double 0x0000000000000000, double %t325
  %t328 = fmul double %t327, 0x41f0000000000000
  %t329 = fdiv double %t328, %sampleRate
  %t330 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t331 = select i1 %t330, double 0x0000000000000000, double %t329
  %t332 = getelementptr inbounds double, ptr %slots, i64 3
  %t333 = load double, ptr %t332, align 8
  %t334 = fmul double 0x403e000000000000, %t333
  %t335 = fmul double 0x401921fb54442d18, %t334
  %t336 = fdiv double %t335, 0x401921fb54442d18
  %t337 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t338 = select i1 %t337, double 0x0000000000000000, double %t336
  %t339 = fmul double %t338, 0x41f0000000000000
  %t340 = fdiv double %t339, %sampleRate
  %t341 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t342 = select i1 %t341, double 0x0000000000000000, double %t340
  %t343 = getelementptr inbounds double, ptr %slots, i64 3
  %t344 = load double, ptr %t343, align 8
  %t345 = fmul double 0x403f000000000000, %t344
  %t346 = fmul double 0x401921fb54442d18, %t345
  %t347 = fdiv double %t346, 0x401921fb54442d18
  %t348 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t349 = select i1 %t348, double 0x0000000000000000, double %t347
  %t350 = fmul double %t349, 0x41f0000000000000
  %t351 = fdiv double %t350, %sampleRate
  %t352 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t353 = select i1 %t352, double 0x0000000000000000, double %t351
  %t354 = getelementptr inbounds double, ptr %slots, i64 3
  %t355 = load double, ptr %t354, align 8
  %t356 = fmul double 0x4040000000000000, %t355
  %t357 = fmul double 0x401921fb54442d18, %t356
  %t358 = fdiv double %t357, 0x401921fb54442d18
  %t359 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t360 = select i1 %t359, double 0x0000000000000000, double %t358
  %t361 = fmul double %t360, 0x41f0000000000000
  %t362 = fdiv double %t361, %sampleRate
  %t363 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t364 = select i1 %t363, double 0x0000000000000000, double %t362
  %t365 = getelementptr inbounds double, ptr %slots, i64 3
  %t366 = load double, ptr %t365, align 8
  %t367 = fmul double 0x4040800000000000, %t366
  %t368 = fmul double 0x401921fb54442d18, %t367
  %t369 = fdiv double %t368, 0x401921fb54442d18
  %t370 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t371 = select i1 %t370, double 0x0000000000000000, double %t369
  %t372 = fmul double %t371, 0x41f0000000000000
  %t373 = fdiv double %t372, %sampleRate
  %t374 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t375 = select i1 %t374, double 0x0000000000000000, double %t373
  %t376 = getelementptr inbounds double, ptr %slots, i64 3
  %t377 = load double, ptr %t376, align 8
  %t378 = fmul double 0x4041000000000000, %t377
  %t379 = fmul double 0x401921fb54442d18, %t378
  %t380 = fdiv double %t379, 0x401921fb54442d18
  %t381 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t382 = select i1 %t381, double 0x0000000000000000, double %t380
  %t383 = fmul double %t382, 0x41f0000000000000
  %t384 = fdiv double %t383, %sampleRate
  %t385 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t386 = select i1 %t385, double 0x0000000000000000, double %t384
  %t387 = getelementptr inbounds double, ptr %slots, i64 3
  %t388 = load double, ptr %t387, align 8
  %t389 = fmul double 0x4041800000000000, %t388
  %t390 = fmul double 0x401921fb54442d18, %t389
  %t391 = fdiv double %t390, 0x401921fb54442d18
  %t392 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t393 = select i1 %t392, double 0x0000000000000000, double %t391
  %t394 = fmul double %t393, 0x41f0000000000000
  %t395 = fdiv double %t394, %sampleRate
  %t396 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t397 = select i1 %t396, double 0x0000000000000000, double %t395
  %t398 = getelementptr inbounds double, ptr %slots, i64 3
  %t399 = load double, ptr %t398, align 8
  %t400 = fmul double 0x4042000000000000, %t399
  %t401 = fmul double 0x401921fb54442d18, %t400
  %t402 = fdiv double %t401, 0x401921fb54442d18
  %t403 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t404 = select i1 %t403, double 0x0000000000000000, double %t402
  %t405 = fmul double %t404, 0x41f0000000000000
  %t406 = fdiv double %t405, %sampleRate
  %t407 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t408 = select i1 %t407, double 0x0000000000000000, double %t406
  %t409 = getelementptr inbounds double, ptr %slots, i64 3
  %t410 = load double, ptr %t409, align 8
  %t411 = fmul double 0x4042800000000000, %t410
  %t412 = fmul double 0x401921fb54442d18, %t411
  %t413 = fdiv double %t412, 0x401921fb54442d18
  %t414 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t415 = select i1 %t414, double 0x0000000000000000, double %t413
  %t416 = fmul double %t415, 0x41f0000000000000
  %t417 = fdiv double %t416, %sampleRate
  %t418 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t419 = select i1 %t418, double 0x0000000000000000, double %t417
  %t420 = getelementptr inbounds double, ptr %slots, i64 3
  %t421 = load double, ptr %t420, align 8
  %t422 = fmul double 0x4043000000000000, %t421
  %t423 = fmul double 0x401921fb54442d18, %t422
  %t424 = fdiv double %t423, 0x401921fb54442d18
  %t425 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t426 = select i1 %t425, double 0x0000000000000000, double %t424
  %t427 = fmul double %t426, 0x41f0000000000000
  %t428 = fdiv double %t427, %sampleRate
  %t429 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t430 = select i1 %t429, double 0x0000000000000000, double %t428
  %t431 = getelementptr inbounds double, ptr %slots, i64 3
  %t432 = load double, ptr %t431, align 8
  %t433 = fmul double 0x4043800000000000, %t432
  %t434 = fmul double 0x401921fb54442d18, %t433
  %t435 = fdiv double %t434, 0x401921fb54442d18
  %t436 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t437 = select i1 %t436, double 0x0000000000000000, double %t435
  %t438 = fmul double %t437, 0x41f0000000000000
  %t439 = fdiv double %t438, %sampleRate
  %t440 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t441 = select i1 %t440, double 0x0000000000000000, double %t439
  %t442 = getelementptr inbounds double, ptr %slots, i64 3
  %t443 = load double, ptr %t442, align 8
  %t444 = fmul double 0x4044000000000000, %t443
  %t445 = fmul double 0x401921fb54442d18, %t444
  %t446 = fdiv double %t445, 0x401921fb54442d18
  %t447 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t448 = select i1 %t447, double 0x0000000000000000, double %t446
  %t449 = fmul double %t448, 0x41f0000000000000
  %t450 = fdiv double %t449, %sampleRate
  %t451 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t452 = select i1 %t451, double 0x0000000000000000, double %t450
  %t453 = getelementptr inbounds double, ptr %slots, i64 3
  %t454 = load double, ptr %t453, align 8
  %t455 = fmul double 0x4044800000000000, %t454
  %t456 = fmul double 0x401921fb54442d18, %t455
  %t457 = fdiv double %t456, 0x401921fb54442d18
  %t458 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t459 = select i1 %t458, double 0x0000000000000000, double %t457
  %t460 = fmul double %t459, 0x41f0000000000000
  %t461 = fdiv double %t460, %sampleRate
  %t462 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t463 = select i1 %t462, double 0x0000000000000000, double %t461
  %t464 = getelementptr inbounds double, ptr %slots, i64 3
  %t465 = load double, ptr %t464, align 8
  %t466 = fmul double 0x4045000000000000, %t465
  %t467 = fmul double 0x401921fb54442d18, %t466
  %t468 = fdiv double %t467, 0x401921fb54442d18
  %t469 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t470 = select i1 %t469, double 0x0000000000000000, double %t468
  %t471 = fmul double %t470, 0x41f0000000000000
  %t472 = fdiv double %t471, %sampleRate
  %t473 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t474 = select i1 %t473, double 0x0000000000000000, double %t472
  %t475 = getelementptr inbounds double, ptr %slots, i64 3
  %t476 = load double, ptr %t475, align 8
  %t477 = fmul double 0x4045800000000000, %t476
  %t478 = fmul double 0x401921fb54442d18, %t477
  %t479 = fdiv double %t478, 0x401921fb54442d18
  %t480 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t481 = select i1 %t480, double 0x0000000000000000, double %t479
  %t482 = fmul double %t481, 0x41f0000000000000
  %t483 = fdiv double %t482, %sampleRate
  %t484 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t485 = select i1 %t484, double 0x0000000000000000, double %t483
  %t486 = getelementptr inbounds double, ptr %slots, i64 3
  %t487 = load double, ptr %t486, align 8
  %t488 = fmul double 0x4046000000000000, %t487
  %t489 = fmul double 0x401921fb54442d18, %t488
  %t490 = fdiv double %t489, 0x401921fb54442d18
  %t491 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t492 = select i1 %t491, double 0x0000000000000000, double %t490
  %t493 = fmul double %t492, 0x41f0000000000000
  %t494 = fdiv double %t493, %sampleRate
  %t495 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t496 = select i1 %t495, double 0x0000000000000000, double %t494
  %t497 = getelementptr inbounds double, ptr %slots, i64 3
  %t498 = load double, ptr %t497, align 8
  %t499 = fmul double 0x4046800000000000, %t498
  %t500 = fmul double 0x401921fb54442d18, %t499
  %t501 = fdiv double %t500, 0x401921fb54442d18
  %t502 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t503 = select i1 %t502, double 0x0000000000000000, double %t501
  %t504 = fmul double %t503, 0x41f0000000000000
  %t505 = fdiv double %t504, %sampleRate
  %t506 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t507 = select i1 %t506, double 0x0000000000000000, double %t505
  %t508 = getelementptr inbounds double, ptr %slots, i64 3
  %t509 = load double, ptr %t508, align 8
  %t510 = fmul double 0x4047000000000000, %t509
  %t511 = fmul double 0x401921fb54442d18, %t510
  %t512 = fdiv double %t511, 0x401921fb54442d18
  %t513 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t514 = select i1 %t513, double 0x0000000000000000, double %t512
  %t515 = fmul double %t514, 0x41f0000000000000
  %t516 = fdiv double %t515, %sampleRate
  %t517 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t518 = select i1 %t517, double 0x0000000000000000, double %t516
  %t519 = getelementptr inbounds double, ptr %slots, i64 3
  %t520 = load double, ptr %t519, align 8
  %t521 = fmul double 0x4047800000000000, %t520
  %t522 = fmul double 0x401921fb54442d18, %t521
  %t523 = fdiv double %t522, 0x401921fb54442d18
  %t524 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t525 = select i1 %t524, double 0x0000000000000000, double %t523
  %t526 = fmul double %t525, 0x41f0000000000000
  %t527 = fdiv double %t526, %sampleRate
  %t528 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t529 = select i1 %t528, double 0x0000000000000000, double %t527
  %t530 = getelementptr inbounds double, ptr %slots, i64 3
  %t531 = load double, ptr %t530, align 8
  %t532 = fmul double 0x4048000000000000, %t531
  %t533 = fmul double 0x401921fb54442d18, %t532
  %t534 = fdiv double %t533, 0x401921fb54442d18
  %t535 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t536 = select i1 %t535, double 0x0000000000000000, double %t534
  %t537 = fmul double %t536, 0x41f0000000000000
  %t538 = fdiv double %t537, %sampleRate
  %t539 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t540 = select i1 %t539, double 0x0000000000000000, double %t538
  %t541 = getelementptr inbounds double, ptr %slots, i64 3
  %t542 = load double, ptr %t541, align 8
  %t543 = fmul double 0x4048800000000000, %t542
  %t544 = fmul double 0x401921fb54442d18, %t543
  %t545 = fdiv double %t544, 0x401921fb54442d18
  %t546 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t547 = select i1 %t546, double 0x0000000000000000, double %t545
  %t548 = fmul double %t547, 0x41f0000000000000
  %t549 = fdiv double %t548, %sampleRate
  %t550 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t551 = select i1 %t550, double 0x0000000000000000, double %t549
  %t552 = getelementptr inbounds double, ptr %slots, i64 3
  %t553 = load double, ptr %t552, align 8
  %t554 = fmul double 0x4049000000000000, %t553
  %t555 = fmul double 0x401921fb54442d18, %t554
  %t556 = fdiv double %t555, 0x401921fb54442d18
  %t557 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t558 = select i1 %t557, double 0x0000000000000000, double %t556
  %t559 = fmul double %t558, 0x41f0000000000000
  %t560 = fdiv double %t559, %sampleRate
  %t561 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t562 = select i1 %t561, double 0x0000000000000000, double %t560
  %t563 = getelementptr inbounds double, ptr %slots, i64 3
  %t564 = load double, ptr %t563, align 8
  %t565 = fmul double 0x4049800000000000, %t564
  %t566 = fmul double 0x401921fb54442d18, %t565
  %t567 = fdiv double %t566, 0x401921fb54442d18
  %t568 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t569 = select i1 %t568, double 0x0000000000000000, double %t567
  %t570 = fmul double %t569, 0x41f0000000000000
  %t571 = fdiv double %t570, %sampleRate
  %t572 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t573 = select i1 %t572, double 0x0000000000000000, double %t571
  %t574 = getelementptr inbounds double, ptr %slots, i64 3
  %t575 = load double, ptr %t574, align 8
  %t576 = fmul double 0x404a000000000000, %t575
  %t577 = fmul double 0x401921fb54442d18, %t576
  %t578 = fdiv double %t577, 0x401921fb54442d18
  %t579 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t580 = select i1 %t579, double 0x0000000000000000, double %t578
  %t581 = fmul double %t580, 0x41f0000000000000
  %t582 = fdiv double %t581, %sampleRate
  %t583 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t584 = select i1 %t583, double 0x0000000000000000, double %t582
  %t585 = getelementptr inbounds double, ptr %slots, i64 3
  %t586 = load double, ptr %t585, align 8
  %t587 = fmul double 0x404a800000000000, %t586
  %t588 = fmul double 0x401921fb54442d18, %t587
  %t589 = fdiv double %t588, 0x401921fb54442d18
  %t590 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t591 = select i1 %t590, double 0x0000000000000000, double %t589
  %t592 = fmul double %t591, 0x41f0000000000000
  %t593 = fdiv double %t592, %sampleRate
  %t594 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t595 = select i1 %t594, double 0x0000000000000000, double %t593
  %t596 = getelementptr inbounds double, ptr %slots, i64 3
  %t597 = load double, ptr %t596, align 8
  %t598 = fmul double 0x404b000000000000, %t597
  %t599 = fmul double 0x401921fb54442d18, %t598
  %t600 = fdiv double %t599, 0x401921fb54442d18
  %t601 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t602 = select i1 %t601, double 0x0000000000000000, double %t600
  %t603 = fmul double %t602, 0x41f0000000000000
  %t604 = fdiv double %t603, %sampleRate
  %t605 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t606 = select i1 %t605, double 0x0000000000000000, double %t604
  %t607 = getelementptr inbounds double, ptr %slots, i64 3
  %t608 = load double, ptr %t607, align 8
  %t609 = fmul double 0x404b800000000000, %t608
  %t610 = fmul double 0x401921fb54442d18, %t609
  %t611 = fdiv double %t610, 0x401921fb54442d18
  %t612 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t613 = select i1 %t612, double 0x0000000000000000, double %t611
  %t614 = fmul double %t613, 0x41f0000000000000
  %t615 = fdiv double %t614, %sampleRate
  %t616 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t617 = select i1 %t616, double 0x0000000000000000, double %t615
  %t618 = getelementptr inbounds double, ptr %slots, i64 3
  %t619 = load double, ptr %t618, align 8
  %t620 = fmul double 0x404c000000000000, %t619
  %t621 = fmul double 0x401921fb54442d18, %t620
  %t622 = fdiv double %t621, 0x401921fb54442d18
  %t623 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t624 = select i1 %t623, double 0x0000000000000000, double %t622
  %t625 = fmul double %t624, 0x41f0000000000000
  %t626 = fdiv double %t625, %sampleRate
  %t627 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t628 = select i1 %t627, double 0x0000000000000000, double %t626
  %t629 = getelementptr inbounds double, ptr %slots, i64 3
  %t630 = load double, ptr %t629, align 8
  %t631 = fmul double 0x404c800000000000, %t630
  %t632 = fmul double 0x401921fb54442d18, %t631
  %t633 = fdiv double %t632, 0x401921fb54442d18
  %t634 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t635 = select i1 %t634, double 0x0000000000000000, double %t633
  %t636 = fmul double %t635, 0x41f0000000000000
  %t637 = fdiv double %t636, %sampleRate
  %t638 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t639 = select i1 %t638, double 0x0000000000000000, double %t637
  %t640 = getelementptr inbounds double, ptr %slots, i64 3
  %t641 = load double, ptr %t640, align 8
  %t642 = fmul double 0x404d000000000000, %t641
  %t643 = fmul double 0x401921fb54442d18, %t642
  %t644 = fdiv double %t643, 0x401921fb54442d18
  %t645 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t646 = select i1 %t645, double 0x0000000000000000, double %t644
  %t647 = fmul double %t646, 0x41f0000000000000
  %t648 = fdiv double %t647, %sampleRate
  %t649 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t650 = select i1 %t649, double 0x0000000000000000, double %t648
  %t651 = getelementptr inbounds double, ptr %slots, i64 3
  %t652 = load double, ptr %t651, align 8
  %t653 = fmul double 0x404d800000000000, %t652
  %t654 = fmul double 0x401921fb54442d18, %t653
  %t655 = fdiv double %t654, 0x401921fb54442d18
  %t656 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t657 = select i1 %t656, double 0x0000000000000000, double %t655
  %t658 = fmul double %t657, 0x41f0000000000000
  %t659 = fdiv double %t658, %sampleRate
  %t660 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t661 = select i1 %t660, double 0x0000000000000000, double %t659
  %t662 = getelementptr inbounds double, ptr %slots, i64 3
  %t663 = load double, ptr %t662, align 8
  %t664 = fmul double 0x404e000000000000, %t663
  %t665 = fmul double 0x401921fb54442d18, %t664
  %t666 = fdiv double %t665, 0x401921fb54442d18
  %t667 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t668 = select i1 %t667, double 0x0000000000000000, double %t666
  %t669 = fmul double %t668, 0x41f0000000000000
  %t670 = fdiv double %t669, %sampleRate
  %t671 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t672 = select i1 %t671, double 0x0000000000000000, double %t670
  %t673 = getelementptr inbounds double, ptr %slots, i64 3
  %t674 = load double, ptr %t673, align 8
  %t675 = fmul double 0x404e800000000000, %t674
  %t676 = fmul double 0x401921fb54442d18, %t675
  %t677 = fdiv double %t676, 0x401921fb54442d18
  %t678 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t679 = select i1 %t678, double 0x0000000000000000, double %t677
  %t680 = fmul double %t679, 0x41f0000000000000
  %t681 = fdiv double %t680, %sampleRate
  %t682 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t683 = select i1 %t682, double 0x0000000000000000, double %t681
  %t684 = getelementptr inbounds double, ptr %slots, i64 3
  %t685 = load double, ptr %t684, align 8
  %t686 = fmul double 0x404f000000000000, %t685
  %t687 = fmul double 0x401921fb54442d18, %t686
  %t688 = fdiv double %t687, 0x401921fb54442d18
  %t689 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t690 = select i1 %t689, double 0x0000000000000000, double %t688
  %t691 = fmul double %t690, 0x41f0000000000000
  %t692 = fdiv double %t691, %sampleRate
  %t693 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t694 = select i1 %t693, double 0x0000000000000000, double %t692
  %t695 = getelementptr inbounds double, ptr %slots, i64 3
  %t696 = load double, ptr %t695, align 8
  %t697 = fmul double 0x404f800000000000, %t696
  %t698 = fmul double 0x401921fb54442d18, %t697
  %t699 = fdiv double %t698, 0x401921fb54442d18
  %t700 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t701 = select i1 %t700, double 0x0000000000000000, double %t699
  %t702 = fmul double %t701, 0x41f0000000000000
  %t703 = fdiv double %t702, %sampleRate
  %t704 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t705 = select i1 %t704, double 0x0000000000000000, double %t703
  %t706 = getelementptr inbounds double, ptr %slots, i64 3
  %t707 = load double, ptr %t706, align 8
  %t708 = fmul double 0x4050000000000000, %t707
  %t709 = fmul double 0x401921fb54442d18, %t708
  %t710 = fdiv double %t709, 0x401921fb54442d18
  %t711 = fcmp oeq double 0x401921fb54442d18, 0x0000000000000000
  %t712 = select i1 %t711, double 0x0000000000000000, double %t710
  %t713 = fmul double %t712, 0x41f0000000000000
  %t714 = fdiv double %t713, %sampleRate
  %t715 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t716 = select i1 %t715, double 0x0000000000000000, double %t714
  %t717 = getelementptr inbounds ptr, ptr %arrays, i64 0
  %t718 = load ptr, ptr %t717, align 8
  %t719 = bitcast double %t23 to i64
  %t720 = getelementptr inbounds i64, ptr %t718, i64 0
  store i64 %t719, ptr %t720, align 8
  %t721 = bitcast double %t34 to i64
  %t722 = getelementptr inbounds i64, ptr %t718, i64 1
  store i64 %t721, ptr %t722, align 8
  %t723 = bitcast double %t45 to i64
  %t724 = getelementptr inbounds i64, ptr %t718, i64 2
  store i64 %t723, ptr %t724, align 8
  %t725 = bitcast double %t56 to i64
  %t726 = getelementptr inbounds i64, ptr %t718, i64 3
  store i64 %t725, ptr %t726, align 8
  %t727 = bitcast double %t67 to i64
  %t728 = getelementptr inbounds i64, ptr %t718, i64 4
  store i64 %t727, ptr %t728, align 8
  %t729 = bitcast double %t78 to i64
  %t730 = getelementptr inbounds i64, ptr %t718, i64 5
  store i64 %t729, ptr %t730, align 8
  %t731 = bitcast double %t89 to i64
  %t732 = getelementptr inbounds i64, ptr %t718, i64 6
  store i64 %t731, ptr %t732, align 8
  %t733 = bitcast double %t100 to i64
  %t734 = getelementptr inbounds i64, ptr %t718, i64 7
  store i64 %t733, ptr %t734, align 8
  %t735 = bitcast double %t111 to i64
  %t736 = getelementptr inbounds i64, ptr %t718, i64 8
  store i64 %t735, ptr %t736, align 8
  %t737 = bitcast double %t122 to i64
  %t738 = getelementptr inbounds i64, ptr %t718, i64 9
  store i64 %t737, ptr %t738, align 8
  %t739 = bitcast double %t133 to i64
  %t740 = getelementptr inbounds i64, ptr %t718, i64 10
  store i64 %t739, ptr %t740, align 8
  %t741 = bitcast double %t144 to i64
  %t742 = getelementptr inbounds i64, ptr %t718, i64 11
  store i64 %t741, ptr %t742, align 8
  %t743 = bitcast double %t155 to i64
  %t744 = getelementptr inbounds i64, ptr %t718, i64 12
  store i64 %t743, ptr %t744, align 8
  %t745 = bitcast double %t166 to i64
  %t746 = getelementptr inbounds i64, ptr %t718, i64 13
  store i64 %t745, ptr %t746, align 8
  %t747 = bitcast double %t177 to i64
  %t748 = getelementptr inbounds i64, ptr %t718, i64 14
  store i64 %t747, ptr %t748, align 8
  %t749 = bitcast double %t188 to i64
  %t750 = getelementptr inbounds i64, ptr %t718, i64 15
  store i64 %t749, ptr %t750, align 8
  %t751 = bitcast double %t199 to i64
  %t752 = getelementptr inbounds i64, ptr %t718, i64 16
  store i64 %t751, ptr %t752, align 8
  %t753 = bitcast double %t210 to i64
  %t754 = getelementptr inbounds i64, ptr %t718, i64 17
  store i64 %t753, ptr %t754, align 8
  %t755 = bitcast double %t221 to i64
  %t756 = getelementptr inbounds i64, ptr %t718, i64 18
  store i64 %t755, ptr %t756, align 8
  %t757 = bitcast double %t232 to i64
  %t758 = getelementptr inbounds i64, ptr %t718, i64 19
  store i64 %t757, ptr %t758, align 8
  %t759 = bitcast double %t243 to i64
  %t760 = getelementptr inbounds i64, ptr %t718, i64 20
  store i64 %t759, ptr %t760, align 8
  %t761 = bitcast double %t254 to i64
  %t762 = getelementptr inbounds i64, ptr %t718, i64 21
  store i64 %t761, ptr %t762, align 8
  %t763 = bitcast double %t265 to i64
  %t764 = getelementptr inbounds i64, ptr %t718, i64 22
  store i64 %t763, ptr %t764, align 8
  %t765 = bitcast double %t276 to i64
  %t766 = getelementptr inbounds i64, ptr %t718, i64 23
  store i64 %t765, ptr %t766, align 8
  %t767 = bitcast double %t287 to i64
  %t768 = getelementptr inbounds i64, ptr %t718, i64 24
  store i64 %t767, ptr %t768, align 8
  %t769 = bitcast double %t298 to i64
  %t770 = getelementptr inbounds i64, ptr %t718, i64 25
  store i64 %t769, ptr %t770, align 8
  %t771 = bitcast double %t309 to i64
  %t772 = getelementptr inbounds i64, ptr %t718, i64 26
  store i64 %t771, ptr %t772, align 8
  %t773 = bitcast double %t320 to i64
  %t774 = getelementptr inbounds i64, ptr %t718, i64 27
  store i64 %t773, ptr %t774, align 8
  %t775 = bitcast double %t331 to i64
  %t776 = getelementptr inbounds i64, ptr %t718, i64 28
  store i64 %t775, ptr %t776, align 8
  %t777 = bitcast double %t342 to i64
  %t778 = getelementptr inbounds i64, ptr %t718, i64 29
  store i64 %t777, ptr %t778, align 8
  %t779 = bitcast double %t353 to i64
  %t780 = getelementptr inbounds i64, ptr %t718, i64 30
  store i64 %t779, ptr %t780, align 8
  %t781 = bitcast double %t364 to i64
  %t782 = getelementptr inbounds i64, ptr %t718, i64 31
  store i64 %t781, ptr %t782, align 8
  %t783 = bitcast double %t375 to i64
  %t784 = getelementptr inbounds i64, ptr %t718, i64 32
  store i64 %t783, ptr %t784, align 8
  %t785 = bitcast double %t386 to i64
  %t786 = getelementptr inbounds i64, ptr %t718, i64 33
  store i64 %t785, ptr %t786, align 8
  %t787 = bitcast double %t397 to i64
  %t788 = getelementptr inbounds i64, ptr %t718, i64 34
  store i64 %t787, ptr %t788, align 8
  %t789 = bitcast double %t408 to i64
  %t790 = getelementptr inbounds i64, ptr %t718, i64 35
  store i64 %t789, ptr %t790, align 8
  %t791 = bitcast double %t419 to i64
  %t792 = getelementptr inbounds i64, ptr %t718, i64 36
  store i64 %t791, ptr %t792, align 8
  %t793 = bitcast double %t430 to i64
  %t794 = getelementptr inbounds i64, ptr %t718, i64 37
  store i64 %t793, ptr %t794, align 8
  %t795 = bitcast double %t441 to i64
  %t796 = getelementptr inbounds i64, ptr %t718, i64 38
  store i64 %t795, ptr %t796, align 8
  %t797 = bitcast double %t452 to i64
  %t798 = getelementptr inbounds i64, ptr %t718, i64 39
  store i64 %t797, ptr %t798, align 8
  %t799 = bitcast double %t463 to i64
  %t800 = getelementptr inbounds i64, ptr %t718, i64 40
  store i64 %t799, ptr %t800, align 8
  %t801 = bitcast double %t474 to i64
  %t802 = getelementptr inbounds i64, ptr %t718, i64 41
  store i64 %t801, ptr %t802, align 8
  %t803 = bitcast double %t485 to i64
  %t804 = getelementptr inbounds i64, ptr %t718, i64 42
  store i64 %t803, ptr %t804, align 8
  %t805 = bitcast double %t496 to i64
  %t806 = getelementptr inbounds i64, ptr %t718, i64 43
  store i64 %t805, ptr %t806, align 8
  %t807 = bitcast double %t507 to i64
  %t808 = getelementptr inbounds i64, ptr %t718, i64 44
  store i64 %t807, ptr %t808, align 8
  %t809 = bitcast double %t518 to i64
  %t810 = getelementptr inbounds i64, ptr %t718, i64 45
  store i64 %t809, ptr %t810, align 8
  %t811 = bitcast double %t529 to i64
  %t812 = getelementptr inbounds i64, ptr %t718, i64 46
  store i64 %t811, ptr %t812, align 8
  %t813 = bitcast double %t540 to i64
  %t814 = getelementptr inbounds i64, ptr %t718, i64 47
  store i64 %t813, ptr %t814, align 8
  %t815 = bitcast double %t551 to i64
  %t816 = getelementptr inbounds i64, ptr %t718, i64 48
  store i64 %t815, ptr %t816, align 8
  %t817 = bitcast double %t562 to i64
  %t818 = getelementptr inbounds i64, ptr %t718, i64 49
  store i64 %t817, ptr %t818, align 8
  %t819 = bitcast double %t573 to i64
  %t820 = getelementptr inbounds i64, ptr %t718, i64 50
  store i64 %t819, ptr %t820, align 8
  %t821 = bitcast double %t584 to i64
  %t822 = getelementptr inbounds i64, ptr %t718, i64 51
  store i64 %t821, ptr %t822, align 8
  %t823 = bitcast double %t595 to i64
  %t824 = getelementptr inbounds i64, ptr %t718, i64 52
  store i64 %t823, ptr %t824, align 8
  %t825 = bitcast double %t606 to i64
  %t826 = getelementptr inbounds i64, ptr %t718, i64 53
  store i64 %t825, ptr %t826, align 8
  %t827 = bitcast double %t617 to i64
  %t828 = getelementptr inbounds i64, ptr %t718, i64 54
  store i64 %t827, ptr %t828, align 8
  %t829 = bitcast double %t628 to i64
  %t830 = getelementptr inbounds i64, ptr %t718, i64 55
  store i64 %t829, ptr %t830, align 8
  %t831 = bitcast double %t639 to i64
  %t832 = getelementptr inbounds i64, ptr %t718, i64 56
  store i64 %t831, ptr %t832, align 8
  %t833 = bitcast double %t650 to i64
  %t834 = getelementptr inbounds i64, ptr %t718, i64 57
  store i64 %t833, ptr %t834, align 8
  %t835 = bitcast double %t661 to i64
  %t836 = getelementptr inbounds i64, ptr %t718, i64 58
  store i64 %t835, ptr %t836, align 8
  %t837 = bitcast double %t672 to i64
  %t838 = getelementptr inbounds i64, ptr %t718, i64 59
  store i64 %t837, ptr %t838, align 8
  %t839 = bitcast double %t683 to i64
  %t840 = getelementptr inbounds i64, ptr %t718, i64 60
  store i64 %t839, ptr %t840, align 8
  %t841 = bitcast double %t694 to i64
  %t842 = getelementptr inbounds i64, ptr %t718, i64 61
  store i64 %t841, ptr %t842, align 8
  %t843 = bitcast double %t705 to i64
  %t844 = getelementptr inbounds i64, ptr %t718, i64 62
  store i64 %t843, ptr %t844, align 8
  %t845 = bitcast double %t716 to i64
  %t846 = getelementptr inbounds i64, ptr %t718, i64 63
  store i64 %t845, ptr %t846, align 8
  %t847 = getelementptr inbounds double, ptr %slots, i64 4
  %t848 = load double, ptr %t847, align 8
  %t849 = fmul double %t848, 0x3ff6666666666666
  %t850 = getelementptr inbounds double, ptr %slots, i64 4
  %t851 = load double, ptr %t850, align 8
  %t852 = fmul double %t851, 0x3ffccccccccccccd
  %t853 = getelementptr inbounds double, ptr %slots, i64 4
  %t854 = load double, ptr %t853, align 8
  %t855 = fmul double %t854, 0x400199999999999a
  %t856 = getelementptr inbounds double, ptr %slots, i64 4
  %t857 = load double, ptr %t856, align 8
  %t858 = fmul double %t857, 0x4004cccccccccccd
  %t859 = getelementptr inbounds double, ptr %slots, i64 4
  %t860 = load double, ptr %t859, align 8
  %t861 = fmul double %t860, 0x4008000000000000
  %t862 = getelementptr inbounds double, ptr %slots, i64 4
  %t863 = load double, ptr %t862, align 8
  %t864 = fmul double %t863, 0x400b333333333333
  %t865 = getelementptr inbounds double, ptr %slots, i64 4
  %t866 = load double, ptr %t865, align 8
  %t867 = fmul double %t866, 0x400e666666666666
  %t868 = getelementptr inbounds double, ptr %slots, i64 4
  %t869 = load double, ptr %t868, align 8
  %t870 = fmul double %t869, 0x4010cccccccccccd
  %t871 = getelementptr inbounds double, ptr %slots, i64 4
  %t872 = load double, ptr %t871, align 8
  %t873 = fmul double %t872, 0x4012666666666666
  %t874 = getelementptr inbounds double, ptr %slots, i64 4
  %t875 = load double, ptr %t874, align 8
  %t876 = fmul double %t875, 0x4014000000000000
  %t877 = getelementptr inbounds double, ptr %slots, i64 4
  %t878 = load double, ptr %t877, align 8
  %t879 = fmul double %t878, 0x401599999999999a
  %t880 = getelementptr inbounds double, ptr %slots, i64 4
  %t881 = load double, ptr %t880, align 8
  %t882 = fmul double %t881, 0x4017333333333333
  %t883 = getelementptr inbounds double, ptr %slots, i64 4
  %t884 = load double, ptr %t883, align 8
  %t885 = fmul double %t884, 0x4018cccccccccccd
  %t886 = getelementptr inbounds double, ptr %slots, i64 4
  %t887 = load double, ptr %t886, align 8
  %t888 = fmul double %t887, 0x401a666666666666
  %t889 = getelementptr inbounds double, ptr %slots, i64 4
  %t890 = load double, ptr %t889, align 8
  %t891 = fmul double %t890, 0x401c000000000000
  %t892 = getelementptr inbounds double, ptr %slots, i64 4
  %t893 = load double, ptr %t892, align 8
  %t894 = fmul double %t893, 0x401d99999999999a
  %t895 = getelementptr inbounds double, ptr %slots, i64 4
  %t896 = load double, ptr %t895, align 8
  %t897 = fmul double %t896, 0x401f333333333333
  %t898 = getelementptr inbounds double, ptr %slots, i64 4
  %t899 = load double, ptr %t898, align 8
  %t900 = fmul double %t899, 0x4020666666666666
  %t901 = getelementptr inbounds double, ptr %slots, i64 4
  %t902 = load double, ptr %t901, align 8
  %t903 = fmul double %t902, 0x4021333333333333
  %t904 = getelementptr inbounds double, ptr %slots, i64 4
  %t905 = load double, ptr %t904, align 8
  %t906 = fmul double %t905, 0x4022000000000000
  %t907 = getelementptr inbounds double, ptr %slots, i64 4
  %t908 = load double, ptr %t907, align 8
  %t909 = fmul double %t908, 0x4022cccccccccccd
  %t910 = getelementptr inbounds double, ptr %slots, i64 4
  %t911 = load double, ptr %t910, align 8
  %t912 = fmul double %t911, 0x402399999999999a
  %t913 = getelementptr inbounds double, ptr %slots, i64 4
  %t914 = load double, ptr %t913, align 8
  %t915 = fmul double %t914, 0x4024666666666666
  %t916 = getelementptr inbounds double, ptr %slots, i64 4
  %t917 = load double, ptr %t916, align 8
  %t918 = fmul double %t917, 0x4025333333333333
  %t919 = getelementptr inbounds double, ptr %slots, i64 4
  %t920 = load double, ptr %t919, align 8
  %t921 = fmul double %t920, 0x4026000000000000
  %t922 = getelementptr inbounds double, ptr %slots, i64 4
  %t923 = load double, ptr %t922, align 8
  %t924 = fmul double %t923, 0x4026cccccccccccd
  %t925 = getelementptr inbounds double, ptr %slots, i64 4
  %t926 = load double, ptr %t925, align 8
  %t927 = fmul double %t926, 0x402799999999999a
  %t928 = getelementptr inbounds double, ptr %slots, i64 4
  %t929 = load double, ptr %t928, align 8
  %t930 = fmul double %t929, 0x4028666666666666
  %t931 = getelementptr inbounds double, ptr %slots, i64 4
  %t932 = load double, ptr %t931, align 8
  %t933 = fmul double %t932, 0x4029333333333333
  %t934 = getelementptr inbounds double, ptr %slots, i64 4
  %t935 = load double, ptr %t934, align 8
  %t936 = fmul double %t935, 0x402a000000000000
  %t937 = getelementptr inbounds double, ptr %slots, i64 4
  %t938 = load double, ptr %t937, align 8
  %t939 = fmul double %t938, 0x402acccccccccccd
  %t940 = getelementptr inbounds double, ptr %slots, i64 4
  %t941 = load double, ptr %t940, align 8
  %t942 = fmul double %t941, 0x402b99999999999a
  %t943 = getelementptr inbounds double, ptr %slots, i64 4
  %t944 = load double, ptr %t943, align 8
  %t945 = fmul double %t944, 0x402c666666666666
  %t946 = getelementptr inbounds double, ptr %slots, i64 4
  %t947 = load double, ptr %t946, align 8
  %t948 = fmul double %t947, 0x402d333333333333
  %t949 = getelementptr inbounds double, ptr %slots, i64 4
  %t950 = load double, ptr %t949, align 8
  %t951 = fmul double %t950, 0x402e000000000000
  %t952 = getelementptr inbounds double, ptr %slots, i64 4
  %t953 = load double, ptr %t952, align 8
  %t954 = fmul double %t953, 0x402ecccccccccccd
  %t955 = getelementptr inbounds double, ptr %slots, i64 4
  %t956 = load double, ptr %t955, align 8
  %t957 = fmul double %t956, 0x402f99999999999a
  %t958 = getelementptr inbounds double, ptr %slots, i64 4
  %t959 = load double, ptr %t958, align 8
  %t960 = fmul double %t959, 0x4030333333333333
  %t961 = getelementptr inbounds double, ptr %slots, i64 4
  %t962 = load double, ptr %t961, align 8
  %t963 = fmul double %t962, 0x403099999999999a
  %t964 = getelementptr inbounds double, ptr %slots, i64 4
  %t965 = load double, ptr %t964, align 8
  %t966 = fmul double %t965, 0x4031000000000000
  %t967 = getelementptr inbounds double, ptr %slots, i64 4
  %t968 = load double, ptr %t967, align 8
  %t969 = fmul double %t968, 0x4031666666666666
  %t970 = getelementptr inbounds double, ptr %slots, i64 4
  %t971 = load double, ptr %t970, align 8
  %t972 = fmul double %t971, 0x4031cccccccccccd
  %t973 = getelementptr inbounds double, ptr %slots, i64 4
  %t974 = load double, ptr %t973, align 8
  %t975 = fmul double %t974, 0x4032333333333333
  %t976 = getelementptr inbounds double, ptr %slots, i64 4
  %t977 = load double, ptr %t976, align 8
  %t978 = fmul double %t977, 0x403299999999999a
  %t979 = getelementptr inbounds double, ptr %slots, i64 4
  %t980 = load double, ptr %t979, align 8
  %t981 = fmul double %t980, 0x4033000000000000
  %t982 = getelementptr inbounds double, ptr %slots, i64 4
  %t983 = load double, ptr %t982, align 8
  %t984 = fmul double %t983, 0x4033666666666666
  %t985 = getelementptr inbounds double, ptr %slots, i64 4
  %t986 = load double, ptr %t985, align 8
  %t987 = fmul double %t986, 0x4033cccccccccccd
  %t988 = getelementptr inbounds double, ptr %slots, i64 4
  %t989 = load double, ptr %t988, align 8
  %t990 = fmul double %t989, 0x4034333333333333
  %t991 = getelementptr inbounds double, ptr %slots, i64 4
  %t992 = load double, ptr %t991, align 8
  %t993 = fmul double %t992, 0x403499999999999a
  %t994 = getelementptr inbounds double, ptr %slots, i64 4
  %t995 = load double, ptr %t994, align 8
  %t996 = fmul double %t995, 0x4035000000000000
  %t997 = getelementptr inbounds double, ptr %slots, i64 4
  %t998 = load double, ptr %t997, align 8
  %t999 = fmul double %t998, 0x4035666666666666
  %t1000 = getelementptr inbounds double, ptr %slots, i64 4
  %t1001 = load double, ptr %t1000, align 8
  %t1002 = fmul double %t1001, 0x4035cccccccccccd
  %t1003 = getelementptr inbounds double, ptr %slots, i64 4
  %t1004 = load double, ptr %t1003, align 8
  %t1005 = fmul double %t1004, 0x4036333333333333
  %t1006 = getelementptr inbounds double, ptr %slots, i64 4
  %t1007 = load double, ptr %t1006, align 8
  %t1008 = fmul double %t1007, 0x403699999999999a
  %t1009 = getelementptr inbounds double, ptr %slots, i64 4
  %t1010 = load double, ptr %t1009, align 8
  %t1011 = fmul double %t1010, 0x4037000000000000
  %t1012 = getelementptr inbounds double, ptr %slots, i64 4
  %t1013 = load double, ptr %t1012, align 8
  %t1014 = fmul double %t1013, 0x4037666666666666
  %t1015 = getelementptr inbounds double, ptr %slots, i64 4
  %t1016 = load double, ptr %t1015, align 8
  %t1017 = fmul double %t1016, 0x4037cccccccccccd
  %t1018 = getelementptr inbounds double, ptr %slots, i64 4
  %t1019 = load double, ptr %t1018, align 8
  %t1020 = fmul double %t1019, 0x4038333333333333
  %t1021 = getelementptr inbounds double, ptr %slots, i64 4
  %t1022 = load double, ptr %t1021, align 8
  %t1023 = fmul double %t1022, 0x403899999999999a
  %t1024 = getelementptr inbounds double, ptr %slots, i64 4
  %t1025 = load double, ptr %t1024, align 8
  %t1026 = fmul double %t1025, 0x4039000000000000
  %t1027 = getelementptr inbounds double, ptr %slots, i64 4
  %t1028 = load double, ptr %t1027, align 8
  %t1029 = fmul double %t1028, 0x4039666666666666
  %t1030 = getelementptr inbounds double, ptr %slots, i64 4
  %t1031 = load double, ptr %t1030, align 8
  %t1032 = fmul double %t1031, 0x4039cccccccccccd
  %t1033 = getelementptr inbounds double, ptr %slots, i64 4
  %t1034 = load double, ptr %t1033, align 8
  %t1035 = fmul double %t1034, 0x403a333333333333
  %t1036 = getelementptr inbounds double, ptr %slots, i64 4
  %t1037 = load double, ptr %t1036, align 8
  %t1038 = fmul double %t1037, 0x403a99999999999a
  %t1039 = getelementptr inbounds ptr, ptr %arrays, i64 1
  %t1040 = load ptr, ptr %t1039, align 8
  %t1041 = bitcast double %t849 to i64
  %t1042 = getelementptr inbounds i64, ptr %t1040, i64 0
  store i64 %t1041, ptr %t1042, align 8
  %t1043 = bitcast double %t852 to i64
  %t1044 = getelementptr inbounds i64, ptr %t1040, i64 1
  store i64 %t1043, ptr %t1044, align 8
  %t1045 = bitcast double %t855 to i64
  %t1046 = getelementptr inbounds i64, ptr %t1040, i64 2
  store i64 %t1045, ptr %t1046, align 8
  %t1047 = bitcast double %t858 to i64
  %t1048 = getelementptr inbounds i64, ptr %t1040, i64 3
  store i64 %t1047, ptr %t1048, align 8
  %t1049 = bitcast double %t861 to i64
  %t1050 = getelementptr inbounds i64, ptr %t1040, i64 4
  store i64 %t1049, ptr %t1050, align 8
  %t1051 = bitcast double %t864 to i64
  %t1052 = getelementptr inbounds i64, ptr %t1040, i64 5
  store i64 %t1051, ptr %t1052, align 8
  %t1053 = bitcast double %t867 to i64
  %t1054 = getelementptr inbounds i64, ptr %t1040, i64 6
  store i64 %t1053, ptr %t1054, align 8
  %t1055 = bitcast double %t870 to i64
  %t1056 = getelementptr inbounds i64, ptr %t1040, i64 7
  store i64 %t1055, ptr %t1056, align 8
  %t1057 = bitcast double %t873 to i64
  %t1058 = getelementptr inbounds i64, ptr %t1040, i64 8
  store i64 %t1057, ptr %t1058, align 8
  %t1059 = bitcast double %t876 to i64
  %t1060 = getelementptr inbounds i64, ptr %t1040, i64 9
  store i64 %t1059, ptr %t1060, align 8
  %t1061 = bitcast double %t879 to i64
  %t1062 = getelementptr inbounds i64, ptr %t1040, i64 10
  store i64 %t1061, ptr %t1062, align 8
  %t1063 = bitcast double %t882 to i64
  %t1064 = getelementptr inbounds i64, ptr %t1040, i64 11
  store i64 %t1063, ptr %t1064, align 8
  %t1065 = bitcast double %t885 to i64
  %t1066 = getelementptr inbounds i64, ptr %t1040, i64 12
  store i64 %t1065, ptr %t1066, align 8
  %t1067 = bitcast double %t888 to i64
  %t1068 = getelementptr inbounds i64, ptr %t1040, i64 13
  store i64 %t1067, ptr %t1068, align 8
  %t1069 = bitcast double %t891 to i64
  %t1070 = getelementptr inbounds i64, ptr %t1040, i64 14
  store i64 %t1069, ptr %t1070, align 8
  %t1071 = bitcast double %t894 to i64
  %t1072 = getelementptr inbounds i64, ptr %t1040, i64 15
  store i64 %t1071, ptr %t1072, align 8
  %t1073 = bitcast double %t897 to i64
  %t1074 = getelementptr inbounds i64, ptr %t1040, i64 16
  store i64 %t1073, ptr %t1074, align 8
  %t1075 = bitcast double %t900 to i64
  %t1076 = getelementptr inbounds i64, ptr %t1040, i64 17
  store i64 %t1075, ptr %t1076, align 8
  %t1077 = bitcast double %t903 to i64
  %t1078 = getelementptr inbounds i64, ptr %t1040, i64 18
  store i64 %t1077, ptr %t1078, align 8
  %t1079 = bitcast double %t906 to i64
  %t1080 = getelementptr inbounds i64, ptr %t1040, i64 19
  store i64 %t1079, ptr %t1080, align 8
  %t1081 = bitcast double %t909 to i64
  %t1082 = getelementptr inbounds i64, ptr %t1040, i64 20
  store i64 %t1081, ptr %t1082, align 8
  %t1083 = bitcast double %t912 to i64
  %t1084 = getelementptr inbounds i64, ptr %t1040, i64 21
  store i64 %t1083, ptr %t1084, align 8
  %t1085 = bitcast double %t915 to i64
  %t1086 = getelementptr inbounds i64, ptr %t1040, i64 22
  store i64 %t1085, ptr %t1086, align 8
  %t1087 = bitcast double %t918 to i64
  %t1088 = getelementptr inbounds i64, ptr %t1040, i64 23
  store i64 %t1087, ptr %t1088, align 8
  %t1089 = bitcast double %t921 to i64
  %t1090 = getelementptr inbounds i64, ptr %t1040, i64 24
  store i64 %t1089, ptr %t1090, align 8
  %t1091 = bitcast double %t924 to i64
  %t1092 = getelementptr inbounds i64, ptr %t1040, i64 25
  store i64 %t1091, ptr %t1092, align 8
  %t1093 = bitcast double %t927 to i64
  %t1094 = getelementptr inbounds i64, ptr %t1040, i64 26
  store i64 %t1093, ptr %t1094, align 8
  %t1095 = bitcast double %t930 to i64
  %t1096 = getelementptr inbounds i64, ptr %t1040, i64 27
  store i64 %t1095, ptr %t1096, align 8
  %t1097 = bitcast double %t933 to i64
  %t1098 = getelementptr inbounds i64, ptr %t1040, i64 28
  store i64 %t1097, ptr %t1098, align 8
  %t1099 = bitcast double %t936 to i64
  %t1100 = getelementptr inbounds i64, ptr %t1040, i64 29
  store i64 %t1099, ptr %t1100, align 8
  %t1101 = bitcast double %t939 to i64
  %t1102 = getelementptr inbounds i64, ptr %t1040, i64 30
  store i64 %t1101, ptr %t1102, align 8
  %t1103 = bitcast double %t942 to i64
  %t1104 = getelementptr inbounds i64, ptr %t1040, i64 31
  store i64 %t1103, ptr %t1104, align 8
  %t1105 = bitcast double %t945 to i64
  %t1106 = getelementptr inbounds i64, ptr %t1040, i64 32
  store i64 %t1105, ptr %t1106, align 8
  %t1107 = bitcast double %t948 to i64
  %t1108 = getelementptr inbounds i64, ptr %t1040, i64 33
  store i64 %t1107, ptr %t1108, align 8
  %t1109 = bitcast double %t951 to i64
  %t1110 = getelementptr inbounds i64, ptr %t1040, i64 34
  store i64 %t1109, ptr %t1110, align 8
  %t1111 = bitcast double %t954 to i64
  %t1112 = getelementptr inbounds i64, ptr %t1040, i64 35
  store i64 %t1111, ptr %t1112, align 8
  %t1113 = bitcast double %t957 to i64
  %t1114 = getelementptr inbounds i64, ptr %t1040, i64 36
  store i64 %t1113, ptr %t1114, align 8
  %t1115 = bitcast double %t960 to i64
  %t1116 = getelementptr inbounds i64, ptr %t1040, i64 37
  store i64 %t1115, ptr %t1116, align 8
  %t1117 = bitcast double %t963 to i64
  %t1118 = getelementptr inbounds i64, ptr %t1040, i64 38
  store i64 %t1117, ptr %t1118, align 8
  %t1119 = bitcast double %t966 to i64
  %t1120 = getelementptr inbounds i64, ptr %t1040, i64 39
  store i64 %t1119, ptr %t1120, align 8
  %t1121 = bitcast double %t969 to i64
  %t1122 = getelementptr inbounds i64, ptr %t1040, i64 40
  store i64 %t1121, ptr %t1122, align 8
  %t1123 = bitcast double %t972 to i64
  %t1124 = getelementptr inbounds i64, ptr %t1040, i64 41
  store i64 %t1123, ptr %t1124, align 8
  %t1125 = bitcast double %t975 to i64
  %t1126 = getelementptr inbounds i64, ptr %t1040, i64 42
  store i64 %t1125, ptr %t1126, align 8
  %t1127 = bitcast double %t978 to i64
  %t1128 = getelementptr inbounds i64, ptr %t1040, i64 43
  store i64 %t1127, ptr %t1128, align 8
  %t1129 = bitcast double %t981 to i64
  %t1130 = getelementptr inbounds i64, ptr %t1040, i64 44
  store i64 %t1129, ptr %t1130, align 8
  %t1131 = bitcast double %t984 to i64
  %t1132 = getelementptr inbounds i64, ptr %t1040, i64 45
  store i64 %t1131, ptr %t1132, align 8
  %t1133 = bitcast double %t987 to i64
  %t1134 = getelementptr inbounds i64, ptr %t1040, i64 46
  store i64 %t1133, ptr %t1134, align 8
  %t1135 = bitcast double %t990 to i64
  %t1136 = getelementptr inbounds i64, ptr %t1040, i64 47
  store i64 %t1135, ptr %t1136, align 8
  %t1137 = bitcast double %t993 to i64
  %t1138 = getelementptr inbounds i64, ptr %t1040, i64 48
  store i64 %t1137, ptr %t1138, align 8
  %t1139 = bitcast double %t996 to i64
  %t1140 = getelementptr inbounds i64, ptr %t1040, i64 49
  store i64 %t1139, ptr %t1140, align 8
  %t1141 = bitcast double %t999 to i64
  %t1142 = getelementptr inbounds i64, ptr %t1040, i64 50
  store i64 %t1141, ptr %t1142, align 8
  %t1143 = bitcast double %t1002 to i64
  %t1144 = getelementptr inbounds i64, ptr %t1040, i64 51
  store i64 %t1143, ptr %t1144, align 8
  %t1145 = bitcast double %t1005 to i64
  %t1146 = getelementptr inbounds i64, ptr %t1040, i64 52
  store i64 %t1145, ptr %t1146, align 8
  %t1147 = bitcast double %t1008 to i64
  %t1148 = getelementptr inbounds i64, ptr %t1040, i64 53
  store i64 %t1147, ptr %t1148, align 8
  %t1149 = bitcast double %t1011 to i64
  %t1150 = getelementptr inbounds i64, ptr %t1040, i64 54
  store i64 %t1149, ptr %t1150, align 8
  %t1151 = bitcast double %t1014 to i64
  %t1152 = getelementptr inbounds i64, ptr %t1040, i64 55
  store i64 %t1151, ptr %t1152, align 8
  %t1153 = bitcast double %t1017 to i64
  %t1154 = getelementptr inbounds i64, ptr %t1040, i64 56
  store i64 %t1153, ptr %t1154, align 8
  %t1155 = bitcast double %t1020 to i64
  %t1156 = getelementptr inbounds i64, ptr %t1040, i64 57
  store i64 %t1155, ptr %t1156, align 8
  %t1157 = bitcast double %t1023 to i64
  %t1158 = getelementptr inbounds i64, ptr %t1040, i64 58
  store i64 %t1157, ptr %t1158, align 8
  %t1159 = bitcast double %t1026 to i64
  %t1160 = getelementptr inbounds i64, ptr %t1040, i64 59
  store i64 %t1159, ptr %t1160, align 8
  %t1161 = bitcast double %t1029 to i64
  %t1162 = getelementptr inbounds i64, ptr %t1040, i64 60
  store i64 %t1161, ptr %t1162, align 8
  %t1163 = bitcast double %t1032 to i64
  %t1164 = getelementptr inbounds i64, ptr %t1040, i64 61
  store i64 %t1163, ptr %t1164, align 8
  %t1165 = bitcast double %t1035 to i64
  %t1166 = getelementptr inbounds i64, ptr %t1040, i64 62
  store i64 %t1165, ptr %t1166, align 8
  %t1167 = bitcast double %t1038 to i64
  %t1168 = getelementptr inbounds i64, ptr %t1040, i64 63
  store i64 %t1167, ptr %t1168, align 8
  %t1169 = fptosi double %t3 to i64
  %t1170 = sitofp i64 %t1169 to double
  %t1171 = getelementptr inbounds double, ptr %slots, i64 8
  store double %t1170, ptr %t1171, align 8
  %t1172 = fptosi double %t9 to i64
  %t1173 = sitofp i64 %t1172 to double
  %t1174 = getelementptr inbounds double, ptr %slots, i64 9
  store double %t1173, ptr %t1174, align 8
  %t1175 = getelementptr inbounds double, ptr %output_buffer, i64 %s
  store double 0x0000000000000000, ptr %t1175, align 8
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
