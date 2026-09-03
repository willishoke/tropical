define void @tropical_kernel(ptr %inputs, ptr %registers, ptr %arrays, ptr %array_sizes, ptr %temps, double %sampleRate, i64 %start_sample_index, ptr %param_ptrs, ptr %output_buffer, i64 %buffer_length, ptr noalias nocapture %slots) {
entry:
  br label %loop_cond

loop_cond:
  %s = phi i64 [ 0, %entry ], [ %s_next, %rd_end_82 ]
  %current_idx = add i64 %start_sample_index, %s
  %loopcond = icmp ult i64 %s, %buffer_length
  br i1 %loopcond, label %loop_body, label %loop_end

loop_body:
  %t0 = getelementptr inbounds double, ptr %slots, i64 7
  %t1 = load double, ptr %t0, align 8
  %t2 = fptosi double %t1 to i64
  %t3 = mul i64 %t2, %current_idx
  %t4 = getelementptr inbounds double, ptr %slots, i64 6
  %t5 = load double, ptr %t4, align 8
  %t6 = fptosi double %t5 to i64
  %t7 = add i64 %t6, %t3
  %t8 = fmul double 0x0000000000000000, 0x41f0000000000000
  %t9 = fptosi double %t8 to i64
  %t10 = sub i64 %t7, %t9
  %t11 = icmp sgt i64 %t10, 0
  %t12 = getelementptr inbounds ptr, ptr %arrays, i64 2
  %t13 = load ptr, ptr %t12, align 8
  %t14 = bitcast double 0x3ff0000000000000 to i64
  %t15 = getelementptr inbounds i64, ptr %t13, i64 0
  store i64 %t14, ptr %t15, align 8
  %t16 = bitcast double 0x3fdddb6801178ea9 to i64
  %t17 = getelementptr inbounds i64, ptr %t13, i64 1
  store i64 %t16, ptr %t17, align 8
  %t18 = bitcast double 0x3fd31d20b7a1f464 to i64
  %t19 = getelementptr inbounds i64, ptr %t13, i64 2
  store i64 %t18, ptr %t19, align 8
  %t20 = bitcast double 0x3fcbdb8cdadbdcc2 to i64
  %t21 = getelementptr inbounds i64, ptr %t13, i64 3
  store i64 %t20, ptr %t21, align 8
  %t22 = bitcast double 0x3fc5cb57608010de to i64
  %t23 = getelementptr inbounds i64, ptr %t13, i64 4
  store i64 %t22, ptr %t23, align 8
  %t24 = bitcast double 0x3fc1d5731da4df14 to i64
  %t25 = getelementptr inbounds i64, ptr %t13, i64 5
  store i64 %t24, ptr %t25, align 8
  %t26 = bitcast double 0x3fbe1ac3b46acb16 to i64
  %t27 = getelementptr inbounds i64, ptr %t13, i64 6
  store i64 %t26, ptr %t27, align 8
  %t28 = bitcast double 0x3fb9fdf8bcced7b1 to i64
  %t29 = getelementptr inbounds i64, ptr %t13, i64 7
  store i64 %t28, ptr %t29, align 8
  %t30 = bitcast double 0x3fb6d562bafce330 to i64
  %t31 = getelementptr inbounds i64, ptr %t13, i64 8
  store i64 %t30, ptr %t31, align 8
  %t32 = bitcast double 0x3fb455b5a30a8ad9 to i64
  %t33 = getelementptr inbounds i64, ptr %t13, i64 9
  store i64 %t32, ptr %t33, align 8
  %t34 = bitcast double 0x3fb24f9280a1310d to i64
  %t35 = getelementptr inbounds i64, ptr %t13, i64 10
  store i64 %t34, ptr %t35, align 8
  %t36 = bitcast double 0x3fb0a3b715029c87 to i64
  %t37 = getelementptr inbounds i64, ptr %t13, i64 11
  store i64 %t36, ptr %t37, align 8
  %t38 = bitcast double 0x3fae7964fcb3ebef to i64
  %t39 = getelementptr inbounds i64, ptr %t13, i64 12
  store i64 %t38, ptr %t39, align 8
  %t40 = bitcast double 0x3fac16aa5edf2079 to i64
  %t41 = getelementptr inbounds i64, ptr %t13, i64 13
  store i64 %t40, ptr %t41, align 8
  %t42 = bitcast double 0x3faa0924e1c8798c to i64
  %t43 = getelementptr inbounds i64, ptr %t13, i64 14
  store i64 %t42, ptr %t43, align 8
  %t44 = bitcast double 0x3fa8406003b1b111 to i64
  %t45 = getelementptr inbounds i64, ptr %t13, i64 15
  store i64 %t44, ptr %t45, align 8
  %t46 = getelementptr inbounds ptr, ptr %arrays, i64 3
  %t47 = load ptr, ptr %t46, align 8
  %t48 = bitcast double 0x0000000000000000 to i64
  %t49 = getelementptr inbounds i64, ptr %t47, i64 0
  store i64 %t48, ptr %t49, align 8
  %t50 = bitcast double 0x0000000000000000 to i64
  %t51 = getelementptr inbounds i64, ptr %t47, i64 1
  store i64 %t50, ptr %t51, align 8
  %t52 = bitcast double 0x0000000000000000 to i64
  %t53 = getelementptr inbounds i64, ptr %t47, i64 2
  store i64 %t52, ptr %t53, align 8
  %t54 = bitcast double 0x0000000000000000 to i64
  %t55 = getelementptr inbounds i64, ptr %t47, i64 3
  store i64 %t54, ptr %t55, align 8
  %t56 = bitcast double 0x0000000000000000 to i64
  %t57 = getelementptr inbounds i64, ptr %t47, i64 4
  store i64 %t56, ptr %t57, align 8
  %t58 = bitcast double 0x0000000000000000 to i64
  %t59 = getelementptr inbounds i64, ptr %t47, i64 5
  store i64 %t58, ptr %t59, align 8
  %t60 = bitcast double 0x0000000000000000 to i64
  %t61 = getelementptr inbounds i64, ptr %t47, i64 6
  store i64 %t60, ptr %t61, align 8
  %t62 = bitcast double 0x0000000000000000 to i64
  %t63 = getelementptr inbounds i64, ptr %t47, i64 7
  store i64 %t62, ptr %t63, align 8
  %t64 = bitcast double 0x0000000000000000 to i64
  %t65 = getelementptr inbounds i64, ptr %t47, i64 8
  store i64 %t64, ptr %t65, align 8
  %t66 = bitcast double 0x0000000000000000 to i64
  %t67 = getelementptr inbounds i64, ptr %t47, i64 9
  store i64 %t66, ptr %t67, align 8
  %t68 = bitcast double 0x0000000000000000 to i64
  %t69 = getelementptr inbounds i64, ptr %t47, i64 10
  store i64 %t68, ptr %t69, align 8
  %t70 = bitcast double 0x0000000000000000 to i64
  %t71 = getelementptr inbounds i64, ptr %t47, i64 11
  store i64 %t70, ptr %t71, align 8
  %t72 = bitcast double 0x0000000000000000 to i64
  %t73 = getelementptr inbounds i64, ptr %t47, i64 12
  store i64 %t72, ptr %t73, align 8
  %t74 = bitcast double 0x0000000000000000 to i64
  %t75 = getelementptr inbounds i64, ptr %t47, i64 13
  store i64 %t74, ptr %t75, align 8
  %t76 = bitcast double 0x0000000000000000 to i64
  %t77 = getelementptr inbounds i64, ptr %t47, i64 14
  store i64 %t76, ptr %t77, align 8
  %t78 = bitcast double 0x0000000000000000 to i64
  %t79 = getelementptr inbounds i64, ptr %t47, i64 15
  store i64 %t78, ptr %t79, align 8
  %t80 = alloca i64, align 8
  store i64 0, ptr %t80, align 8
  %t81 = alloca i64, align 8
  store i64 0, ptr %t81, align 8
  br label %rd_cond_82
rd_cond_82:
  %t83 = load i64, ptr %t81, align 8
  %t84 = icmp ult i64 %t83, 16
  br i1 %t84, label %rd_body_82, label %rd_end_82
rd_body_82:
  %t85 = getelementptr inbounds i64, ptr %array_sizes, i64 1
  %t86 = load i64, ptr %t85, align 8
  %t87 = getelementptr inbounds ptr, ptr %arrays, i64 1
  %t88 = load ptr, ptr %t87, align 8
  %t89 = load i64, ptr %t81, align 8
  %t90 = icmp slt i64 %t89, 0
  %t91 = xor i1 %t90, true
  %t92 = icmp ult i64 %t89, %t86
  %t93 = and i1 %t91, %t92
  %t94 = getelementptr inbounds i64, ptr %t88, i64 %t89
  %t95 = load i64, ptr %t94, align 8
  %t96 = bitcast i64 %t95 to double
  %t97 = select i1 %t93, double %t96, double 0x0000000000000000
  %t98 = sitofp i64 %t10 to double
  %t99 = fdiv double %t98, 0x41f0000000000000
  %t100 = fcmp oeq double 0x41f0000000000000, 0x0000000000000000
  %t101 = select i1 %t100, double 0x0000000000000000, double %t99
  %t102 = fdiv double %t101, %sampleRate
  %t103 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t104 = select i1 %t103, double 0x0000000000000000, double %t102
  %t105 = fmul double %t97, %t104
  %t106 = fneg double %t105
  %t107 = fcmp ogt double %t106, 0xc055c00000000000
  %t108 = select i1 %t107, double %t106, double 0xc055c00000000000
  %t109 = fcmp olt double %t108, 0x4056000000000000
  %t110 = select i1 %t109, double %t108, double 0x4056000000000000
  %t111 = fmul double %t110, 0x3ff71547652b82fe
  %t112 = call double @llvm.round.f64(double %t111)
  %t113 = fmul double %t112, 0x3fe62e4000000000
  %t114 = fsub double %t110, %t113
  %t115 = fmul double %t112, 0x3eb7f7d1cf79abca
  %t116 = fsub double %t114, %t115
  %t117 = fmul double 0x3f2a0d2ce64969e6, %t116
  %t118 = fadd double 0x3f56e879c3f007dd, %t117
  %t119 = fmul double %t118, %t116
  %t120 = fadd double 0x3f811120fb3cb51d, %t119
  %t121 = fmul double %t120, %t116
  %t122 = fadd double 0x3fa555381d73fd31, %t121
  %t123 = fmul double %t122, %t116
  %t124 = fadd double 0x3fc555553b661d99, %t123
  %t125 = fmul double %t124, %t116
  %t126 = fadd double 0x3fe000000672a44f, %t125
  %t127 = fmul double %t116, %t126
  %t128 = fadd double 0x3ff0000000000000, %t127
  %t129 = fmul double %t116, %t128
  %t130 = fadd double 0x3ff0000000000000, %t129
  %t131 = fptosi double %t112 to i64
  %t132 = add i64 %t131, 1023
  %t133 = shl i64 %t132, 52
  %t134 = bitcast i64 %t133 to double
  %t135 = fmul double %t130, %t134
  %t136 = getelementptr inbounds i64, ptr %array_sizes, i64 2
  %t137 = load i64, ptr %t136, align 8
  %t138 = getelementptr inbounds ptr, ptr %arrays, i64 2
  %t139 = load ptr, ptr %t138, align 8
  %t140 = load i64, ptr %t81, align 8
  %t141 = icmp slt i64 %t140, 0
  %t142 = xor i1 %t141, true
  %t143 = icmp ult i64 %t140, %t137
  %t144 = and i1 %t142, %t143
  %t145 = getelementptr inbounds i64, ptr %t139, i64 %t140
  %t146 = load i64, ptr %t145, align 8
  %t147 = bitcast i64 %t146 to double
  %t148 = select i1 %t144, double %t147, double 0x0000000000000000
  %t149 = fmul double %t135, %t148
  %t150 = fmul double %t149, 0x41b0000000000000
  %t151 = fptosi double %t150 to i64
  %t152 = fptosi double 0x3ff0000000000000 to i64
  %t153 = fptosi double 0x4000000000000000 to i64
  %t154 = getelementptr inbounds i64, ptr %array_sizes, i64 0
  %t155 = load i64, ptr %t154, align 8
  %t156 = getelementptr inbounds ptr, ptr %arrays, i64 0
  %t157 = load ptr, ptr %t156, align 8
  %t158 = load i64, ptr %t81, align 8
  %t159 = icmp slt i64 %t158, 0
  %t160 = xor i1 %t159, true
  %t161 = icmp ult i64 %t158, %t155
  %t162 = and i1 %t160, %t161
  %t163 = getelementptr inbounds i64, ptr %t157, i64 %t158
  %t164 = load i64, ptr %t163, align 8
  %t165 = bitcast i64 %t164 to double
  %t166 = select i1 %t162, double %t165, double 0x0000000000000000
  %t167 = fptosi double %t166 to i64
  %t168 = getelementptr inbounds double, ptr %slots, i64 9
  %t169 = load double, ptr %t168, align 8
  %t170 = fptosi double %t169 to i64
  %t171 = mul i64 %t170, %current_idx
  %t172 = getelementptr inbounds double, ptr %slots, i64 8
  %t173 = load double, ptr %t172, align 8
  %t174 = fptosi double %t173 to i64
  %t175 = add i64 %t174, %t171
  %t176 = fptosi double %t8 to i64
  %t177 = sub i64 %t175, %t176
  %t178 = ashr i64 %t177, 32
  %t179 = mul i64 %t167, %t178
  %t180 = and i64 %t177, 4294967295
  %t181 = mul i64 %t167, %t180
  %t182 = ashr i64 %t181, 32
  %t183 = add i64 %t179, %t182
  %t184 = and i64 %t183, 4294967295
  %t185 = add i64 %t184, 1073741824
  %t186 = and i64 %t185, 4294967295
  %t187 = add i64 %t186, 1073741824
  %t188 = ashr i64 %t187, 31
  %t189 = and i64 %t188, 1
  %t190 = mul i64 %t153, %t189
  %t191 = sub i64 %t152, %t190
  %t192 = shl i64 %t188, 31
  %t193 = sub i64 %t186, %t192
  %t194 = mul i64 %t193, %t193
  %t195 = ashr i64 %t194, 30
  %t196 = ashr i64 %t195, 30
  %t197 = sub i64 61, %t196
  %t198 = mul i64 %t197, %t195
  %t199 = ashr i64 %t198, 30
  %t200 = sub i64 3864, %t199
  %t201 = mul i64 %t200, %t195
  %t202 = ashr i64 %t201, 30
  %t203 = sub i64 172272, %t202
  %t204 = mul i64 %t203, %t195
  %t205 = ashr i64 %t204, 30
  %t206 = sub i64 5026995, %t205
  %t207 = mul i64 %t206, %t195
  %t208 = ashr i64 %t207, 30
  %t209 = sub i64 85569306, %t208
  %t210 = mul i64 %t209, %t195
  %t211 = ashr i64 %t210, 30
  %t212 = sub i64 693598668, %t211
  %t213 = mul i64 %t212, %t195
  %t214 = ashr i64 %t213, 30
  %t215 = sub i64 1686629713, %t214
  %t216 = mul i64 %t193, %t215
  %t217 = ashr i64 %t216, 30
  %t218 = mul i64 %t191, %t217
  %t219 = mul i64 %t151, %t218
  %t220 = getelementptr inbounds i64, ptr %array_sizes, i64 3
  %t221 = load i64, ptr %t220, align 8
  %t222 = getelementptr inbounds ptr, ptr %arrays, i64 3
  %t223 = load ptr, ptr %t222, align 8
  %t224 = load i64, ptr %t81, align 8
  %t225 = icmp slt i64 %t224, 0
  %t226 = xor i1 %t225, true
  %t227 = icmp ult i64 %t224, %t221
  %t228 = and i1 %t226, %t227
  %t229 = getelementptr inbounds i64, ptr %t223, i64 %t224
  %t230 = load i64, ptr %t229, align 8
  %t231 = bitcast i64 %t230 to double
  %t232 = select i1 %t228, double %t231, double 0x0000000000000000
  %t233 = fmul double %t135, %t232
  %t234 = fmul double %t233, 0x41b0000000000000
  %t235 = fptosi double %t234 to i64
  %t236 = ashr i64 %t185, 31
  %t237 = and i64 %t236, 1
  %t238 = mul i64 %t153, %t237
  %t239 = sub i64 %t152, %t238
  %t240 = shl i64 %t236, 31
  %t241 = sub i64 %t184, %t240
  %t242 = mul i64 %t241, %t241
  %t243 = ashr i64 %t242, 30
  %t244 = ashr i64 %t243, 30
  %t245 = sub i64 61, %t244
  %t246 = mul i64 %t245, %t243
  %t247 = ashr i64 %t246, 30
  %t248 = sub i64 3864, %t247
  %t249 = mul i64 %t248, %t243
  %t250 = ashr i64 %t249, 30
  %t251 = sub i64 172272, %t250
  %t252 = mul i64 %t251, %t243
  %t253 = ashr i64 %t252, 30
  %t254 = sub i64 5026995, %t253
  %t255 = mul i64 %t254, %t243
  %t256 = ashr i64 %t255, 30
  %t257 = sub i64 85569306, %t256
  %t258 = mul i64 %t257, %t243
  %t259 = ashr i64 %t258, 30
  %t260 = sub i64 693598668, %t259
  %t261 = mul i64 %t260, %t243
  %t262 = ashr i64 %t261, 30
  %t263 = sub i64 1686629713, %t262
  %t264 = mul i64 %t241, %t263
  %t265 = ashr i64 %t264, 30
  %t266 = mul i64 %t239, %t265
  %t267 = mul i64 %t235, %t266
  %t268 = sub i64 %t219, %t267
  %t269 = ashr i64 %t268, 28
  %t270 = load i64, ptr %t80, align 8
  %t271 = add i64 %t270, %t269
  store i64 %t271, ptr %t80, align 8
  %t272 = load i64, ptr %t81, align 8
  %t273 = add i64 %t272, 1
  store i64 %t273, ptr %t81, align 8
  br label %rd_cond_82
rd_end_82:
  %t274 = load i64, ptr %t80, align 8
  %t275 = sitofp i64 %t274 to double
  %t276 = fdiv double %t275, 0x41d0000000000000
  %t277 = fcmp oeq double 0x41d0000000000000, 0x0000000000000000
  %t278 = select i1 %t277, double 0x0000000000000000, double %t276
  %t279 = select i1 %t11, double %t278, double 0x0000000000000000
  %t280 = getelementptr inbounds double, ptr %slots, i64 2
  %t281 = load double, ptr %t280, align 8
  %t282 = fmul double %t279, %t281
  %t283 = fadd double %t282, 0x0000000000000000
  %t284 = getelementptr inbounds double, ptr %slots, i64 5
  store double %t283, ptr %t284, align 8
  %t285 = getelementptr inbounds double, ptr %slots, i64 5
  %t286 = load double, ptr %t285, align 8
  %t287 = fadd double 0x0000000000000000, %t286
  %t288 = fmul double %t287, 0x3ff0000000000000
  %t289 = getelementptr inbounds double, ptr %output_buffer, i64 %s
  store double %t288, ptr %t289, align 8
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
