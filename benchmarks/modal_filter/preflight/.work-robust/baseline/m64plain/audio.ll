define void @tropical_kernel(ptr %inputs, ptr %registers, ptr %arrays, ptr %array_sizes, ptr %temps, double %sampleRate, i64 %start_sample_index, ptr %param_ptrs, ptr %output_buffer, i64 %buffer_length, ptr noalias nocapture %slots) {
entry:
  br label %loop_cond

loop_cond:
  %s = phi i64 [ 0, %entry ], [ %s_next, %rd_end_274 ]
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
  %t46 = bitcast double 0x3fa6afdc11f1fe1a to i64
  %t47 = getelementptr inbounds i64, ptr %t13, i64 16
  store i64 %t46, ptr %t47, align 8
  %t48 = bitcast double 0x3fa54df00b6e4928 to i64
  %t49 = getelementptr inbounds i64, ptr %t13, i64 17
  store i64 %t48, ptr %t49, align 8
  %t50 = bitcast double 0x3fa413071c7f96a7 to i64
  %t51 = getelementptr inbounds i64, ptr %t13, i64 18
  store i64 %t50, ptr %t51, align 8
  %t52 = bitcast double 0x3fa2f919461514bd to i64
  %t53 = getelementptr inbounds i64, ptr %t13, i64 19
  store i64 %t52, ptr %t53, align 8
  %t54 = bitcast double 0x3fa1fb4b3bff0719 to i64
  %t55 = getelementptr inbounds i64, ptr %t13, i64 20
  store i64 %t54, ptr %t55, align 8
  %t56 = bitcast double 0x3fa115a8da6e43b2 to i64
  %t57 = getelementptr inbounds i64, ptr %t13, i64 21
  store i64 %t56, ptr %t57, align 8
  %t58 = bitcast double 0x3fa044f2021ca549 to i64
  %t59 = getelementptr inbounds i64, ptr %t13, i64 22
  store i64 %t58, ptr %t59, align 8
  %t60 = bitcast double 0x3f9f0ce8d94c51dd to i64
  %t61 = getelementptr inbounds i64, ptr %t13, i64 23
  store i64 %t60, ptr %t61, align 8
  %t62 = bitcast double 0x3f9dafdd985dcc6f to i64
  %t63 = getelementptr inbounds i64, ptr %t13, i64 24
  store i64 %t62, ptr %t63, align 8
  %t64 = bitcast double 0x3f9c6ef55bb60ad1 to i64
  %t65 = getelementptr inbounds i64, ptr %t13, i64 25
  store i64 %t64, ptr %t65, align 8
  %t66 = bitcast double 0x3f9b46f6b1d31759 to i64
  %t67 = getelementptr inbounds i64, ptr %t13, i64 26
  store i64 %t66, ptr %t67, align 8
  %t68 = bitcast double 0x3f9a3520ce911266 to i64
  %t69 = getelementptr inbounds i64, ptr %t13, i64 27
  store i64 %t68, ptr %t69, align 8
  %t70 = bitcast double 0x3f9937165d51810f to i64
  %t71 = getelementptr inbounds i64, ptr %t13, i64 28
  store i64 %t70, ptr %t71, align 8
  %t72 = bitcast double 0x3f984acc9fa0462e to i64
  %t73 = getelementptr inbounds i64, ptr %t13, i64 29
  store i64 %t72, ptr %t73, align 8
  %t74 = bitcast double 0x3f976e7ddcdbaae5 to i64
  %t75 = getelementptr inbounds i64, ptr %t13, i64 30
  store i64 %t74, ptr %t75, align 8
  %t76 = bitcast double 0x3f96a09e667ee22b to i64
  %t77 = getelementptr inbounds i64, ptr %t13, i64 31
  store i64 %t76, ptr %t77, align 8
  %t78 = bitcast double 0x3f95dfd3a3951da1 to i64
  %t79 = getelementptr inbounds i64, ptr %t13, i64 32
  store i64 %t78, ptr %t79, align 8
  %t80 = bitcast double 0x3f952aecb6aff4af to i64
  %t81 = getelementptr inbounds i64, ptr %t13, i64 33
  store i64 %t80, ptr %t81, align 8
  %t82 = bitcast double 0x3f9480dc6b73bf06 to i64
  %t83 = getelementptr inbounds i64, ptr %t13, i64 34
  store i64 %t82, ptr %t83, align 8
  %t84 = bitcast double 0x3f93e0b42b2e6ca6 to i64
  %t85 = getelementptr inbounds i64, ptr %t13, i64 35
  store i64 %t84, ptr %t85, align 8
  %t86 = bitcast double 0x3f93499fc603cb4e to i64
  %t87 = getelementptr inbounds i64, ptr %t13, i64 36
  store i64 %t86, ptr %t87, align 8
  %t88 = bitcast double 0x3f92bae1e9042281 to i64
  %t89 = getelementptr inbounds i64, ptr %t13, i64 37
  store i64 %t88, ptr %t89, align 8
  %t90 = bitcast double 0x3f9233d121d71caa to i64
  %t91 = getelementptr inbounds i64, ptr %t13, i64 38
  store i64 %t90, ptr %t91, align 8
  %t92 = bitcast double 0x3f91b3d556b576a0 to i64
  %t93 = getelementptr inbounds i64, ptr %t13, i64 39
  store i64 %t92, ptr %t93, align 8
  %t94 = bitcast double 0x3f913a659ecc8570 to i64
  %t95 = getelementptr inbounds i64, ptr %t13, i64 40
  store i64 %t94, ptr %t95, align 8
  %t96 = bitcast double 0x3f90c7066a963e53 to i64
  %t97 = getelementptr inbounds i64, ptr %t13, i64 41
  store i64 %t96, ptr %t97, align 8
  %t98 = bitcast double 0x3f905947eef40daa to i64
  %t99 = getelementptr inbounds i64, ptr %t13, i64 42
  store i64 %t98, ptr %t99, align 8
  %t100 = bitcast double 0x3f8fe1899108f8d2 to i64
  %t101 = getelementptr inbounds i64, ptr %t13, i64 43
  store i64 %t100, ptr %t101, align 8
  %t102 = bitcast double 0x3f8f1a419c0e2a94 to i64
  %t103 = getelementptr inbounds i64, ptr %t13, i64 44
  store i64 %t102, ptr %t103, align 8
  %t104 = bitcast double 0x3f8e5c10158f79b1 to i64
  %t105 = getelementptr inbounds i64, ptr %t13, i64 45
  store i64 %t104, ptr %t105, align 8
  %t106 = bitcast double 0x3f8da65bb5dec3f0 to i64
  %t107 = getelementptr inbounds i64, ptr %t13, i64 46
  store i64 %t106, ptr %t107, align 8
  %t108 = bitcast double 0x3f8cf898497d2ff1 to i64
  %t109 = getelementptr inbounds i64, ptr %t13, i64 47
  store i64 %t108, ptr %t109, align 8
  %t110 = bitcast double 0x3f8c524554f92973 to i64
  %t111 = getelementptr inbounds i64, ptr %t13, i64 48
  store i64 %t110, ptr %t111, align 8
  %t112 = bitcast double 0x3f8bb2ece332be9b to i64
  %t113 = getelementptr inbounds i64, ptr %t13, i64 49
  store i64 %t112, ptr %t113, align 8
  %t114 = bitcast double 0x3f8b1a2278496b9b to i64
  %t115 = getelementptr inbounds i64, ptr %t13, i64 50
  store i64 %t114, ptr %t115, align 8
  %t116 = bitcast double 0x3f8a878223dfefc6 to i64
  %t117 = getelementptr inbounds i64, ptr %t13, i64 51
  store i64 %t116, ptr %t117, align 8
  %t118 = bitcast double 0x3f89faafaeee9a21 to i64
  %t119 = getelementptr inbounds i64, ptr %t13, i64 52
  store i64 %t118, ptr %t119, align 8
  %t120 = bitcast double 0x3f897355e10d5129 to i64
  %t121 = getelementptr inbounds i64, ptr %t13, i64 53
  store i64 %t120, ptr %t121, align 8
  %t122 = bitcast double 0x3f88f125dab3388f to i64
  %t123 = getelementptr inbounds i64, ptr %t13, i64 54
  store i64 %t122, ptr %t123, align 8
  %t124 = bitcast double 0x3f8873d6814b8302 to i64
  %t125 = getelementptr inbounds i64, ptr %t13, i64 55
  store i64 %t124, ptr %t125, align 8
  %t126 = bitcast double 0x3f87fb23fb32db4a to i64
  %t127 = getelementptr inbounds i64, ptr %t13, i64 56
  store i64 %t126, ptr %t127, align 8
  %t128 = bitcast double 0x3f8786cf3984d4f0 to i64
  %t129 = getelementptr inbounds i64, ptr %t13, i64 57
  store i64 %t128, ptr %t129, align 8
  %t130 = bitcast double 0x3f87169d8e3f2903 to i64
  %t131 = getelementptr inbounds i64, ptr %t13, i64 58
  store i64 %t130, ptr %t131, align 8
  %t132 = bitcast double 0x3f86aa584cfa2534 to i64
  %t133 = getelementptr inbounds i64, ptr %t13, i64 59
  store i64 %t132, ptr %t133, align 8
  %t134 = bitcast double 0x3f8641cc7548ceb7 to i64
  %t135 = getelementptr inbounds i64, ptr %t13, i64 60
  store i64 %t134, ptr %t135, align 8
  %t136 = bitcast double 0x3f85dcca6569aca9 to i64
  %t137 = getelementptr inbounds i64, ptr %t13, i64 61
  store i64 %t136, ptr %t137, align 8
  %t138 = bitcast double 0x3f857b25948f8245 to i64
  %t139 = getelementptr inbounds i64, ptr %t13, i64 62
  store i64 %t138, ptr %t139, align 8
  %t140 = bitcast double 0x3f851cb453ba16cc to i64
  %t141 = getelementptr inbounds i64, ptr %t13, i64 63
  store i64 %t140, ptr %t141, align 8
  %t142 = getelementptr inbounds ptr, ptr %arrays, i64 3
  %t143 = load ptr, ptr %t142, align 8
  %t144 = bitcast double 0x0000000000000000 to i64
  %t145 = getelementptr inbounds i64, ptr %t143, i64 0
  store i64 %t144, ptr %t145, align 8
  %t146 = bitcast double 0x0000000000000000 to i64
  %t147 = getelementptr inbounds i64, ptr %t143, i64 1
  store i64 %t146, ptr %t147, align 8
  %t148 = bitcast double 0x0000000000000000 to i64
  %t149 = getelementptr inbounds i64, ptr %t143, i64 2
  store i64 %t148, ptr %t149, align 8
  %t150 = bitcast double 0x0000000000000000 to i64
  %t151 = getelementptr inbounds i64, ptr %t143, i64 3
  store i64 %t150, ptr %t151, align 8
  %t152 = bitcast double 0x0000000000000000 to i64
  %t153 = getelementptr inbounds i64, ptr %t143, i64 4
  store i64 %t152, ptr %t153, align 8
  %t154 = bitcast double 0x0000000000000000 to i64
  %t155 = getelementptr inbounds i64, ptr %t143, i64 5
  store i64 %t154, ptr %t155, align 8
  %t156 = bitcast double 0x0000000000000000 to i64
  %t157 = getelementptr inbounds i64, ptr %t143, i64 6
  store i64 %t156, ptr %t157, align 8
  %t158 = bitcast double 0x0000000000000000 to i64
  %t159 = getelementptr inbounds i64, ptr %t143, i64 7
  store i64 %t158, ptr %t159, align 8
  %t160 = bitcast double 0x0000000000000000 to i64
  %t161 = getelementptr inbounds i64, ptr %t143, i64 8
  store i64 %t160, ptr %t161, align 8
  %t162 = bitcast double 0x0000000000000000 to i64
  %t163 = getelementptr inbounds i64, ptr %t143, i64 9
  store i64 %t162, ptr %t163, align 8
  %t164 = bitcast double 0x0000000000000000 to i64
  %t165 = getelementptr inbounds i64, ptr %t143, i64 10
  store i64 %t164, ptr %t165, align 8
  %t166 = bitcast double 0x0000000000000000 to i64
  %t167 = getelementptr inbounds i64, ptr %t143, i64 11
  store i64 %t166, ptr %t167, align 8
  %t168 = bitcast double 0x0000000000000000 to i64
  %t169 = getelementptr inbounds i64, ptr %t143, i64 12
  store i64 %t168, ptr %t169, align 8
  %t170 = bitcast double 0x0000000000000000 to i64
  %t171 = getelementptr inbounds i64, ptr %t143, i64 13
  store i64 %t170, ptr %t171, align 8
  %t172 = bitcast double 0x0000000000000000 to i64
  %t173 = getelementptr inbounds i64, ptr %t143, i64 14
  store i64 %t172, ptr %t173, align 8
  %t174 = bitcast double 0x0000000000000000 to i64
  %t175 = getelementptr inbounds i64, ptr %t143, i64 15
  store i64 %t174, ptr %t175, align 8
  %t176 = bitcast double 0x0000000000000000 to i64
  %t177 = getelementptr inbounds i64, ptr %t143, i64 16
  store i64 %t176, ptr %t177, align 8
  %t178 = bitcast double 0x0000000000000000 to i64
  %t179 = getelementptr inbounds i64, ptr %t143, i64 17
  store i64 %t178, ptr %t179, align 8
  %t180 = bitcast double 0x0000000000000000 to i64
  %t181 = getelementptr inbounds i64, ptr %t143, i64 18
  store i64 %t180, ptr %t181, align 8
  %t182 = bitcast double 0x0000000000000000 to i64
  %t183 = getelementptr inbounds i64, ptr %t143, i64 19
  store i64 %t182, ptr %t183, align 8
  %t184 = bitcast double 0x0000000000000000 to i64
  %t185 = getelementptr inbounds i64, ptr %t143, i64 20
  store i64 %t184, ptr %t185, align 8
  %t186 = bitcast double 0x0000000000000000 to i64
  %t187 = getelementptr inbounds i64, ptr %t143, i64 21
  store i64 %t186, ptr %t187, align 8
  %t188 = bitcast double 0x0000000000000000 to i64
  %t189 = getelementptr inbounds i64, ptr %t143, i64 22
  store i64 %t188, ptr %t189, align 8
  %t190 = bitcast double 0x0000000000000000 to i64
  %t191 = getelementptr inbounds i64, ptr %t143, i64 23
  store i64 %t190, ptr %t191, align 8
  %t192 = bitcast double 0x0000000000000000 to i64
  %t193 = getelementptr inbounds i64, ptr %t143, i64 24
  store i64 %t192, ptr %t193, align 8
  %t194 = bitcast double 0x0000000000000000 to i64
  %t195 = getelementptr inbounds i64, ptr %t143, i64 25
  store i64 %t194, ptr %t195, align 8
  %t196 = bitcast double 0x0000000000000000 to i64
  %t197 = getelementptr inbounds i64, ptr %t143, i64 26
  store i64 %t196, ptr %t197, align 8
  %t198 = bitcast double 0x0000000000000000 to i64
  %t199 = getelementptr inbounds i64, ptr %t143, i64 27
  store i64 %t198, ptr %t199, align 8
  %t200 = bitcast double 0x0000000000000000 to i64
  %t201 = getelementptr inbounds i64, ptr %t143, i64 28
  store i64 %t200, ptr %t201, align 8
  %t202 = bitcast double 0x0000000000000000 to i64
  %t203 = getelementptr inbounds i64, ptr %t143, i64 29
  store i64 %t202, ptr %t203, align 8
  %t204 = bitcast double 0x0000000000000000 to i64
  %t205 = getelementptr inbounds i64, ptr %t143, i64 30
  store i64 %t204, ptr %t205, align 8
  %t206 = bitcast double 0x0000000000000000 to i64
  %t207 = getelementptr inbounds i64, ptr %t143, i64 31
  store i64 %t206, ptr %t207, align 8
  %t208 = bitcast double 0x0000000000000000 to i64
  %t209 = getelementptr inbounds i64, ptr %t143, i64 32
  store i64 %t208, ptr %t209, align 8
  %t210 = bitcast double 0x0000000000000000 to i64
  %t211 = getelementptr inbounds i64, ptr %t143, i64 33
  store i64 %t210, ptr %t211, align 8
  %t212 = bitcast double 0x0000000000000000 to i64
  %t213 = getelementptr inbounds i64, ptr %t143, i64 34
  store i64 %t212, ptr %t213, align 8
  %t214 = bitcast double 0x0000000000000000 to i64
  %t215 = getelementptr inbounds i64, ptr %t143, i64 35
  store i64 %t214, ptr %t215, align 8
  %t216 = bitcast double 0x0000000000000000 to i64
  %t217 = getelementptr inbounds i64, ptr %t143, i64 36
  store i64 %t216, ptr %t217, align 8
  %t218 = bitcast double 0x0000000000000000 to i64
  %t219 = getelementptr inbounds i64, ptr %t143, i64 37
  store i64 %t218, ptr %t219, align 8
  %t220 = bitcast double 0x0000000000000000 to i64
  %t221 = getelementptr inbounds i64, ptr %t143, i64 38
  store i64 %t220, ptr %t221, align 8
  %t222 = bitcast double 0x0000000000000000 to i64
  %t223 = getelementptr inbounds i64, ptr %t143, i64 39
  store i64 %t222, ptr %t223, align 8
  %t224 = bitcast double 0x0000000000000000 to i64
  %t225 = getelementptr inbounds i64, ptr %t143, i64 40
  store i64 %t224, ptr %t225, align 8
  %t226 = bitcast double 0x0000000000000000 to i64
  %t227 = getelementptr inbounds i64, ptr %t143, i64 41
  store i64 %t226, ptr %t227, align 8
  %t228 = bitcast double 0x0000000000000000 to i64
  %t229 = getelementptr inbounds i64, ptr %t143, i64 42
  store i64 %t228, ptr %t229, align 8
  %t230 = bitcast double 0x0000000000000000 to i64
  %t231 = getelementptr inbounds i64, ptr %t143, i64 43
  store i64 %t230, ptr %t231, align 8
  %t232 = bitcast double 0x0000000000000000 to i64
  %t233 = getelementptr inbounds i64, ptr %t143, i64 44
  store i64 %t232, ptr %t233, align 8
  %t234 = bitcast double 0x0000000000000000 to i64
  %t235 = getelementptr inbounds i64, ptr %t143, i64 45
  store i64 %t234, ptr %t235, align 8
  %t236 = bitcast double 0x0000000000000000 to i64
  %t237 = getelementptr inbounds i64, ptr %t143, i64 46
  store i64 %t236, ptr %t237, align 8
  %t238 = bitcast double 0x0000000000000000 to i64
  %t239 = getelementptr inbounds i64, ptr %t143, i64 47
  store i64 %t238, ptr %t239, align 8
  %t240 = bitcast double 0x0000000000000000 to i64
  %t241 = getelementptr inbounds i64, ptr %t143, i64 48
  store i64 %t240, ptr %t241, align 8
  %t242 = bitcast double 0x0000000000000000 to i64
  %t243 = getelementptr inbounds i64, ptr %t143, i64 49
  store i64 %t242, ptr %t243, align 8
  %t244 = bitcast double 0x0000000000000000 to i64
  %t245 = getelementptr inbounds i64, ptr %t143, i64 50
  store i64 %t244, ptr %t245, align 8
  %t246 = bitcast double 0x0000000000000000 to i64
  %t247 = getelementptr inbounds i64, ptr %t143, i64 51
  store i64 %t246, ptr %t247, align 8
  %t248 = bitcast double 0x0000000000000000 to i64
  %t249 = getelementptr inbounds i64, ptr %t143, i64 52
  store i64 %t248, ptr %t249, align 8
  %t250 = bitcast double 0x0000000000000000 to i64
  %t251 = getelementptr inbounds i64, ptr %t143, i64 53
  store i64 %t250, ptr %t251, align 8
  %t252 = bitcast double 0x0000000000000000 to i64
  %t253 = getelementptr inbounds i64, ptr %t143, i64 54
  store i64 %t252, ptr %t253, align 8
  %t254 = bitcast double 0x0000000000000000 to i64
  %t255 = getelementptr inbounds i64, ptr %t143, i64 55
  store i64 %t254, ptr %t255, align 8
  %t256 = bitcast double 0x0000000000000000 to i64
  %t257 = getelementptr inbounds i64, ptr %t143, i64 56
  store i64 %t256, ptr %t257, align 8
  %t258 = bitcast double 0x0000000000000000 to i64
  %t259 = getelementptr inbounds i64, ptr %t143, i64 57
  store i64 %t258, ptr %t259, align 8
  %t260 = bitcast double 0x0000000000000000 to i64
  %t261 = getelementptr inbounds i64, ptr %t143, i64 58
  store i64 %t260, ptr %t261, align 8
  %t262 = bitcast double 0x0000000000000000 to i64
  %t263 = getelementptr inbounds i64, ptr %t143, i64 59
  store i64 %t262, ptr %t263, align 8
  %t264 = bitcast double 0x0000000000000000 to i64
  %t265 = getelementptr inbounds i64, ptr %t143, i64 60
  store i64 %t264, ptr %t265, align 8
  %t266 = bitcast double 0x0000000000000000 to i64
  %t267 = getelementptr inbounds i64, ptr %t143, i64 61
  store i64 %t266, ptr %t267, align 8
  %t268 = bitcast double 0x0000000000000000 to i64
  %t269 = getelementptr inbounds i64, ptr %t143, i64 62
  store i64 %t268, ptr %t269, align 8
  %t270 = bitcast double 0x0000000000000000 to i64
  %t271 = getelementptr inbounds i64, ptr %t143, i64 63
  store i64 %t270, ptr %t271, align 8
  %t272 = alloca i64, align 8
  store i64 0, ptr %t272, align 8
  %t273 = alloca i64, align 8
  store i64 0, ptr %t273, align 8
  br label %rd_cond_274
rd_cond_274:
  %t275 = load i64, ptr %t273, align 8
  %t276 = icmp ult i64 %t275, 64
  br i1 %t276, label %rd_body_274, label %rd_end_274
rd_body_274:
  %t277 = getelementptr inbounds i64, ptr %array_sizes, i64 1
  %t278 = load i64, ptr %t277, align 8
  %t279 = getelementptr inbounds ptr, ptr %arrays, i64 1
  %t280 = load ptr, ptr %t279, align 8
  %t281 = load i64, ptr %t273, align 8
  %t282 = icmp slt i64 %t281, 0
  %t283 = xor i1 %t282, true
  %t284 = icmp ult i64 %t281, %t278
  %t285 = and i1 %t283, %t284
  %t286 = getelementptr inbounds i64, ptr %t280, i64 %t281
  %t287 = load i64, ptr %t286, align 8
  %t288 = bitcast i64 %t287 to double
  %t289 = select i1 %t285, double %t288, double 0x0000000000000000
  %t290 = sitofp i64 %t10 to double
  %t291 = fdiv double %t290, 0x41f0000000000000
  %t292 = fcmp oeq double 0x41f0000000000000, 0x0000000000000000
  %t293 = select i1 %t292, double 0x0000000000000000, double %t291
  %t294 = fdiv double %t293, %sampleRate
  %t295 = fcmp oeq double %sampleRate, 0x0000000000000000
  %t296 = select i1 %t295, double 0x0000000000000000, double %t294
  %t297 = fmul double %t289, %t296
  %t298 = fneg double %t297
  %t299 = fcmp ogt double %t298, 0xc055c00000000000
  %t300 = select i1 %t299, double %t298, double 0xc055c00000000000
  %t301 = fcmp olt double %t300, 0x4056000000000000
  %t302 = select i1 %t301, double %t300, double 0x4056000000000000
  %t303 = fmul double %t302, 0x3ff71547652b82fe
  %t304 = call double @llvm.round.f64(double %t303)
  %t305 = fmul double %t304, 0x3fe62e4000000000
  %t306 = fsub double %t302, %t305
  %t307 = fmul double %t304, 0x3eb7f7d1cf79abca
  %t308 = fsub double %t306, %t307
  %t309 = fmul double 0x3f2a0d2ce64969e6, %t308
  %t310 = fadd double 0x3f56e879c3f007dd, %t309
  %t311 = fmul double %t310, %t308
  %t312 = fadd double 0x3f811120fb3cb51d, %t311
  %t313 = fmul double %t312, %t308
  %t314 = fadd double 0x3fa555381d73fd31, %t313
  %t315 = fmul double %t314, %t308
  %t316 = fadd double 0x3fc555553b661d99, %t315
  %t317 = fmul double %t316, %t308
  %t318 = fadd double 0x3fe000000672a44f, %t317
  %t319 = fmul double %t308, %t318
  %t320 = fadd double 0x3ff0000000000000, %t319
  %t321 = fmul double %t308, %t320
  %t322 = fadd double 0x3ff0000000000000, %t321
  %t323 = fptosi double %t304 to i64
  %t324 = add i64 %t323, 1023
  %t325 = shl i64 %t324, 52
  %t326 = bitcast i64 %t325 to double
  %t327 = fmul double %t322, %t326
  %t328 = getelementptr inbounds i64, ptr %array_sizes, i64 2
  %t329 = load i64, ptr %t328, align 8
  %t330 = getelementptr inbounds ptr, ptr %arrays, i64 2
  %t331 = load ptr, ptr %t330, align 8
  %t332 = load i64, ptr %t273, align 8
  %t333 = icmp slt i64 %t332, 0
  %t334 = xor i1 %t333, true
  %t335 = icmp ult i64 %t332, %t329
  %t336 = and i1 %t334, %t335
  %t337 = getelementptr inbounds i64, ptr %t331, i64 %t332
  %t338 = load i64, ptr %t337, align 8
  %t339 = bitcast i64 %t338 to double
  %t340 = select i1 %t336, double %t339, double 0x0000000000000000
  %t341 = fmul double %t327, %t340
  %t342 = fmul double %t341, 0x41b0000000000000
  %t343 = fptosi double %t342 to i64
  %t344 = fptosi double 0x3ff0000000000000 to i64
  %t345 = fptosi double 0x4000000000000000 to i64
  %t346 = getelementptr inbounds i64, ptr %array_sizes, i64 0
  %t347 = load i64, ptr %t346, align 8
  %t348 = getelementptr inbounds ptr, ptr %arrays, i64 0
  %t349 = load ptr, ptr %t348, align 8
  %t350 = load i64, ptr %t273, align 8
  %t351 = icmp slt i64 %t350, 0
  %t352 = xor i1 %t351, true
  %t353 = icmp ult i64 %t350, %t347
  %t354 = and i1 %t352, %t353
  %t355 = getelementptr inbounds i64, ptr %t349, i64 %t350
  %t356 = load i64, ptr %t355, align 8
  %t357 = bitcast i64 %t356 to double
  %t358 = select i1 %t354, double %t357, double 0x0000000000000000
  %t359 = fptosi double %t358 to i64
  %t360 = getelementptr inbounds double, ptr %slots, i64 9
  %t361 = load double, ptr %t360, align 8
  %t362 = fptosi double %t361 to i64
  %t363 = mul i64 %t362, %current_idx
  %t364 = getelementptr inbounds double, ptr %slots, i64 8
  %t365 = load double, ptr %t364, align 8
  %t366 = fptosi double %t365 to i64
  %t367 = add i64 %t366, %t363
  %t368 = fptosi double %t8 to i64
  %t369 = sub i64 %t367, %t368
  %t370 = ashr i64 %t369, 32
  %t371 = mul i64 %t359, %t370
  %t372 = and i64 %t369, 4294967295
  %t373 = mul i64 %t359, %t372
  %t374 = ashr i64 %t373, 32
  %t375 = add i64 %t371, %t374
  %t376 = and i64 %t375, 4294967295
  %t377 = add i64 %t376, 1073741824
  %t378 = and i64 %t377, 4294967295
  %t379 = add i64 %t378, 1073741824
  %t380 = ashr i64 %t379, 31
  %t381 = and i64 %t380, 1
  %t382 = mul i64 %t345, %t381
  %t383 = sub i64 %t344, %t382
  %t384 = shl i64 %t380, 31
  %t385 = sub i64 %t378, %t384
  %t386 = mul i64 %t385, %t385
  %t387 = ashr i64 %t386, 30
  %t388 = ashr i64 %t387, 30
  %t389 = sub i64 61, %t388
  %t390 = mul i64 %t389, %t387
  %t391 = ashr i64 %t390, 30
  %t392 = sub i64 3864, %t391
  %t393 = mul i64 %t392, %t387
  %t394 = ashr i64 %t393, 30
  %t395 = sub i64 172272, %t394
  %t396 = mul i64 %t395, %t387
  %t397 = ashr i64 %t396, 30
  %t398 = sub i64 5026995, %t397
  %t399 = mul i64 %t398, %t387
  %t400 = ashr i64 %t399, 30
  %t401 = sub i64 85569306, %t400
  %t402 = mul i64 %t401, %t387
  %t403 = ashr i64 %t402, 30
  %t404 = sub i64 693598668, %t403
  %t405 = mul i64 %t404, %t387
  %t406 = ashr i64 %t405, 30
  %t407 = sub i64 1686629713, %t406
  %t408 = mul i64 %t385, %t407
  %t409 = ashr i64 %t408, 30
  %t410 = mul i64 %t383, %t409
  %t411 = mul i64 %t343, %t410
  %t412 = getelementptr inbounds i64, ptr %array_sizes, i64 3
  %t413 = load i64, ptr %t412, align 8
  %t414 = getelementptr inbounds ptr, ptr %arrays, i64 3
  %t415 = load ptr, ptr %t414, align 8
  %t416 = load i64, ptr %t273, align 8
  %t417 = icmp slt i64 %t416, 0
  %t418 = xor i1 %t417, true
  %t419 = icmp ult i64 %t416, %t413
  %t420 = and i1 %t418, %t419
  %t421 = getelementptr inbounds i64, ptr %t415, i64 %t416
  %t422 = load i64, ptr %t421, align 8
  %t423 = bitcast i64 %t422 to double
  %t424 = select i1 %t420, double %t423, double 0x0000000000000000
  %t425 = fmul double %t327, %t424
  %t426 = fmul double %t425, 0x41b0000000000000
  %t427 = fptosi double %t426 to i64
  %t428 = ashr i64 %t377, 31
  %t429 = and i64 %t428, 1
  %t430 = mul i64 %t345, %t429
  %t431 = sub i64 %t344, %t430
  %t432 = shl i64 %t428, 31
  %t433 = sub i64 %t376, %t432
  %t434 = mul i64 %t433, %t433
  %t435 = ashr i64 %t434, 30
  %t436 = ashr i64 %t435, 30
  %t437 = sub i64 61, %t436
  %t438 = mul i64 %t437, %t435
  %t439 = ashr i64 %t438, 30
  %t440 = sub i64 3864, %t439
  %t441 = mul i64 %t440, %t435
  %t442 = ashr i64 %t441, 30
  %t443 = sub i64 172272, %t442
  %t444 = mul i64 %t443, %t435
  %t445 = ashr i64 %t444, 30
  %t446 = sub i64 5026995, %t445
  %t447 = mul i64 %t446, %t435
  %t448 = ashr i64 %t447, 30
  %t449 = sub i64 85569306, %t448
  %t450 = mul i64 %t449, %t435
  %t451 = ashr i64 %t450, 30
  %t452 = sub i64 693598668, %t451
  %t453 = mul i64 %t452, %t435
  %t454 = ashr i64 %t453, 30
  %t455 = sub i64 1686629713, %t454
  %t456 = mul i64 %t433, %t455
  %t457 = ashr i64 %t456, 30
  %t458 = mul i64 %t431, %t457
  %t459 = mul i64 %t427, %t458
  %t460 = sub i64 %t411, %t459
  %t461 = ashr i64 %t460, 28
  %t462 = load i64, ptr %t272, align 8
  %t463 = add i64 %t462, %t461
  store i64 %t463, ptr %t272, align 8
  %t464 = load i64, ptr %t273, align 8
  %t465 = add i64 %t464, 1
  store i64 %t465, ptr %t273, align 8
  br label %rd_cond_274
rd_end_274:
  %t466 = load i64, ptr %t272, align 8
  %t467 = sitofp i64 %t466 to double
  %t468 = fdiv double %t467, 0x41d0000000000000
  %t469 = fcmp oeq double 0x41d0000000000000, 0x0000000000000000
  %t470 = select i1 %t469, double 0x0000000000000000, double %t468
  %t471 = select i1 %t11, double %t470, double 0x0000000000000000
  %t472 = getelementptr inbounds double, ptr %slots, i64 2
  %t473 = load double, ptr %t472, align 8
  %t474 = fmul double %t471, %t473
  %t475 = fadd double %t474, 0x0000000000000000
  %t476 = getelementptr inbounds double, ptr %slots, i64 5
  store double %t475, ptr %t476, align 8
  %t477 = getelementptr inbounds double, ptr %slots, i64 5
  %t478 = load double, ptr %t477, align 8
  %t479 = fadd double 0x0000000000000000, %t478
  %t480 = fmul double %t479, 0x3ff0000000000000
  %t481 = getelementptr inbounds double, ptr %output_buffer, i64 %s
  store double %t480, ptr %t481, align 8
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
