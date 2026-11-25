define void @tropical_kernel(ptr noalias nocapture readonly %tau, ptr noalias nocapture readonly %in_1, ptr noalias nocapture readonly %in_2, ptr noalias nocapture %out) nounwind {
entry:
  %slots = alloca double, i64 6, align 8
  %temps = alloca i64, i64 67, align 8
  %islots = alloca i64, i64 6, align 8
  %i32slots = alloca i32, i64 6, align 4
  br label %zero_cond_0
zero_cond_0:
  %t1 = phi i64 [ 0, %entry ], [ %t1_next, %zero_body_0 ]
  %t2 = icmp ult i64 %t1, 67
  br i1 %t2, label %zero_body_0, label %zero_end_0
zero_body_0:
  %t3 = getelementptr inbounds i64, ptr %temps, i64 %t1
  store i64 0, ptr %t3, align 8
  %t1_next = add i64 %t1, 1
  br label %zero_cond_0
zero_end_0:
  br label %zero_cond_4
zero_cond_4:
  %t5 = phi i64 [ 0, %zero_end_0 ], [ %t5_next, %zero_body_4 ]
  %t6 = icmp ult i64 %t5, 6
  br i1 %t6, label %zero_body_4, label %zero_end_4
zero_body_4:
  %t7 = getelementptr inbounds i64, ptr %slots, i64 %t5
  store i64 0, ptr %t7, align 8
  %t5_next = add i64 %t5, 1
  br label %zero_cond_4
zero_end_4:
  br label %zero_cond_8
zero_cond_8:
  %t9 = phi i64 [ 0, %zero_end_4 ], [ %t9_next, %zero_body_8 ]
  %t10 = icmp ult i64 %t9, 6
  br i1 %t10, label %zero_body_8, label %zero_end_8
zero_body_8:
  %t11 = getelementptr inbounds i64, ptr %islots, i64 %t9
  store i64 0, ptr %t11, align 8
  %t9_next = add i64 %t9, 1
  br label %zero_cond_8
zero_end_8:
  %t12 = load i32, ptr %in_1, align 4
  %t13 = getelementptr inbounds i32, ptr %i32slots, i64 0
  store i32 %t12, ptr %t13, align 4
  %t14 = sitofp i32 %t12 to double
  %t15 = getelementptr inbounds double, ptr %slots, i64 0
  store double %t14, ptr %t15, align 8
  %t16 = load i32, ptr %in_2, align 4
  %t17 = getelementptr inbounds i32, ptr %i32slots, i64 1
  store i32 %t16, ptr %t17, align 4
  %t18 = sitofp i32 %t16 to double
  %t19 = getelementptr inbounds double, ptr %slots, i64 1
  store double %t18, ptr %t19, align 8
  %tau0 = add i64 0, 0
  br label %loop_cond
loop_cond:
  %s = phi i32 [ 0, %zero_end_8 ], [ %s_next, %loop_body ]
  %loopcond = icmp ult i32 %s, 1
  br i1 %loopcond, label %loop_body, label %loop_end
loop_body:
  %tau_ptr = getelementptr inbounds i64, ptr %tau, i32 %s
  %tau_s = load i64, ptr %tau_ptr, align 8
  %t20 = getelementptr inbounds i32, ptr %i32slots, i64 0
  %t21 = load i32, ptr %t20, align 4
  %t22 = getelementptr inbounds i32, ptr %i32slots, i64 3
  store i32 %t21, ptr %t22, align 4
  %t23 = sitofp i32 %t21 to double
  %t24 = getelementptr inbounds double, ptr %slots, i64 3
  store double %t23, ptr %t24, align 8
  %t25 = getelementptr inbounds i64, ptr %islots, i64 4
  store i64 %tau_s, ptr %t25, align 8
  %t26 = sitofp i64 %tau_s to double
  %t27 = getelementptr inbounds double, ptr %slots, i64 4
  store double %t26, ptr %t27, align 8
  %t28 = getelementptr inbounds i32, ptr %i32slots, i64 1
  %t29 = load i32, ptr %t28, align 4
  %t30 = getelementptr inbounds i32, ptr %i32slots, i64 5
  store i32 %t29, ptr %t30, align 4
  %t31 = sitofp i32 %t29 to double
  %t32 = getelementptr inbounds double, ptr %slots, i64 5
  store double %t31, ptr %t32, align 8
  %t33 = fdiv double 0x42f0000000000000, 0x40e7700000000000
  %t34 = fcmp oeq double 0x40e7700000000000, 0x0000000000000000
  %t35 = select i1 %t34, double 0x0000000000000000, double %t33
  %t36 = call double @llvm.round.f64(double %t35)
  %t37 = call i64 @llvm.fptosi.sat.i64.f64(double %t36)
  %t38 = ashr i64 %t37, 32
  %t39 = trunc i64 %t38 to i32
  %t40 = getelementptr inbounds i32, ptr %i32slots, i64 3
  %t41 = load i32, ptr %t40, align 4
  %t42 = mul i32 %t39, %t41
  %t43 = getelementptr inbounds i32, ptr %i32slots, i64 3
  %t44 = load i32, ptr %t43, align 4
  %t45 = ashr i32 %t44, 16
  %t46 = and i64 %t37, 4294967295
  %t47 = ashr i64 %t46, 16
  %t48 = trunc i64 %t47 to i32
  %t49 = mul i32 %t45, %t48
  %t50 = add i32 %t42, %t49
  %t51 = and i64 %t46, 65535
  %t52 = trunc i64 %t51 to i32
  %t53 = mul i32 %t45, %t52
  %t54 = ashr i32 %t53, 16
  %t55 = add i32 %t50, %t54
  %t56 = getelementptr inbounds i32, ptr %i32slots, i64 3
  %t57 = load i32, ptr %t56, align 4
  %t58 = and i32 %t57, 65535
  %t59 = mul i32 %t58, %t48
  %t60 = ashr i32 %t59, 16
  %t61 = and i32 %t60, 65535
  %t62 = add i32 %t55, %t61
  %t63 = getelementptr inbounds i64, ptr %islots, i64 4
  %t64 = load i64, ptr %t63, align 8
  %t65 = ashr i64 %t64, 16
  %t66 = trunc i64 %t65 to i32
  %t67 = mul i32 %t62, %t66
  %t68 = ashr i32 %t62, 16
  %t69 = getelementptr inbounds i64, ptr %islots, i64 4
  %t70 = load i64, ptr %t69, align 8
  %t71 = and i64 %t70, 65535
  %t72 = trunc i64 %t71 to i32
  %t73 = mul i32 %t68, %t72
  %t74 = add i32 %t67, %t73
  %t75 = and i32 %t62, 65535
  %t76 = mul i32 %t75, %t72
  %t77 = ashr i32 %t76, 16
  %t78 = and i32 %t77, 65535
  %t79 = add i32 %t74, %t78
  %t80 = getelementptr inbounds i32, ptr %i32slots, i64 5
  %t81 = load i32, ptr %t80, align 4
  %t82 = add i32 %t79, %t81
  %t83 = add i32 %t82, 32768
  %t84 = ashr i32 %t83, 16
  %t85 = and i32 %t84, 65535
  %t86 = add i32 %t85, 16384
  %t87 = ashr i32 %t86, 15
  %t88 = and i32 %t87, 1
  %t89 = mul i32 2, %t88
  %t90 = sub i32 1, %t89
  %t91 = shl i32 %t87, 15
  %t92 = sub i32 %t85, %t91
  %t93 = mul i32 %t90, %t92
  %t94 = mul i32 %t93, %t93
  %t95 = add i32 %t94, 8192
  %t96 = ashr i32 %t95, 14
  %t97 = mul i32 141, %t96
  %t98 = add i32 %t97, 8192
  %t99 = ashr i32 %t98, 14
  %t100 = sub i32 2604, %t99
  %t101 = mul i32 %t100, %t96
  %t102 = add i32 %t101, 8192
  %t103 = ashr i32 %t102, 14
  %t104 = sub i32 21169, %t103
  %t105 = mul i32 %t104, %t96
  %t106 = add i32 %t105, 8192
  %t107 = ashr i32 %t106, 14
  %t108 = sub i32 51474, %t107
  %t109 = mul i32 %t93, %t108
  %t110 = add i32 %t109, 8192
  %t111 = ashr i32 %t110, 14
  %t112 = icmp sgt i32 %t111, -32767
  %t113 = select i1 %t112, i32 %t111, i32 -32767
  %t114 = icmp slt i32 %t113, 32767
  %t115 = select i1 %t114, i32 %t113, i32 32767
  %t116 = add i32 %t115, 0
  %t117 = getelementptr inbounds i32, ptr %i32slots, i64 2
  store i32 %t116, ptr %t117, align 4
  %t118 = sitofp i32 %t116 to double
  %t119 = getelementptr inbounds double, ptr %slots, i64 2
  store double %t118, ptr %t119, align 8
  %t120 = getelementptr inbounds i32, ptr %i32slots, i64 2
  %t121 = load i32, ptr %t120, align 4
  %t122 = getelementptr inbounds i32, ptr %out, i32 %s
  store i32 %t121, ptr %t122, align 4
  %s_next = add i32 %s, 1
  br label %loop_cond

loop_end:
  ret void
}

define void @tropical_kernel_argv(ptr %argv) nounwind {
entry:
  %ap0 = getelementptr inbounds ptr, ptr %argv, i64 0
  %a0 = load ptr, ptr %ap0, align 8
  %ap1 = getelementptr inbounds ptr, ptr %argv, i64 1
  %a1 = load ptr, ptr %ap1, align 8
  %ap2 = getelementptr inbounds ptr, ptr %argv, i64 2
  %a2 = load ptr, ptr %ap2, align 8
  %ap3 = getelementptr inbounds ptr, ptr %argv, i64 3
  %a3 = load ptr, ptr %ap3, align 8
  call void @tropical_kernel(ptr %a0, ptr %a1, ptr %a2, ptr %a3)
  ret void
}

declare double @llvm.sqrt.f64(double)
declare double @llvm.floor.f64(double)
declare double @llvm.ceil.f64(double)
declare double @llvm.round.f64(double)
declare double @llvm.fabs.f64(double)
declare i64 @llvm.fptosi.sat.i64.f64(double)
declare i32 @llvm.fptosi.sat.i32.f64(double)
