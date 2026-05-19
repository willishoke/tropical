; ModuleID = 'spike.c'
source_filename = "spike.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx16.0.0"

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @scheduler_A_always_on(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %15

6:                                                ; preds = %4
  %7 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !10, !noalias !13
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %10 = load double, ptr %8, align 8, !tbaa !6, !alias.scope !15, !noalias !18
  %11 = zext nneg i32 %3 to i64
  br label %16

12:                                               ; preds = %16
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store double %20, ptr %1, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %19, ptr %13, align 8, !tbaa !6, !alias.scope !10, !noalias !13
  store double %22, ptr %8, align 8, !tbaa !6, !alias.scope !15, !noalias !18
  store double %22, ptr %14, align 8, !tbaa !6, !alias.scope !18, !noalias !15
  br label %15

15:                                               ; preds = %12, %4
  ret void

16:                                               ; preds = %6, %16
  %17 = phi i64 [ 0, %6 ], [ %25, %16 ]
  %18 = phi double [ %10, %6 ], [ %22, %16 ]
  %19 = phi double [ %9, %6 ], [ %20, %16 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !13)
  %20 = tail call double @llvm.fmuladd.f64(double %7, double 1.000000e-04, double %19)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !18)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !15)
  %21 = fmul double %18, 5.000000e-01
  %22 = tail call double @llvm.fmuladd.f64(double %19, double 5.000000e-01, double %21)
  %23 = fptrunc double %22 to float
  %24 = getelementptr inbounds nuw float, ptr %2, i64 %17
  store float %23, ptr %24, align 4, !tbaa !20
  %25 = add nuw nsw i64 %17, 1
  %26 = icmp eq i64 %25, %11
  br i1 %26, label %12, label %16, !llvm.loop !22
}

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @scheduler_B_dynamic_alive(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %18

6:                                                ; preds = %4
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %8 = load double, ptr %7, align 8, !tbaa !6
  %9 = fcmp ogt double %8, 5.000000e-01
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %11 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %12 = load double, ptr %11, align 8, !tbaa !6
  %13 = fcmp ogt double %12, 5.000000e-01
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %16 = load double, ptr %15, align 8, !tbaa !6
  %17 = zext nneg i32 %3 to i64
  br label %19

18:                                               ; preds = %32, %4
  ret void

19:                                               ; preds = %6, %32
  %20 = phi i64 [ 0, %6 ], [ %36, %32 ]
  %21 = phi double [ %16, %6 ], [ %33, %32 ]
  br i1 %9, label %22, label %26

22:                                               ; preds = %19
  tail call void @llvm.experimental.noalias.scope.decl(metadata !24)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !27)
  %23 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !24, !noalias !27
  %24 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !27, !noalias !24
  store double %24, ptr %10, align 8, !tbaa !6, !alias.scope !24, !noalias !27
  %25 = tail call double @llvm.fmuladd.f64(double %23, double 1.000000e-04, double %24)
  store double %25, ptr %1, align 8, !tbaa !6, !alias.scope !27, !noalias !24
  br label %26

26:                                               ; preds = %22, %19
  br i1 %13, label %27, label %32

27:                                               ; preds = %26
  tail call void @llvm.experimental.noalias.scope.decl(metadata !29)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !32)
  %28 = load double, ptr %10, align 8, !tbaa !6, !alias.scope !29, !noalias !32
  %29 = load double, ptr %14, align 8, !tbaa !6, !alias.scope !32, !noalias !29
  %30 = fmul double %29, 5.000000e-01
  %31 = tail call double @llvm.fmuladd.f64(double %28, double 5.000000e-01, double %30)
  store double %31, ptr %15, align 8, !tbaa !6, !alias.scope !29, !noalias !32
  store double %31, ptr %14, align 8, !tbaa !6, !alias.scope !32, !noalias !29
  br label %32

32:                                               ; preds = %27, %26
  %33 = phi double [ %31, %27 ], [ %21, %26 ]
  %34 = fptrunc double %33 to float
  %35 = getelementptr inbounds nuw float, ptr %2, i64 %20
  store float %34, ptr %35, align 4, !tbaa !20
  %36 = add nuw nsw i64 %20, 1
  %37 = icmp eq i64 %36, %17
  br i1 %37, label %18, label %19, !llvm.loop !34
}

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @scheduler_C_per_block_alive(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %6 = load double, ptr %5, align 8, !tbaa !6
  %7 = fcmp ogt double %6, 5.000000e-01
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %9 = load double, ptr %8, align 8, !tbaa !6
  %10 = fcmp ogt double %9, 5.000000e-01
  %11 = icmp sgt i32 %3, 0
  br i1 %11, label %12, label %18

12:                                               ; preds = %4
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %16 = load double, ptr %15, align 8, !tbaa !6
  %17 = zext nneg i32 %3 to i64
  br label %19

18:                                               ; preds = %32, %4
  ret void

19:                                               ; preds = %12, %32
  %20 = phi i64 [ 0, %12 ], [ %36, %32 ]
  %21 = phi double [ %16, %12 ], [ %33, %32 ]
  br i1 %7, label %22, label %26

22:                                               ; preds = %19
  tail call void @llvm.experimental.noalias.scope.decl(metadata !35)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !38)
  %23 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !35, !noalias !38
  %24 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !38, !noalias !35
  store double %24, ptr %13, align 8, !tbaa !6, !alias.scope !35, !noalias !38
  %25 = tail call double @llvm.fmuladd.f64(double %23, double 1.000000e-04, double %24)
  store double %25, ptr %1, align 8, !tbaa !6, !alias.scope !38, !noalias !35
  br label %26

26:                                               ; preds = %22, %19
  br i1 %10, label %27, label %32

27:                                               ; preds = %26
  tail call void @llvm.experimental.noalias.scope.decl(metadata !40)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !43)
  %28 = load double, ptr %13, align 8, !tbaa !6, !alias.scope !40, !noalias !43
  %29 = load double, ptr %14, align 8, !tbaa !6, !alias.scope !43, !noalias !40
  %30 = fmul double %29, 5.000000e-01
  %31 = tail call double @llvm.fmuladd.f64(double %28, double 5.000000e-01, double %30)
  store double %31, ptr %15, align 8, !tbaa !6, !alias.scope !40, !noalias !43
  store double %31, ptr %14, align 8, !tbaa !6, !alias.scope !43, !noalias !40
  br label %32

32:                                               ; preds = %27, %26
  %33 = phi double [ %31, %27 ], [ %21, %26 ]
  %34 = fptrunc double %33 to float
  %35 = getelementptr inbounds nuw float, ptr %2, i64 %20
  store float %34, ptr %35, align 4, !tbaa !20
  %36 = add nuw nsw i64 %20, 1
  %37 = icmp eq i64 %36, %17
  br i1 %37, label %18, label %19, !llvm.loop !45
}

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @scheduler_D_mixed(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %18

6:                                                ; preds = %4
  %7 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !46, !noalias !49
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %9 = load double, ptr %8, align 8, !tbaa !6
  %10 = fcmp ogt double %9, 5.000000e-01
  %11 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %12 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %13 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !49, !noalias !46
  %14 = load double, ptr %12, align 8, !tbaa !6
  %15 = zext nneg i32 %3 to i64
  br label %19

16:                                               ; preds = %28
  %17 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %23, ptr %1, align 8, !tbaa !6, !alias.scope !49, !noalias !46
  store double %22, ptr %17, align 8, !tbaa !6, !alias.scope !46, !noalias !49
  br label %18

18:                                               ; preds = %16, %4
  ret void

19:                                               ; preds = %6, %28
  %20 = phi i64 [ 0, %6 ], [ %32, %28 ]
  %21 = phi double [ %14, %6 ], [ %29, %28 ]
  %22 = phi double [ %13, %6 ], [ %23, %28 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !46)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !49)
  %23 = tail call double @llvm.fmuladd.f64(double %7, double 1.000000e-04, double %22)
  br i1 %10, label %24, label %28

24:                                               ; preds = %19
  tail call void @llvm.experimental.noalias.scope.decl(metadata !51)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !54)
  %25 = load double, ptr %11, align 8, !tbaa !6, !alias.scope !54, !noalias !51
  %26 = fmul double %25, 5.000000e-01
  %27 = tail call double @llvm.fmuladd.f64(double %22, double 5.000000e-01, double %26)
  store double %27, ptr %12, align 8, !tbaa !6, !alias.scope !51, !noalias !54
  store double %27, ptr %11, align 8, !tbaa !6, !alias.scope !54, !noalias !51
  br label %28

28:                                               ; preds = %24, %19
  %29 = phi double [ %27, %24 ], [ %21, %19 ]
  %30 = fptrunc double %29 to float
  %31 = getelementptr inbounds nuw float, ptr %2, i64 %20
  store float %30, ptr %31, align 4, !tbaa !20
  %32 = add nuw nsw i64 %20, 1
  %33 = icmp eq i64 %32, %15
  br i1 %33, label %16, label %19, !llvm.loop !56
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #2

attributes #0 = { nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

!llvm.module.flags = !{!0, !1, !2, !3, !4}
!llvm.ident = !{!5}

!0 = !{i32 2, !"SDK Version", [2 x i32] [i32 26, i32 2]}
!1 = !{i32 1, !"wchar_size", i32 4}
!2 = !{i32 8, !"PIC Level", i32 2}
!3 = !{i32 7, !"uwtable", i32 1}
!4 = !{i32 7, !"frame-pointer", i32 1}
!5 = !{!"Homebrew clang version 20.1.8"}
!6 = !{!7, !7, i64 0}
!7 = !{!"double", !8, i64 0}
!8 = !{!"omnipotent char", !9, i64 0}
!9 = !{!"Simple C/C++ TBAA"}
!10 = !{!11}
!11 = distinct !{!11, !12, !"osc_A: argument 0"}
!12 = distinct !{!12, !"osc_A"}
!13 = !{!14}
!14 = distinct !{!14, !12, !"osc_A: argument 1"}
!15 = !{!16}
!16 = distinct !{!16, !17, !"filter_A: argument 1"}
!17 = distinct !{!17, !"filter_A"}
!18 = !{!19}
!19 = distinct !{!19, !17, !"filter_A: argument 0"}
!20 = !{!21, !21, i64 0}
!21 = !{!"float", !8, i64 0}
!22 = distinct !{!22, !23}
!23 = !{!"llvm.loop.mustprogress"}
!24 = !{!25}
!25 = distinct !{!25, !26, !"osc_B: argument 0"}
!26 = distinct !{!26, !"osc_B"}
!27 = !{!28}
!28 = distinct !{!28, !26, !"osc_B: argument 1"}
!29 = !{!30}
!30 = distinct !{!30, !31, !"filter_B: argument 0"}
!31 = distinct !{!31, !"filter_B"}
!32 = !{!33}
!33 = distinct !{!33, !31, !"filter_B: argument 1"}
!34 = distinct !{!34, !23}
!35 = !{!36}
!36 = distinct !{!36, !37, !"osc_C: argument 0"}
!37 = distinct !{!37, !"osc_C"}
!38 = !{!39}
!39 = distinct !{!39, !37, !"osc_C: argument 1"}
!40 = !{!41}
!41 = distinct !{!41, !42, !"filter_C: argument 0"}
!42 = distinct !{!42, !"filter_C"}
!43 = !{!44}
!44 = distinct !{!44, !42, !"filter_C: argument 1"}
!45 = distinct !{!45, !23}
!46 = !{!47}
!47 = distinct !{!47, !48, !"osc_D: argument 0"}
!48 = distinct !{!48, !"osc_D"}
!49 = !{!50}
!50 = distinct !{!50, !48, !"osc_D: argument 1"}
!51 = !{!52}
!52 = distinct !{!52, !53, !"filter_D: argument 0"}
!53 = distinct !{!53, !"filter_D"}
!54 = !{!55}
!55 = distinct !{!55, !53, !"filter_D: argument 1"}
!56 = distinct !{!56, !23}
