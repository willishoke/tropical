; ModuleID = 'current_vs_proposed.c'
source_filename = "current_vs_proposed.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx16.0.0"

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite) uwtable(sync)
define void @current_unified_kernel(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %15

6:                                                ; preds = %4
  %7 = load double, ptr %0, align 8, !tbaa !6
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load double, ptr %1, align 8, !tbaa !6
  %10 = load double, ptr %8, align 8, !tbaa !6
  %11 = zext nneg i32 %3 to i64
  br label %16

12:                                               ; preds = %16
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store double %20, ptr %1, align 8, !tbaa !6
  store double %19, ptr %13, align 8, !tbaa !6
  store double %22, ptr %8, align 8, !tbaa !6
  store double %22, ptr %14, align 8, !tbaa !6
  br label %15

15:                                               ; preds = %12, %4
  ret void

16:                                               ; preds = %6, %16
  %17 = phi i64 [ 0, %6 ], [ %25, %16 ]
  %18 = phi double [ %10, %6 ], [ %22, %16 ]
  %19 = phi double [ %9, %6 ], [ %20, %16 ]
  %20 = tail call double @llvm.fmuladd.f64(double %7, double 1.000000e-04, double %19)
  %21 = fmul double %18, 5.000000e-01
  %22 = tail call double @llvm.fmuladd.f64(double %19, double 5.000000e-01, double %21)
  %23 = fptrunc double %22 to float
  %24 = getelementptr inbounds nuw float, ptr %2, i64 %17
  store float %23, ptr %24, align 4, !tbaa !10
  %25 = add nuw nsw i64 %17, 1
  %26 = icmp eq i64 %25, %11
  br i1 %26, label %12, label %16, !llvm.loop !12
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @proposed_scheduler_always_on(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #2 {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %15

6:                                                ; preds = %4
  %7 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !14, !noalias !17
  %8 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %9 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %10 = load double, ptr %8, align 8, !tbaa !6, !alias.scope !19, !noalias !22
  %11 = zext nneg i32 %3 to i64
  br label %16

12:                                               ; preds = %16
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %14 = getelementptr inbounds nuw i8, ptr %0, i64 16
  store double %20, ptr %1, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %19, ptr %13, align 8, !tbaa !6, !alias.scope !14, !noalias !17
  store double %22, ptr %8, align 8, !tbaa !6, !alias.scope !19, !noalias !22
  store double %22, ptr %14, align 8, !tbaa !6, !alias.scope !22, !noalias !19
  br label %15

15:                                               ; preds = %12, %4
  ret void

16:                                               ; preds = %6, %16
  %17 = phi i64 [ 0, %6 ], [ %25, %16 ]
  %18 = phi double [ %10, %6 ], [ %22, %16 ]
  %19 = phi double [ %9, %6 ], [ %20, %16 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !14)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !17)
  %20 = tail call double @llvm.fmuladd.f64(double %7, double 1.000000e-04, double %19)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !22)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !19)
  %21 = fmul double %18, 5.000000e-01
  %22 = tail call double @llvm.fmuladd.f64(double %19, double 5.000000e-01, double %21)
  %23 = fptrunc double %22 to float
  %24 = getelementptr inbounds nuw float, ptr %2, i64 %17
  store float %23, ptr %24, align 4, !tbaa !10
  %25 = add nuw nsw i64 %17, 1
  %26 = icmp eq i64 %25, %11
  br i1 %26, label %12, label %16, !llvm.loop !24
}

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @proposed_scheduler_dynamic(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #2 {
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
  tail call void @llvm.experimental.noalias.scope.decl(metadata !25)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !28)
  %23 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !25, !noalias !28
  %24 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !28, !noalias !25
  store double %24, ptr %10, align 8, !tbaa !6, !alias.scope !25, !noalias !28
  %25 = tail call double @llvm.fmuladd.f64(double %23, double 1.000000e-04, double %24)
  store double %25, ptr %1, align 8, !tbaa !6, !alias.scope !28, !noalias !25
  br label %26

26:                                               ; preds = %22, %19
  br i1 %13, label %27, label %32

27:                                               ; preds = %26
  tail call void @llvm.experimental.noalias.scope.decl(metadata !30)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !33)
  %28 = load double, ptr %10, align 8, !tbaa !6, !alias.scope !30, !noalias !33
  %29 = load double, ptr %14, align 8, !tbaa !6, !alias.scope !33, !noalias !30
  %30 = fmul double %29, 5.000000e-01
  %31 = tail call double @llvm.fmuladd.f64(double %28, double 5.000000e-01, double %30)
  store double %31, ptr %15, align 8, !tbaa !6, !alias.scope !30, !noalias !33
  store double %31, ptr %14, align 8, !tbaa !6, !alias.scope !33, !noalias !30
  br label %32

32:                                               ; preds = %27, %26
  %33 = phi double [ %31, %27 ], [ %21, %26 ]
  %34 = fptrunc double %33 to float
  %35 = getelementptr inbounds nuw float, ptr %2, i64 %20
  store float %34, ptr %35, align 4, !tbaa !10
  %36 = add nuw nsw i64 %20, 1
  %37 = icmp eq i64 %36, %17
  br i1 %37, label %18, label %19, !llvm.loop !35
}

; Function Attrs: nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite)
declare void @llvm.experimental.noalias.scope.decl(metadata) #3

attributes #0 = { nofree norecurse nosync nounwind ssp memory(argmem: readwrite) uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #1 = { mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none) }
attributes #2 = { nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync) "frame-pointer"="non-leaf" "no-trapping-math"="true" "stack-protector-buffer-size"="8" "target-cpu"="apple-m1" "target-features"="+aes,+altnzcv,+ccdp,+ccidx,+ccpp,+complxnum,+crc,+dit,+dotprod,+flagm,+fp-armv8,+fp16fml,+fptoint,+fullfp16,+jsconv,+lse,+neon,+pauth,+perfmon,+predres,+ras,+rcpc,+rdm,+sb,+sha2,+sha3,+specrestrict,+ssbs,+v8.1a,+v8.2a,+v8.3a,+v8.4a,+v8a,+zcm,+zcz" }
attributes #3 = { nocallback nofree nosync nounwind willreturn memory(inaccessiblemem: readwrite) }

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
!10 = !{!11, !11, i64 0}
!11 = !{!"float", !8, i64 0}
!12 = distinct !{!12, !13}
!13 = !{!"llvm.loop.mustprogress"}
!14 = !{!15}
!15 = distinct !{!15, !16, !"osc_instance: argument 0"}
!16 = distinct !{!16, !"osc_instance"}
!17 = !{!18}
!18 = distinct !{!18, !16, !"osc_instance: argument 1"}
!19 = !{!20}
!20 = distinct !{!20, !21, !"onepole_instance: argument 1"}
!21 = distinct !{!21, !"onepole_instance"}
!22 = !{!23}
!23 = distinct !{!23, !21, !"onepole_instance: argument 0"}
!24 = distinct !{!24, !13}
!25 = !{!26}
!26 = distinct !{!26, !27, !"osc_instance: argument 0"}
!27 = distinct !{!27, !"osc_instance"}
!28 = !{!29}
!29 = distinct !{!29, !27, !"osc_instance: argument 1"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"onepole_instance: argument 0"}
!32 = distinct !{!32, !"onepole_instance"}
!33 = !{!34}
!34 = distinct !{!34, !32, !"onepole_instance: argument 1"}
!35 = distinct !{!35, !13}
