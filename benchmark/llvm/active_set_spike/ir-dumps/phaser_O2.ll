; ModuleID = 'phaser.c'
source_filename = "phaser.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx16.0.0"

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @scheduler(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %28

6:                                                ; preds = %4
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %8 = load double, ptr %7, align 8, !tbaa !6
  %9 = fcmp ogt double %8, 5.000000e-01
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %11 = load double, ptr %10, align 8, !tbaa !6
  %12 = zext nneg i32 %3 to i64
  %13 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 120
  br label %29

28:                                               ; preds = %82, %4
  ret void

29:                                               ; preds = %6, %82
  %30 = phi i64 [ 0, %6 ], [ %86, %82 ]
  %31 = phi double [ %11, %6 ], [ %83, %82 ]
  br i1 %9, label %32, label %82

32:                                               ; preds = %29
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !13)
  %33 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !10, !noalias !13
  %34 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %35 = tail call double @llvm.fmuladd.f64(double %33, double -5.000000e-01, double %34)
  %36 = tail call double @llvm.fmuladd.f64(double %35, double 5.000000e-01, double %33)
  store double %36, ptr %1, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %37 = load double, ptr %13, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %38 = tail call double @llvm.fmuladd.f64(double %35, double -5.000000e-01, double %37)
  %39 = tail call double @llvm.fmuladd.f64(double %38, double 5.000000e-01, double %35)
  store double %39, ptr %13, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %40 = load double, ptr %14, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %41 = tail call double @llvm.fmuladd.f64(double %38, double -5.000000e-01, double %40)
  %42 = tail call double @llvm.fmuladd.f64(double %41, double 5.000000e-01, double %38)
  store double %42, ptr %14, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %43 = load double, ptr %15, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %44 = tail call double @llvm.fmuladd.f64(double %41, double -5.000000e-01, double %43)
  %45 = tail call double @llvm.fmuladd.f64(double %44, double 5.000000e-01, double %41)
  store double %45, ptr %15, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %46 = load double, ptr %16, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %47 = tail call double @llvm.fmuladd.f64(double %44, double -5.000000e-01, double %46)
  %48 = tail call double @llvm.fmuladd.f64(double %47, double 5.000000e-01, double %44)
  store double %48, ptr %16, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %49 = load double, ptr %17, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %50 = tail call double @llvm.fmuladd.f64(double %47, double -5.000000e-01, double %49)
  %51 = tail call double @llvm.fmuladd.f64(double %50, double 5.000000e-01, double %47)
  store double %51, ptr %17, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %52 = load double, ptr %18, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %53 = tail call double @llvm.fmuladd.f64(double %50, double -5.000000e-01, double %52)
  %54 = tail call double @llvm.fmuladd.f64(double %53, double 5.000000e-01, double %50)
  store double %54, ptr %18, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %55 = load double, ptr %19, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %56 = tail call double @llvm.fmuladd.f64(double %53, double -5.000000e-01, double %55)
  %57 = tail call double @llvm.fmuladd.f64(double %56, double 5.000000e-01, double %53)
  store double %57, ptr %19, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %58 = load double, ptr %20, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %59 = tail call double @llvm.fmuladd.f64(double %56, double -5.000000e-01, double %58)
  %60 = tail call double @llvm.fmuladd.f64(double %59, double 5.000000e-01, double %56)
  store double %60, ptr %20, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %61 = load double, ptr %21, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %62 = tail call double @llvm.fmuladd.f64(double %59, double -5.000000e-01, double %61)
  %63 = tail call double @llvm.fmuladd.f64(double %62, double 5.000000e-01, double %59)
  store double %63, ptr %21, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %64 = load double, ptr %22, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %65 = tail call double @llvm.fmuladd.f64(double %62, double -5.000000e-01, double %64)
  %66 = tail call double @llvm.fmuladd.f64(double %65, double 5.000000e-01, double %62)
  store double %66, ptr %22, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %67 = load double, ptr %23, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %68 = tail call double @llvm.fmuladd.f64(double %65, double -5.000000e-01, double %67)
  %69 = tail call double @llvm.fmuladd.f64(double %68, double 5.000000e-01, double %65)
  store double %69, ptr %23, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %70 = load double, ptr %24, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %71 = tail call double @llvm.fmuladd.f64(double %68, double -5.000000e-01, double %70)
  %72 = tail call double @llvm.fmuladd.f64(double %71, double 5.000000e-01, double %68)
  store double %72, ptr %24, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %73 = load double, ptr %25, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %74 = tail call double @llvm.fmuladd.f64(double %71, double -5.000000e-01, double %73)
  %75 = tail call double @llvm.fmuladd.f64(double %74, double 5.000000e-01, double %71)
  store double %75, ptr %25, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %76 = load double, ptr %26, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %77 = tail call double @llvm.fmuladd.f64(double %74, double -5.000000e-01, double %76)
  %78 = tail call double @llvm.fmuladd.f64(double %77, double 5.000000e-01, double %74)
  store double %78, ptr %26, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %79 = load double, ptr %27, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %80 = tail call double @llvm.fmuladd.f64(double %77, double -5.000000e-01, double %79)
  %81 = tail call double @llvm.fmuladd.f64(double %80, double 5.000000e-01, double %77)
  store double %81, ptr %27, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %80, ptr %10, align 8, !tbaa !6, !alias.scope !10, !noalias !13
  br label %82

82:                                               ; preds = %32, %29
  %83 = phi double [ %80, %32 ], [ %31, %29 ]
  %84 = fptrunc double %83 to float
  %85 = getelementptr inbounds nuw float, ptr %2, i64 %30
  store float %84, ptr %85, align 4, !tbaa !15
  %86 = add nuw nsw i64 %30, 1
  %87 = icmp eq i64 %86, %12
  br i1 %87, label %28, label %29, !llvm.loop !17
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
!11 = distinct !{!11, !12, !"phaser16: argument 0"}
!12 = distinct !{!12, !"phaser16"}
!13 = !{!14}
!14 = distinct !{!14, !12, !"phaser16: argument 1"}
!15 = !{!16, !16, i64 0}
!16 = !{!"float", !8, i64 0}
!17 = distinct !{!17, !18}
!18 = !{!"llvm.loop.mustprogress"}
