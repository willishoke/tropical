; ModuleID = 'phaser_compare.c'
source_filename = "phaser_compare.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx16.0.0"

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite) uwtable(sync)
define void @phaser_flat(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %42

6:                                                ; preds = %4
  %7 = load double, ptr %0, align 8, !tbaa !6
  %8 = zext nneg i32 %3 to i64
  %9 = load double, ptr %1, align 8, !tbaa !6
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load double, ptr %10, align 8, !tbaa !6
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load double, ptr %12, align 8, !tbaa !6
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %15 = load double, ptr %14, align 8, !tbaa !6
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %17 = load double, ptr %16, align 8, !tbaa !6
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %19 = load double, ptr %18, align 8, !tbaa !6
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %21 = load double, ptr %20, align 8, !tbaa !6
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %23 = load double, ptr %22, align 8, !tbaa !6
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %25 = load double, ptr %24, align 8, !tbaa !6
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %27 = load double, ptr %26, align 8, !tbaa !6
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %29 = load double, ptr %28, align 8, !tbaa !6
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %31 = load double, ptr %30, align 8, !tbaa !6
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %33 = load double, ptr %32, align 8, !tbaa !6
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %35 = load double, ptr %34, align 8, !tbaa !6
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %37 = load double, ptr %36, align 8, !tbaa !6
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %39 = load double, ptr %38, align 8, !tbaa !6
  br label %43

40:                                               ; preds = %43
  store double %62, ptr %1, align 8, !tbaa !6
  store double %64, ptr %10, align 8, !tbaa !6
  store double %66, ptr %12, align 8, !tbaa !6
  store double %68, ptr %14, align 8, !tbaa !6
  store double %70, ptr %16, align 8, !tbaa !6
  store double %72, ptr %18, align 8, !tbaa !6
  store double %74, ptr %20, align 8, !tbaa !6
  store double %76, ptr %22, align 8, !tbaa !6
  store double %78, ptr %24, align 8, !tbaa !6
  store double %80, ptr %26, align 8, !tbaa !6
  store double %82, ptr %28, align 8, !tbaa !6
  store double %84, ptr %30, align 8, !tbaa !6
  store double %86, ptr %32, align 8, !tbaa !6
  store double %88, ptr %34, align 8, !tbaa !6
  store double %90, ptr %36, align 8, !tbaa !6
  store double %92, ptr %38, align 8, !tbaa !6
  %41 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %91, ptr %41, align 8, !tbaa !6
  br label %42

42:                                               ; preds = %40, %4
  ret void

43:                                               ; preds = %6, %43
  %44 = phi double [ %39, %6 ], [ %92, %43 ]
  %45 = phi double [ %37, %6 ], [ %90, %43 ]
  %46 = phi double [ %35, %6 ], [ %88, %43 ]
  %47 = phi double [ %33, %6 ], [ %86, %43 ]
  %48 = phi double [ %31, %6 ], [ %84, %43 ]
  %49 = phi double [ %29, %6 ], [ %82, %43 ]
  %50 = phi double [ %27, %6 ], [ %80, %43 ]
  %51 = phi double [ %25, %6 ], [ %78, %43 ]
  %52 = phi double [ %23, %6 ], [ %76, %43 ]
  %53 = phi double [ %21, %6 ], [ %74, %43 ]
  %54 = phi double [ %19, %6 ], [ %72, %43 ]
  %55 = phi double [ %17, %6 ], [ %70, %43 ]
  %56 = phi double [ %15, %6 ], [ %68, %43 ]
  %57 = phi double [ %13, %6 ], [ %66, %43 ]
  %58 = phi double [ %11, %6 ], [ %64, %43 ]
  %59 = phi double [ %9, %6 ], [ %62, %43 ]
  %60 = phi i64 [ 0, %6 ], [ %95, %43 ]
  %61 = tail call double @llvm.fmuladd.f64(double %7, double -5.000000e-01, double %59)
  %62 = tail call double @llvm.fmuladd.f64(double %61, double 5.000000e-01, double %7)
  %63 = tail call double @llvm.fmuladd.f64(double %61, double -5.000000e-01, double %58)
  %64 = tail call double @llvm.fmuladd.f64(double %63, double 5.000000e-01, double %61)
  %65 = tail call double @llvm.fmuladd.f64(double %63, double -5.000000e-01, double %57)
  %66 = tail call double @llvm.fmuladd.f64(double %65, double 5.000000e-01, double %63)
  %67 = tail call double @llvm.fmuladd.f64(double %65, double -5.000000e-01, double %56)
  %68 = tail call double @llvm.fmuladd.f64(double %67, double 5.000000e-01, double %65)
  %69 = tail call double @llvm.fmuladd.f64(double %67, double -5.000000e-01, double %55)
  %70 = tail call double @llvm.fmuladd.f64(double %69, double 5.000000e-01, double %67)
  %71 = tail call double @llvm.fmuladd.f64(double %69, double -5.000000e-01, double %54)
  %72 = tail call double @llvm.fmuladd.f64(double %71, double 5.000000e-01, double %69)
  %73 = tail call double @llvm.fmuladd.f64(double %71, double -5.000000e-01, double %53)
  %74 = tail call double @llvm.fmuladd.f64(double %73, double 5.000000e-01, double %71)
  %75 = tail call double @llvm.fmuladd.f64(double %73, double -5.000000e-01, double %52)
  %76 = tail call double @llvm.fmuladd.f64(double %75, double 5.000000e-01, double %73)
  %77 = tail call double @llvm.fmuladd.f64(double %75, double -5.000000e-01, double %51)
  %78 = tail call double @llvm.fmuladd.f64(double %77, double 5.000000e-01, double %75)
  %79 = tail call double @llvm.fmuladd.f64(double %77, double -5.000000e-01, double %50)
  %80 = tail call double @llvm.fmuladd.f64(double %79, double 5.000000e-01, double %77)
  %81 = tail call double @llvm.fmuladd.f64(double %79, double -5.000000e-01, double %49)
  %82 = tail call double @llvm.fmuladd.f64(double %81, double 5.000000e-01, double %79)
  %83 = tail call double @llvm.fmuladd.f64(double %81, double -5.000000e-01, double %48)
  %84 = tail call double @llvm.fmuladd.f64(double %83, double 5.000000e-01, double %81)
  %85 = tail call double @llvm.fmuladd.f64(double %83, double -5.000000e-01, double %47)
  %86 = tail call double @llvm.fmuladd.f64(double %85, double 5.000000e-01, double %83)
  %87 = tail call double @llvm.fmuladd.f64(double %85, double -5.000000e-01, double %46)
  %88 = tail call double @llvm.fmuladd.f64(double %87, double 5.000000e-01, double %85)
  %89 = tail call double @llvm.fmuladd.f64(double %87, double -5.000000e-01, double %45)
  %90 = tail call double @llvm.fmuladd.f64(double %89, double 5.000000e-01, double %87)
  %91 = tail call double @llvm.fmuladd.f64(double %89, double -5.000000e-01, double %44)
  %92 = tail call double @llvm.fmuladd.f64(double %91, double 5.000000e-01, double %89)
  %93 = fptrunc double %91 to float
  %94 = getelementptr inbounds nuw float, ptr %2, i64 %60
  store float %93, ptr %94, align 4, !tbaa !10
  %95 = add nuw nsw i64 %60, 1
  %96 = icmp eq i64 %95, %8
  br i1 %96, label %40, label %43, !llvm.loop !12
}

; Function Attrs: mustprogress nocallback nofree nosync nounwind speculatable willreturn memory(none)
declare double @llvm.fmuladd.f64(double, double, double) #1

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @phaser_inline_uncond(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #2 {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %42

6:                                                ; preds = %4
  %7 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !14, !noalias !17
  %8 = zext nneg i32 %3 to i64
  %9 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %10 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %11 = load double, ptr %10, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %13 = load double, ptr %12, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %15 = load double, ptr %14, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %16 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %17 = load double, ptr %16, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %18 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %19 = load double, ptr %18, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %20 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %21 = load double, ptr %20, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %22 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %23 = load double, ptr %22, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %24 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %25 = load double, ptr %24, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %26 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %27 = load double, ptr %26, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %28 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %29 = load double, ptr %28, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %30 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %31 = load double, ptr %30, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %32 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %33 = load double, ptr %32, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %34 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %35 = load double, ptr %34, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %36 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %37 = load double, ptr %36, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %38 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %39 = load double, ptr %38, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  br label %43

40:                                               ; preds = %43
  store double %62, ptr %1, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %64, ptr %10, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %66, ptr %12, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %68, ptr %14, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %70, ptr %16, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %72, ptr %18, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %74, ptr %20, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %76, ptr %22, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %78, ptr %24, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %80, ptr %26, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %82, ptr %28, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %84, ptr %30, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %86, ptr %32, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %88, ptr %34, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %90, ptr %36, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  store double %92, ptr %38, align 8, !tbaa !6, !alias.scope !17, !noalias !14
  %41 = getelementptr inbounds nuw i8, ptr %0, i64 8
  store double %91, ptr %41, align 8, !tbaa !6, !alias.scope !14, !noalias !17
  br label %42

42:                                               ; preds = %40, %4
  ret void

43:                                               ; preds = %6, %43
  %44 = phi double [ %39, %6 ], [ %92, %43 ]
  %45 = phi double [ %37, %6 ], [ %90, %43 ]
  %46 = phi double [ %35, %6 ], [ %88, %43 ]
  %47 = phi double [ %33, %6 ], [ %86, %43 ]
  %48 = phi double [ %31, %6 ], [ %84, %43 ]
  %49 = phi double [ %29, %6 ], [ %82, %43 ]
  %50 = phi double [ %27, %6 ], [ %80, %43 ]
  %51 = phi double [ %25, %6 ], [ %78, %43 ]
  %52 = phi double [ %23, %6 ], [ %76, %43 ]
  %53 = phi double [ %21, %6 ], [ %74, %43 ]
  %54 = phi double [ %19, %6 ], [ %72, %43 ]
  %55 = phi double [ %17, %6 ], [ %70, %43 ]
  %56 = phi double [ %15, %6 ], [ %68, %43 ]
  %57 = phi double [ %13, %6 ], [ %66, %43 ]
  %58 = phi double [ %11, %6 ], [ %64, %43 ]
  %59 = phi double [ %9, %6 ], [ %62, %43 ]
  %60 = phi i64 [ 0, %6 ], [ %95, %43 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !14)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !17)
  %61 = tail call double @llvm.fmuladd.f64(double %7, double -5.000000e-01, double %59)
  %62 = tail call double @llvm.fmuladd.f64(double %61, double 5.000000e-01, double %7)
  %63 = tail call double @llvm.fmuladd.f64(double %61, double -5.000000e-01, double %58)
  %64 = tail call double @llvm.fmuladd.f64(double %63, double 5.000000e-01, double %61)
  %65 = tail call double @llvm.fmuladd.f64(double %63, double -5.000000e-01, double %57)
  %66 = tail call double @llvm.fmuladd.f64(double %65, double 5.000000e-01, double %63)
  %67 = tail call double @llvm.fmuladd.f64(double %65, double -5.000000e-01, double %56)
  %68 = tail call double @llvm.fmuladd.f64(double %67, double 5.000000e-01, double %65)
  %69 = tail call double @llvm.fmuladd.f64(double %67, double -5.000000e-01, double %55)
  %70 = tail call double @llvm.fmuladd.f64(double %69, double 5.000000e-01, double %67)
  %71 = tail call double @llvm.fmuladd.f64(double %69, double -5.000000e-01, double %54)
  %72 = tail call double @llvm.fmuladd.f64(double %71, double 5.000000e-01, double %69)
  %73 = tail call double @llvm.fmuladd.f64(double %71, double -5.000000e-01, double %53)
  %74 = tail call double @llvm.fmuladd.f64(double %73, double 5.000000e-01, double %71)
  %75 = tail call double @llvm.fmuladd.f64(double %73, double -5.000000e-01, double %52)
  %76 = tail call double @llvm.fmuladd.f64(double %75, double 5.000000e-01, double %73)
  %77 = tail call double @llvm.fmuladd.f64(double %75, double -5.000000e-01, double %51)
  %78 = tail call double @llvm.fmuladd.f64(double %77, double 5.000000e-01, double %75)
  %79 = tail call double @llvm.fmuladd.f64(double %77, double -5.000000e-01, double %50)
  %80 = tail call double @llvm.fmuladd.f64(double %79, double 5.000000e-01, double %77)
  %81 = tail call double @llvm.fmuladd.f64(double %79, double -5.000000e-01, double %49)
  %82 = tail call double @llvm.fmuladd.f64(double %81, double 5.000000e-01, double %79)
  %83 = tail call double @llvm.fmuladd.f64(double %81, double -5.000000e-01, double %48)
  %84 = tail call double @llvm.fmuladd.f64(double %83, double 5.000000e-01, double %81)
  %85 = tail call double @llvm.fmuladd.f64(double %83, double -5.000000e-01, double %47)
  %86 = tail call double @llvm.fmuladd.f64(double %85, double 5.000000e-01, double %83)
  %87 = tail call double @llvm.fmuladd.f64(double %85, double -5.000000e-01, double %46)
  %88 = tail call double @llvm.fmuladd.f64(double %87, double 5.000000e-01, double %85)
  %89 = tail call double @llvm.fmuladd.f64(double %87, double -5.000000e-01, double %45)
  %90 = tail call double @llvm.fmuladd.f64(double %89, double 5.000000e-01, double %87)
  %91 = tail call double @llvm.fmuladd.f64(double %89, double -5.000000e-01, double %44)
  %92 = tail call double @llvm.fmuladd.f64(double %91, double 5.000000e-01, double %89)
  %93 = fptrunc double %91 to float
  %94 = getelementptr inbounds nuw float, ptr %2, i64 %60
  store float %93, ptr %94, align 4, !tbaa !10
  %95 = add nuw nsw i64 %60, 1
  %96 = icmp eq i64 %95, %8
  br i1 %96, label %40, label %43, !llvm.loop !19
}

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @phaser_inline_cond(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #2 {
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
  tail call void @llvm.experimental.noalias.scope.decl(metadata !20)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !23)
  %33 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !20, !noalias !23
  %34 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %35 = tail call double @llvm.fmuladd.f64(double %33, double -5.000000e-01, double %34)
  %36 = tail call double @llvm.fmuladd.f64(double %35, double 5.000000e-01, double %33)
  store double %36, ptr %1, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %37 = load double, ptr %13, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %38 = tail call double @llvm.fmuladd.f64(double %35, double -5.000000e-01, double %37)
  %39 = tail call double @llvm.fmuladd.f64(double %38, double 5.000000e-01, double %35)
  store double %39, ptr %13, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %40 = load double, ptr %14, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %41 = tail call double @llvm.fmuladd.f64(double %38, double -5.000000e-01, double %40)
  %42 = tail call double @llvm.fmuladd.f64(double %41, double 5.000000e-01, double %38)
  store double %42, ptr %14, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %43 = load double, ptr %15, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %44 = tail call double @llvm.fmuladd.f64(double %41, double -5.000000e-01, double %43)
  %45 = tail call double @llvm.fmuladd.f64(double %44, double 5.000000e-01, double %41)
  store double %45, ptr %15, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %46 = load double, ptr %16, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %47 = tail call double @llvm.fmuladd.f64(double %44, double -5.000000e-01, double %46)
  %48 = tail call double @llvm.fmuladd.f64(double %47, double 5.000000e-01, double %44)
  store double %48, ptr %16, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %49 = load double, ptr %17, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %50 = tail call double @llvm.fmuladd.f64(double %47, double -5.000000e-01, double %49)
  %51 = tail call double @llvm.fmuladd.f64(double %50, double 5.000000e-01, double %47)
  store double %51, ptr %17, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %52 = load double, ptr %18, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %53 = tail call double @llvm.fmuladd.f64(double %50, double -5.000000e-01, double %52)
  %54 = tail call double @llvm.fmuladd.f64(double %53, double 5.000000e-01, double %50)
  store double %54, ptr %18, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %55 = load double, ptr %19, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %56 = tail call double @llvm.fmuladd.f64(double %53, double -5.000000e-01, double %55)
  %57 = tail call double @llvm.fmuladd.f64(double %56, double 5.000000e-01, double %53)
  store double %57, ptr %19, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %58 = load double, ptr %20, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %59 = tail call double @llvm.fmuladd.f64(double %56, double -5.000000e-01, double %58)
  %60 = tail call double @llvm.fmuladd.f64(double %59, double 5.000000e-01, double %56)
  store double %60, ptr %20, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %61 = load double, ptr %21, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %62 = tail call double @llvm.fmuladd.f64(double %59, double -5.000000e-01, double %61)
  %63 = tail call double @llvm.fmuladd.f64(double %62, double 5.000000e-01, double %59)
  store double %63, ptr %21, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %64 = load double, ptr %22, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %65 = tail call double @llvm.fmuladd.f64(double %62, double -5.000000e-01, double %64)
  %66 = tail call double @llvm.fmuladd.f64(double %65, double 5.000000e-01, double %62)
  store double %66, ptr %22, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %67 = load double, ptr %23, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %68 = tail call double @llvm.fmuladd.f64(double %65, double -5.000000e-01, double %67)
  %69 = tail call double @llvm.fmuladd.f64(double %68, double 5.000000e-01, double %65)
  store double %69, ptr %23, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %70 = load double, ptr %24, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %71 = tail call double @llvm.fmuladd.f64(double %68, double -5.000000e-01, double %70)
  %72 = tail call double @llvm.fmuladd.f64(double %71, double 5.000000e-01, double %68)
  store double %72, ptr %24, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %73 = load double, ptr %25, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %74 = tail call double @llvm.fmuladd.f64(double %71, double -5.000000e-01, double %73)
  %75 = tail call double @llvm.fmuladd.f64(double %74, double 5.000000e-01, double %71)
  store double %75, ptr %25, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %76 = load double, ptr %26, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %77 = tail call double @llvm.fmuladd.f64(double %74, double -5.000000e-01, double %76)
  %78 = tail call double @llvm.fmuladd.f64(double %77, double 5.000000e-01, double %74)
  store double %78, ptr %26, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %79 = load double, ptr %27, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  %80 = tail call double @llvm.fmuladd.f64(double %77, double -5.000000e-01, double %79)
  %81 = tail call double @llvm.fmuladd.f64(double %80, double 5.000000e-01, double %77)
  store double %81, ptr %27, align 8, !tbaa !6, !alias.scope !23, !noalias !20
  store double %80, ptr %10, align 8, !tbaa !6, !alias.scope !20, !noalias !23
  br label %82

82:                                               ; preds = %32, %29
  %83 = phi double [ %80, %32 ], [ %31, %29 ]
  %84 = fptrunc double %83 to float
  %85 = getelementptr inbounds nuw float, ptr %2, i64 %30
  store float %84, ptr %85, align 4, !tbaa !10
  %86 = add nuw nsw i64 %30, 1
  %87 = icmp eq i64 %86, %12
  br i1 %87, label %28, label %29, !llvm.loop !25
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
!15 = distinct !{!15, !16, !"phaser16_b: argument 0"}
!16 = distinct !{!16, !"phaser16_b"}
!17 = !{!18}
!18 = distinct !{!18, !16, !"phaser16_b: argument 1"}
!19 = distinct !{!19, !13}
!20 = !{!21}
!21 = distinct !{!21, !22, !"phaser16_c: argument 0"}
!22 = distinct !{!22, !"phaser16_c"}
!23 = !{!24}
!24 = distinct !{!24, !22, !"phaser16_c: argument 1"}
!25 = distinct !{!25, !13}
