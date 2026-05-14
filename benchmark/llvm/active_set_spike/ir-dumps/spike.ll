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
  br i1 %5, label %6, label %99

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
  br i1 %13, label %16, label %48

16:                                               ; preds = %6
  %17 = load double, ptr %10, align 8, !tbaa !6, !noalias !24
  %18 = load double, ptr %14, align 8, !tbaa !6, !alias.scope !25, !noalias !28
  br i1 %9, label %21, label %19

19:                                               ; preds = %16
  %20 = zext nneg i32 %3 to i64
  br label %37

21:                                               ; preds = %16
  %22 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !30, !noalias !33
  %23 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !33, !noalias !30
  %24 = zext nneg i32 %3 to i64
  br label %25

25:                                               ; preds = %25, %21
  %26 = phi i64 [ %34, %25 ], [ 0, %21 ]
  %27 = phi double [ %29, %25 ], [ %23, %21 ]
  %28 = phi double [ %31, %25 ], [ %18, %21 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !30)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !33)
  %29 = tail call double @llvm.fmuladd.f64(double %22, double 1.000000e-04, double %27)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !28)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !25)
  %30 = fmul double %28, 5.000000e-01
  %31 = tail call double @llvm.fmuladd.f64(double %27, double 5.000000e-01, double %30)
  %32 = fptrunc double %31 to float
  %33 = getelementptr inbounds nuw float, ptr %2, i64 %26
  store float %32, ptr %33, align 4, !tbaa !20
  %34 = add nuw nsw i64 %26, 1
  %35 = icmp eq i64 %34, %24
  br i1 %35, label %36, label %25, !llvm.loop !35

36:                                               ; preds = %25
  store double %29, ptr %1, align 8, !tbaa !6, !alias.scope !33, !noalias !30
  store double %27, ptr %10, align 8, !tbaa !6, !alias.scope !30, !noalias !33
  br label %46

37:                                               ; preds = %19, %37
  %38 = phi i64 [ 0, %19 ], [ %44, %37 ]
  %39 = phi double [ %18, %19 ], [ %41, %37 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !28)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !25)
  %40 = fmul double %39, 5.000000e-01
  %41 = tail call double @llvm.fmuladd.f64(double %17, double 5.000000e-01, double %40)
  %42 = fptrunc double %41 to float
  %43 = getelementptr inbounds nuw float, ptr %2, i64 %38
  store float %42, ptr %43, align 4, !tbaa !20
  %44 = add nuw nsw i64 %38, 1
  %45 = icmp eq i64 %44, %20
  br i1 %45, label %46, label %37, !llvm.loop !35

46:                                               ; preds = %37, %36
  %47 = phi double [ %31, %36 ], [ %41, %37 ]
  store double %47, ptr %14, align 8, !tbaa !6, !alias.scope !25, !noalias !28
  store double %47, ptr %15, align 8, !tbaa !6, !alias.scope !28, !noalias !25
  br label %99

48:                                               ; preds = %6
  %49 = load double, ptr %15, align 8, !tbaa !6
  %50 = fptrunc double %49 to float
  br i1 %9, label %87, label %51

51:                                               ; preds = %48
  %52 = zext nneg i32 %3 to i64
  %53 = icmp ult i32 %3, 4
  br i1 %53, label %54, label %56

54:                                               ; preds = %72, %85, %51
  %55 = phi i64 [ %59, %72 ], [ 0, %51 ], [ %77, %85 ]
  br label %100

56:                                               ; preds = %51
  %57 = icmp ult i32 %3, 16
  br i1 %57, label %75, label %58

58:                                               ; preds = %56
  %59 = and i64 %52, 2147483632
  %60 = insertelement <4 x float> poison, float %50, i64 0
  %61 = shufflevector <4 x float> %60, <4 x float> poison, <4 x i32> zeroinitializer
  br label %62

62:                                               ; preds = %62, %58
  %63 = phi i64 [ 0, %58 ], [ %68, %62 ]
  %64 = getelementptr inbounds nuw float, ptr %2, i64 %63
  %65 = getelementptr inbounds nuw i8, ptr %64, i64 16
  %66 = getelementptr inbounds nuw i8, ptr %64, i64 32
  %67 = getelementptr inbounds nuw i8, ptr %64, i64 48
  store <4 x float> %61, ptr %64, align 4, !tbaa !20
  store <4 x float> %61, ptr %65, align 4, !tbaa !20
  store <4 x float> %61, ptr %66, align 4, !tbaa !20
  store <4 x float> %61, ptr %67, align 4, !tbaa !20
  %68 = add nuw i64 %63, 16
  %69 = icmp eq i64 %68, %59
  br i1 %69, label %70, label %62, !llvm.loop !36

70:                                               ; preds = %62
  %71 = icmp eq i64 %59, %52
  br i1 %71, label %99, label %72

72:                                               ; preds = %70
  %73 = and i64 %52, 12
  %74 = icmp eq i64 %73, 0
  br i1 %74, label %54, label %75

75:                                               ; preds = %72, %56
  %76 = phi i64 [ %59, %72 ], [ 0, %56 ]
  %77 = and i64 %52, 2147483644
  %78 = insertelement <4 x float> poison, float %50, i64 0
  %79 = shufflevector <4 x float> %78, <4 x float> poison, <4 x i32> zeroinitializer
  br label %80

80:                                               ; preds = %80, %75
  %81 = phi i64 [ %76, %75 ], [ %83, %80 ]
  %82 = getelementptr inbounds nuw float, ptr %2, i64 %81
  store <4 x float> %79, ptr %82, align 4, !tbaa !20
  %83 = add nuw i64 %81, 4
  %84 = icmp eq i64 %83, %77
  br i1 %84, label %85, label %80, !llvm.loop !39

85:                                               ; preds = %80
  %86 = icmp eq i64 %77, %52
  br i1 %86, label %99, label %54

87:                                               ; preds = %48
  %88 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !30, !noalias !33
  %89 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !33, !noalias !30
  %90 = zext nneg i32 %3 to i64
  br label %91

91:                                               ; preds = %91, %87
  %92 = phi i64 [ %96, %91 ], [ 0, %87 ]
  %93 = phi double [ %94, %91 ], [ %89, %87 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !30)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !33)
  %94 = tail call double @llvm.fmuladd.f64(double %88, double 1.000000e-04, double %93)
  %95 = getelementptr inbounds nuw float, ptr %2, i64 %92
  store float %50, ptr %95, align 4, !tbaa !20
  %96 = add nuw nsw i64 %92, 1
  %97 = icmp eq i64 %96, %90
  br i1 %97, label %98, label %91, !llvm.loop !35

98:                                               ; preds = %91
  store double %94, ptr %1, align 8, !tbaa !6, !alias.scope !33, !noalias !30
  store double %93, ptr %10, align 8, !tbaa !6, !alias.scope !30, !noalias !33
  br label %99

99:                                               ; preds = %100, %70, %85, %46, %98, %4
  ret void

100:                                              ; preds = %54, %100
  %101 = phi i64 [ %103, %100 ], [ %55, %54 ]
  %102 = getelementptr inbounds nuw float, ptr %2, i64 %101
  store float %50, ptr %102, align 4, !tbaa !20
  %103 = add nuw nsw i64 %101, 1
  %104 = icmp eq i64 %103, %52
  br i1 %104, label %99, label %100, !llvm.loop !40
}

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @scheduler_C_per_block_alive(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = getelementptr inbounds nuw i8, ptr %0, i64 24
  %6 = load double, ptr %5, align 8, !tbaa !6
  %7 = fcmp ogt double %6, 5.000000e-01
  %8 = icmp sgt i32 %3, 0
  br i1 %8, label %9, label %99

9:                                                ; preds = %4
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %11 = load double, ptr %10, align 8, !tbaa !6
  %12 = fcmp ogt double %11, 5.000000e-01
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %14 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %15 = getelementptr inbounds nuw i8, ptr %0, i64 16
  br i1 %12, label %16, label %48

16:                                               ; preds = %9
  %17 = load double, ptr %13, align 8, !tbaa !6, !noalias !24
  %18 = load double, ptr %14, align 8, !tbaa !6, !alias.scope !41, !noalias !44
  br i1 %7, label %21, label %19

19:                                               ; preds = %16
  %20 = zext nneg i32 %3 to i64
  br label %37

21:                                               ; preds = %16
  %22 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !46, !noalias !49
  %23 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !49, !noalias !46
  %24 = zext nneg i32 %3 to i64
  br label %25

25:                                               ; preds = %25, %21
  %26 = phi i64 [ %34, %25 ], [ 0, %21 ]
  %27 = phi double [ %29, %25 ], [ %23, %21 ]
  %28 = phi double [ %31, %25 ], [ %18, %21 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !46)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !49)
  %29 = tail call double @llvm.fmuladd.f64(double %22, double 1.000000e-04, double %27)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !44)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !41)
  %30 = fmul double %28, 5.000000e-01
  %31 = tail call double @llvm.fmuladd.f64(double %27, double 5.000000e-01, double %30)
  %32 = fptrunc double %31 to float
  %33 = getelementptr inbounds nuw float, ptr %2, i64 %26
  store float %32, ptr %33, align 4, !tbaa !20
  %34 = add nuw nsw i64 %26, 1
  %35 = icmp eq i64 %34, %24
  br i1 %35, label %36, label %25, !llvm.loop !51

36:                                               ; preds = %25
  store double %29, ptr %1, align 8, !tbaa !6, !alias.scope !49, !noalias !46
  store double %27, ptr %13, align 8, !tbaa !6, !alias.scope !46, !noalias !49
  br label %46

37:                                               ; preds = %19, %37
  %38 = phi i64 [ 0, %19 ], [ %44, %37 ]
  %39 = phi double [ %18, %19 ], [ %41, %37 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !44)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !41)
  %40 = fmul double %39, 5.000000e-01
  %41 = tail call double @llvm.fmuladd.f64(double %17, double 5.000000e-01, double %40)
  %42 = fptrunc double %41 to float
  %43 = getelementptr inbounds nuw float, ptr %2, i64 %38
  store float %42, ptr %43, align 4, !tbaa !20
  %44 = add nuw nsw i64 %38, 1
  %45 = icmp eq i64 %44, %20
  br i1 %45, label %46, label %37, !llvm.loop !51

46:                                               ; preds = %37, %36
  %47 = phi double [ %31, %36 ], [ %41, %37 ]
  store double %47, ptr %14, align 8, !tbaa !6, !alias.scope !41, !noalias !44
  store double %47, ptr %15, align 8, !tbaa !6, !alias.scope !44, !noalias !41
  br label %99

48:                                               ; preds = %9
  %49 = load double, ptr %15, align 8, !tbaa !6
  %50 = fptrunc double %49 to float
  br i1 %7, label %87, label %51

51:                                               ; preds = %48
  %52 = zext nneg i32 %3 to i64
  %53 = icmp ult i32 %3, 4
  br i1 %53, label %54, label %56

54:                                               ; preds = %72, %85, %51
  %55 = phi i64 [ %59, %72 ], [ 0, %51 ], [ %77, %85 ]
  br label %100

56:                                               ; preds = %51
  %57 = icmp ult i32 %3, 16
  br i1 %57, label %75, label %58

58:                                               ; preds = %56
  %59 = and i64 %52, 2147483632
  %60 = insertelement <4 x float> poison, float %50, i64 0
  %61 = shufflevector <4 x float> %60, <4 x float> poison, <4 x i32> zeroinitializer
  br label %62

62:                                               ; preds = %62, %58
  %63 = phi i64 [ 0, %58 ], [ %68, %62 ]
  %64 = getelementptr inbounds nuw float, ptr %2, i64 %63
  %65 = getelementptr inbounds nuw i8, ptr %64, i64 16
  %66 = getelementptr inbounds nuw i8, ptr %64, i64 32
  %67 = getelementptr inbounds nuw i8, ptr %64, i64 48
  store <4 x float> %61, ptr %64, align 4, !tbaa !20
  store <4 x float> %61, ptr %65, align 4, !tbaa !20
  store <4 x float> %61, ptr %66, align 4, !tbaa !20
  store <4 x float> %61, ptr %67, align 4, !tbaa !20
  %68 = add nuw i64 %63, 16
  %69 = icmp eq i64 %68, %59
  br i1 %69, label %70, label %62, !llvm.loop !52

70:                                               ; preds = %62
  %71 = icmp eq i64 %59, %52
  br i1 %71, label %99, label %72

72:                                               ; preds = %70
  %73 = and i64 %52, 12
  %74 = icmp eq i64 %73, 0
  br i1 %74, label %54, label %75

75:                                               ; preds = %72, %56
  %76 = phi i64 [ %59, %72 ], [ 0, %56 ]
  %77 = and i64 %52, 2147483644
  %78 = insertelement <4 x float> poison, float %50, i64 0
  %79 = shufflevector <4 x float> %78, <4 x float> poison, <4 x i32> zeroinitializer
  br label %80

80:                                               ; preds = %80, %75
  %81 = phi i64 [ %76, %75 ], [ %83, %80 ]
  %82 = getelementptr inbounds nuw float, ptr %2, i64 %81
  store <4 x float> %79, ptr %82, align 4, !tbaa !20
  %83 = add nuw i64 %81, 4
  %84 = icmp eq i64 %83, %77
  br i1 %84, label %85, label %80, !llvm.loop !53

85:                                               ; preds = %80
  %86 = icmp eq i64 %77, %52
  br i1 %86, label %99, label %54

87:                                               ; preds = %48
  %88 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !46, !noalias !49
  %89 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !49, !noalias !46
  %90 = zext nneg i32 %3 to i64
  br label %91

91:                                               ; preds = %91, %87
  %92 = phi i64 [ %96, %91 ], [ 0, %87 ]
  %93 = phi double [ %94, %91 ], [ %89, %87 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !46)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !49)
  %94 = tail call double @llvm.fmuladd.f64(double %88, double 1.000000e-04, double %93)
  %95 = getelementptr inbounds nuw float, ptr %2, i64 %92
  store float %50, ptr %95, align 4, !tbaa !20
  %96 = add nuw nsw i64 %92, 1
  %97 = icmp eq i64 %96, %90
  br i1 %97, label %98, label %91, !llvm.loop !51

98:                                               ; preds = %91
  store double %94, ptr %1, align 8, !tbaa !6, !alias.scope !49, !noalias !46
  store double %93, ptr %13, align 8, !tbaa !6, !alias.scope !46, !noalias !49
  br label %99

99:                                               ; preds = %100, %70, %85, %46, %98, %4
  ret void

100:                                              ; preds = %54, %100
  %101 = phi i64 [ %103, %100 ], [ %55, %54 ]
  %102 = getelementptr inbounds nuw float, ptr %2, i64 %101
  store float %50, ptr %102, align 4, !tbaa !20
  %103 = add nuw nsw i64 %101, 1
  %104 = icmp eq i64 %103, %52
  br i1 %104, label %99, label %100, !llvm.loop !54
}

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @scheduler_D_mixed(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %37

6:                                                ; preds = %4
  %7 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !55, !noalias !58
  %8 = getelementptr inbounds nuw i8, ptr %0, i64 8
  %9 = getelementptr inbounds nuw i8, ptr %0, i64 32
  %10 = load double, ptr %9, align 8, !tbaa !6
  %11 = fcmp ogt double %10, 5.000000e-01
  %12 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %13 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %14 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !58, !noalias !55
  br i1 %11, label %15, label %30

15:                                               ; preds = %6
  %16 = load double, ptr %12, align 8, !tbaa !6, !alias.scope !60, !noalias !63
  %17 = zext nneg i32 %3 to i64
  br label %18

18:                                               ; preds = %18, %15
  %19 = phi i64 [ %27, %18 ], [ 0, %15 ]
  %20 = phi double [ %24, %18 ], [ %16, %15 ]
  %21 = phi double [ %22, %18 ], [ %14, %15 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !55)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !58)
  %22 = tail call double @llvm.fmuladd.f64(double %7, double 1.000000e-04, double %21)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !63)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !60)
  %23 = fmul double %20, 5.000000e-01
  %24 = tail call double @llvm.fmuladd.f64(double %21, double 5.000000e-01, double %23)
  %25 = fptrunc double %24 to float
  %26 = getelementptr inbounds nuw float, ptr %2, i64 %19
  store float %25, ptr %26, align 4, !tbaa !20
  %27 = add nuw nsw i64 %19, 1
  %28 = icmp eq i64 %27, %17
  br i1 %28, label %29, label %18, !llvm.loop !65

29:                                               ; preds = %18
  store double %24, ptr %12, align 8, !tbaa !6, !alias.scope !60, !noalias !63
  store double %24, ptr %13, align 8, !tbaa !6, !alias.scope !63, !noalias !60
  br label %34

30:                                               ; preds = %6
  %31 = load double, ptr %13, align 8, !tbaa !6
  %32 = fptrunc double %31 to float
  %33 = zext nneg i32 %3 to i64
  br label %38

34:                                               ; preds = %38, %29
  %35 = phi double [ %21, %29 ], [ %40, %38 ]
  %36 = phi double [ %22, %29 ], [ %41, %38 ]
  store double %36, ptr %1, align 8, !tbaa !6, !alias.scope !58, !noalias !55
  store double %35, ptr %8, align 8, !tbaa !6, !alias.scope !55, !noalias !58
  br label %37

37:                                               ; preds = %34, %4
  ret void

38:                                               ; preds = %30, %38
  %39 = phi i64 [ 0, %30 ], [ %43, %38 ]
  %40 = phi double [ %14, %30 ], [ %41, %38 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !55)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !58)
  %41 = tail call double @llvm.fmuladd.f64(double %7, double 1.000000e-04, double %40)
  %42 = getelementptr inbounds nuw float, ptr %2, i64 %39
  store float %32, ptr %42, align 4, !tbaa !20
  %43 = add nuw nsw i64 %39, 1
  %44 = icmp eq i64 %43, %33
  br i1 %44, label %34, label %38, !llvm.loop !65
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
!24 = !{}
!25 = !{!26}
!26 = distinct !{!26, !27, !"filter_B: argument 1"}
!27 = distinct !{!27, !"filter_B"}
!28 = !{!29}
!29 = distinct !{!29, !27, !"filter_B: argument 0"}
!30 = !{!31}
!31 = distinct !{!31, !32, !"osc_B: argument 0"}
!32 = distinct !{!32, !"osc_B"}
!33 = !{!34}
!34 = distinct !{!34, !32, !"osc_B: argument 1"}
!35 = distinct !{!35, !23}
!36 = distinct !{!36, !23, !37, !38}
!37 = !{!"llvm.loop.isvectorized", i32 1}
!38 = !{!"llvm.loop.unroll.runtime.disable"}
!39 = distinct !{!39, !23, !37, !38}
!40 = distinct !{!40, !23, !38, !37}
!41 = !{!42}
!42 = distinct !{!42, !43, !"filter_C: argument 1"}
!43 = distinct !{!43, !"filter_C"}
!44 = !{!45}
!45 = distinct !{!45, !43, !"filter_C: argument 0"}
!46 = !{!47}
!47 = distinct !{!47, !48, !"osc_C: argument 0"}
!48 = distinct !{!48, !"osc_C"}
!49 = !{!50}
!50 = distinct !{!50, !48, !"osc_C: argument 1"}
!51 = distinct !{!51, !23}
!52 = distinct !{!52, !23, !37, !38}
!53 = distinct !{!53, !23, !37, !38}
!54 = distinct !{!54, !23, !38, !37}
!55 = !{!56}
!56 = distinct !{!56, !57, !"osc_D: argument 0"}
!57 = distinct !{!57, !"osc_D"}
!58 = !{!59}
!59 = distinct !{!59, !57, !"osc_D: argument 1"}
!60 = !{!61}
!61 = distinct !{!61, !62, !"filter_D: argument 1"}
!62 = distinct !{!62, !"filter_D"}
!63 = !{!64}
!64 = distinct !{!64, !62, !"filter_D: argument 0"}
!65 = distinct !{!65, !23}
