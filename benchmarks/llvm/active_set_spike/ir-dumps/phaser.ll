; ModuleID = 'phaser.c'
source_filename = "phaser.c"
target datalayout = "e-m:o-p270:32:32-p271:32:32-p272:64:64-i64:64-i128:128-n32:64-S128-Fn32"
target triple = "arm64-apple-macosx16.0.0"

; Function Attrs: nofree norecurse nosync nounwind ssp memory(argmem: readwrite, inaccessiblemem: readwrite) uwtable(sync)
define void @scheduler(ptr noalias nocapture noundef %0, ptr noalias nocapture noundef %1, ptr noalias nocapture noundef writeonly %2, i32 noundef %3) local_unnamed_addr #0 {
  %5 = icmp sgt i32 %3, 0
  br i1 %5, label %6, label %138

6:                                                ; preds = %4
  %7 = getelementptr inbounds nuw i8, ptr %0, i64 16
  %8 = load double, ptr %7, align 8, !tbaa !6
  %9 = fcmp ogt double %8, 5.000000e-01
  %10 = getelementptr inbounds nuw i8, ptr %0, i64 8
  br i1 %9, label %11, label %100

11:                                               ; preds = %6
  %12 = load double, ptr %0, align 8, !tbaa !6, !alias.scope !10, !noalias !13
  %13 = zext nneg i32 %3 to i64
  %14 = load double, ptr %1, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %15 = getelementptr inbounds nuw i8, ptr %1, i64 8
  %16 = load double, ptr %15, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %17 = getelementptr inbounds nuw i8, ptr %1, i64 16
  %18 = load double, ptr %17, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %19 = getelementptr inbounds nuw i8, ptr %1, i64 24
  %20 = load double, ptr %19, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %21 = getelementptr inbounds nuw i8, ptr %1, i64 32
  %22 = load double, ptr %21, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %23 = getelementptr inbounds nuw i8, ptr %1, i64 40
  %24 = load double, ptr %23, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %25 = getelementptr inbounds nuw i8, ptr %1, i64 48
  %26 = load double, ptr %25, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %27 = getelementptr inbounds nuw i8, ptr %1, i64 56
  %28 = load double, ptr %27, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %29 = getelementptr inbounds nuw i8, ptr %1, i64 64
  %30 = load double, ptr %29, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %31 = getelementptr inbounds nuw i8, ptr %1, i64 72
  %32 = load double, ptr %31, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %33 = getelementptr inbounds nuw i8, ptr %1, i64 80
  %34 = load double, ptr %33, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %35 = getelementptr inbounds nuw i8, ptr %1, i64 88
  %36 = load double, ptr %35, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %37 = getelementptr inbounds nuw i8, ptr %1, i64 96
  %38 = load double, ptr %37, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %39 = getelementptr inbounds nuw i8, ptr %1, i64 104
  %40 = load double, ptr %39, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %41 = getelementptr inbounds nuw i8, ptr %1, i64 112
  %42 = load double, ptr %41, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  %43 = getelementptr inbounds nuw i8, ptr %1, i64 120
  %44 = load double, ptr %43, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  br label %45

45:                                               ; preds = %45, %11
  %46 = phi double [ %94, %45 ], [ %44, %11 ]
  %47 = phi double [ %92, %45 ], [ %42, %11 ]
  %48 = phi double [ %90, %45 ], [ %40, %11 ]
  %49 = phi double [ %88, %45 ], [ %38, %11 ]
  %50 = phi double [ %86, %45 ], [ %36, %11 ]
  %51 = phi double [ %84, %45 ], [ %34, %11 ]
  %52 = phi double [ %82, %45 ], [ %32, %11 ]
  %53 = phi double [ %80, %45 ], [ %30, %11 ]
  %54 = phi double [ %78, %45 ], [ %28, %11 ]
  %55 = phi double [ %76, %45 ], [ %26, %11 ]
  %56 = phi double [ %74, %45 ], [ %24, %11 ]
  %57 = phi double [ %72, %45 ], [ %22, %11 ]
  %58 = phi double [ %70, %45 ], [ %20, %11 ]
  %59 = phi double [ %68, %45 ], [ %18, %11 ]
  %60 = phi double [ %66, %45 ], [ %16, %11 ]
  %61 = phi double [ %64, %45 ], [ %14, %11 ]
  %62 = phi i64 [ %97, %45 ], [ 0, %11 ]
  tail call void @llvm.experimental.noalias.scope.decl(metadata !10)
  tail call void @llvm.experimental.noalias.scope.decl(metadata !13)
  %63 = tail call double @llvm.fmuladd.f64(double %12, double -5.000000e-01, double %61)
  %64 = tail call double @llvm.fmuladd.f64(double %63, double 5.000000e-01, double %12)
  %65 = tail call double @llvm.fmuladd.f64(double %63, double -5.000000e-01, double %60)
  %66 = tail call double @llvm.fmuladd.f64(double %65, double 5.000000e-01, double %63)
  %67 = tail call double @llvm.fmuladd.f64(double %65, double -5.000000e-01, double %59)
  %68 = tail call double @llvm.fmuladd.f64(double %67, double 5.000000e-01, double %65)
  %69 = tail call double @llvm.fmuladd.f64(double %67, double -5.000000e-01, double %58)
  %70 = tail call double @llvm.fmuladd.f64(double %69, double 5.000000e-01, double %67)
  %71 = tail call double @llvm.fmuladd.f64(double %69, double -5.000000e-01, double %57)
  %72 = tail call double @llvm.fmuladd.f64(double %71, double 5.000000e-01, double %69)
  %73 = tail call double @llvm.fmuladd.f64(double %71, double -5.000000e-01, double %56)
  %74 = tail call double @llvm.fmuladd.f64(double %73, double 5.000000e-01, double %71)
  %75 = tail call double @llvm.fmuladd.f64(double %73, double -5.000000e-01, double %55)
  %76 = tail call double @llvm.fmuladd.f64(double %75, double 5.000000e-01, double %73)
  %77 = tail call double @llvm.fmuladd.f64(double %75, double -5.000000e-01, double %54)
  %78 = tail call double @llvm.fmuladd.f64(double %77, double 5.000000e-01, double %75)
  %79 = tail call double @llvm.fmuladd.f64(double %77, double -5.000000e-01, double %53)
  %80 = tail call double @llvm.fmuladd.f64(double %79, double 5.000000e-01, double %77)
  %81 = tail call double @llvm.fmuladd.f64(double %79, double -5.000000e-01, double %52)
  %82 = tail call double @llvm.fmuladd.f64(double %81, double 5.000000e-01, double %79)
  %83 = tail call double @llvm.fmuladd.f64(double %81, double -5.000000e-01, double %51)
  %84 = tail call double @llvm.fmuladd.f64(double %83, double 5.000000e-01, double %81)
  %85 = tail call double @llvm.fmuladd.f64(double %83, double -5.000000e-01, double %50)
  %86 = tail call double @llvm.fmuladd.f64(double %85, double 5.000000e-01, double %83)
  %87 = tail call double @llvm.fmuladd.f64(double %85, double -5.000000e-01, double %49)
  %88 = tail call double @llvm.fmuladd.f64(double %87, double 5.000000e-01, double %85)
  %89 = tail call double @llvm.fmuladd.f64(double %87, double -5.000000e-01, double %48)
  %90 = tail call double @llvm.fmuladd.f64(double %89, double 5.000000e-01, double %87)
  %91 = tail call double @llvm.fmuladd.f64(double %89, double -5.000000e-01, double %47)
  %92 = tail call double @llvm.fmuladd.f64(double %91, double 5.000000e-01, double %89)
  %93 = tail call double @llvm.fmuladd.f64(double %91, double -5.000000e-01, double %46)
  %94 = tail call double @llvm.fmuladd.f64(double %93, double 5.000000e-01, double %91)
  %95 = fptrunc double %93 to float
  %96 = getelementptr inbounds nuw float, ptr %2, i64 %62
  store float %95, ptr %96, align 4, !tbaa !15
  %97 = add nuw nsw i64 %62, 1
  %98 = icmp eq i64 %97, %13
  br i1 %98, label %99, label %45, !llvm.loop !17

99:                                               ; preds = %45
  store double %64, ptr %1, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %66, ptr %15, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %68, ptr %17, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %70, ptr %19, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %72, ptr %21, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %74, ptr %23, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %76, ptr %25, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %78, ptr %27, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %80, ptr %29, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %82, ptr %31, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %84, ptr %33, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %86, ptr %35, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %88, ptr %37, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %90, ptr %39, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %92, ptr %41, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %94, ptr %43, align 8, !tbaa !6, !alias.scope !13, !noalias !10
  store double %93, ptr %10, align 8, !tbaa !6, !alias.scope !10, !noalias !13
  br label %138

100:                                              ; preds = %6
  %101 = load double, ptr %10, align 8, !tbaa !6
  %102 = fptrunc double %101 to float
  %103 = zext nneg i32 %3 to i64
  %104 = icmp ult i32 %3, 4
  br i1 %104, label %105, label %107

105:                                              ; preds = %123, %136, %100
  %106 = phi i64 [ %110, %123 ], [ 0, %100 ], [ %128, %136 ]
  br label %139

107:                                              ; preds = %100
  %108 = icmp ult i32 %3, 16
  br i1 %108, label %126, label %109

109:                                              ; preds = %107
  %110 = and i64 %103, 2147483632
  %111 = insertelement <4 x float> poison, float %102, i64 0
  %112 = shufflevector <4 x float> %111, <4 x float> poison, <4 x i32> zeroinitializer
  br label %113

113:                                              ; preds = %113, %109
  %114 = phi i64 [ 0, %109 ], [ %119, %113 ]
  %115 = getelementptr inbounds nuw float, ptr %2, i64 %114
  %116 = getelementptr inbounds nuw i8, ptr %115, i64 16
  %117 = getelementptr inbounds nuw i8, ptr %115, i64 32
  %118 = getelementptr inbounds nuw i8, ptr %115, i64 48
  store <4 x float> %112, ptr %115, align 4, !tbaa !15
  store <4 x float> %112, ptr %116, align 4, !tbaa !15
  store <4 x float> %112, ptr %117, align 4, !tbaa !15
  store <4 x float> %112, ptr %118, align 4, !tbaa !15
  %119 = add nuw i64 %114, 16
  %120 = icmp eq i64 %119, %110
  br i1 %120, label %121, label %113, !llvm.loop !19

121:                                              ; preds = %113
  %122 = icmp eq i64 %110, %103
  br i1 %122, label %138, label %123

123:                                              ; preds = %121
  %124 = and i64 %103, 12
  %125 = icmp eq i64 %124, 0
  br i1 %125, label %105, label %126

126:                                              ; preds = %123, %107
  %127 = phi i64 [ %110, %123 ], [ 0, %107 ]
  %128 = and i64 %103, 2147483644
  %129 = insertelement <4 x float> poison, float %102, i64 0
  %130 = shufflevector <4 x float> %129, <4 x float> poison, <4 x i32> zeroinitializer
  br label %131

131:                                              ; preds = %131, %126
  %132 = phi i64 [ %127, %126 ], [ %134, %131 ]
  %133 = getelementptr inbounds nuw float, ptr %2, i64 %132
  store <4 x float> %130, ptr %133, align 4, !tbaa !15
  %134 = add nuw i64 %132, 4
  %135 = icmp eq i64 %134, %128
  br i1 %135, label %136, label %131, !llvm.loop !22

136:                                              ; preds = %131
  %137 = icmp eq i64 %128, %103
  br i1 %137, label %138, label %105

138:                                              ; preds = %139, %121, %136, %99, %4
  ret void

139:                                              ; preds = %105, %139
  %140 = phi i64 [ %142, %139 ], [ %106, %105 ]
  %141 = getelementptr inbounds nuw float, ptr %2, i64 %140
  store float %102, ptr %141, align 4, !tbaa !15
  %142 = add nuw nsw i64 %140, 1
  %143 = icmp eq i64 %142, %103
  br i1 %143, label %138, label %139, !llvm.loop !23
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
!19 = distinct !{!19, !18, !20, !21}
!20 = !{!"llvm.loop.isvectorized", i32 1}
!21 = !{!"llvm.loop.unroll.runtime.disable"}
!22 = distinct !{!22, !18, !20, !21}
!23 = distinct !{!23, !18, !21, !20}
