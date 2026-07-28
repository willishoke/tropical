define void @kernel(ptr %inputs, ptr %registers, ptr %arrays,
ptr %array_sizes, ptr %temps, double %sampleRate, i64 %start_sample_index,
ptr %param_ptrs, ptr %output_buffer, i64 %buffer_length, ptr %slots) {
entry:
  br label %loop.cond
loop.cond:
  %s = phi i64 [0, %entry], [%next, %loop.body]
  %more = icmp ult i64 %s, %buffer_length
  br i1 %more, label %loop.body, label %done
loop.body:
  %value = load double, ptr %slots, align 8
  %dst = getelementptr inbounds double, ptr %output_buffer, i64 %s
  store double %value, ptr %dst, align 8
  %next = add i64 %s, 1
  br label %loop.cond
done:
  ret void
}
