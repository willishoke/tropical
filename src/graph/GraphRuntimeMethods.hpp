#pragma once

void Graph::wait_for_runtime_available(uint32_t runtime_index) const
{
  while (audio_processing_.load(std::memory_order_acquire) &&
         audio_runtime_index_.load(std::memory_order_acquire) == runtime_index)
  {
    std::this_thread::yield();
  }
}

uint64_t Graph::estimate_module_execution_cost(const Module & module)
{
  uint64_t cost = static_cast<uint64_t>(module.program_.instructions.size()) +
                  static_cast<uint64_t>(module.composite_output_program_.instructions.size()) +
                  static_cast<uint64_t>(module.composite_register_program_.instructions.size()) +
                  static_cast<uint64_t>(module.delay_update_program_.instructions.size()) +
                  static_cast<uint64_t>(module.inputs.size()) +
                  static_cast<uint64_t>(module.outputs.size()) +
                  static_cast<uint64_t>(module.registers_.size());

  for (const auto & nested : module.nested_modules_)
  {
    if (nested.module)
    {
      cost += estimate_module_execution_cost(*nested.module);
      cost += static_cast<uint64_t>(nested.input_program.instructions.size());
    }
  }

  return std::max<uint64_t>(cost, 1);
}

Graph::RuntimeState Graph::build_runtime_locked() const
{
  RuntimeState runtime;
  runtime.modules.reserve(control_modules_.size());
  runtime.name_to_id.reserve(control_modules_.size());

  std::vector<const std::pair<const std::string, ControlModule> *> ordered_modules;
  ordered_modules.reserve(control_modules_.size());
  for (const auto & entry : control_modules_)
  {
    ordered_modules.push_back(&entry);
  }
  std::sort(
    ordered_modules.begin(),
    ordered_modules.end(),
    [](const auto * lhs, const auto * rhs)
    {
      const uint64_t lhs_cost =
        lhs->second.module ? estimate_module_execution_cost(*lhs->second.module) : uint64_t(0);
      const uint64_t rhs_cost =
        rhs->second.module ? estimate_module_execution_cost(*rhs->second.module) : uint64_t(0);
      if (lhs_cost != rhs_cost)
      {
        return lhs_cost > rhs_cost;
      }
      return lhs->first < rhs->first;
    });

  for (const auto * entry : ordered_modules)
  {
    const auto & name = entry->first;
    const auto & module = entry->second;
    const uint32_t module_id = static_cast<uint32_t>(runtime.modules.size());
    runtime.name_to_id.emplace(name, module_id);

    ModuleSlot slot;
    slot.name = name;
    slot.module = module.module;
    slot.input_program.result_registers.resize(module.in_count, 0);
    runtime.modules.push_back(std::move(slot));
  }

  for (const auto * entry : ordered_modules)
  {
    const auto & name = entry->first;
    const auto & module = entry->second;
    auto id_it = runtime.name_to_id.find(name);
    if (id_it == runtime.name_to_id.end())
    {
      continue;
    }

    ModuleSlot & slot = runtime.modules[id_it->second];
    const unsigned int input_count = static_cast<unsigned int>(slot.input_program.result_registers.size());
    slot.input_program = compile_input_program(module.input_exprs, input_count, runtime);
    slot.input_registers.assign(slot.input_program.register_count, float_value(0.0));
    slot.output_materialize_mask.assign(module.out_count, false);
    slot.output_prev_materialize_mask.assign(module.out_count, false);
    slot.indexed_output_indices.assign(module.out_count, {});
    slot.indexed_prev_output_values.assign(module.out_count, {});
  }

  // Collect all TriggerParam ControlParam pointers across modules (deduped).
  {
    std::unordered_set<egress_expr::ControlParam *> seen;
    for (const auto & slot : runtime.modules)
    {
      if (slot.module)
      {
        for (auto * p : slot.module->trigger_params())
        {
          if (seen.insert(p).second)
          {
            runtime.trigger_params.push_back(p);
          }
        }
      }
    }
  }

  runtime.mix.reserve(control_mix_.size());
  for (const auto & tap : control_mix_)
  {
    auto it = runtime.name_to_id.find(tap.first);
    if (it == runtime.name_to_id.end())
    {
      continue;
    }

    const uint32_t module_id = it->second;
    if (module_id >= runtime.modules.size() ||
        !runtime.modules[module_id].module ||
        tap.second >= runtime.modules[module_id].module->outputs.size())
    {
      continue;
    }

    runtime.mix.push_back(MixTap{module_id, tap.second});
  }

  runtime.mix_exprs.reserve(control_mix_exprs_.size());
  for (const auto & mix_expr_spec : control_mix_exprs_)
  {
    MixExpr mix_expr;
    ExprSpecPtr inlined = egress_expr_inline::inline_functions(mix_expr_spec);
    mix_expr.result_register = compile_expr_node(inlined, mix_expr.program, runtime);
    mix_expr.registers.assign(mix_expr.program.register_count, float_value(0.0));
    runtime.mix_exprs.push_back(std::move(mix_expr));
  }

  refresh_runtime_ref_metadata(runtime);

  runtime.taps.reserve(control_taps_.size());
  for (const auto & tap_spec : control_taps_)
  {
    egress_graph_detail::OutputTap tap;
    tap.buffer.buffers[0].assign(bufferLength_, 0.0);
    tap.buffer.buffers[1].assign(bufferLength_, 0.0);
    if (!tap_spec.active)
    {
      runtime.taps.push_back(std::move(tap));
      continue;
    }

    const auto it = runtime.name_to_id.find(tap_spec.output.first);
    if (it == runtime.name_to_id.end())
    {
      runtime.taps.push_back(std::move(tap));
      continue;
    }

    const uint32_t module_id = it->second;
    if (module_id >= runtime.modules.size() ||
        !runtime.modules[module_id].module ||
        tap_spec.output.second >= runtime.modules[module_id].module->outputs.size())
    {
      runtime.taps.push_back(std::move(tap));
      continue;
    }

    tap.module_id = module_id;
    tap.output_id = tap_spec.output.second;
    tap.valid = true;
    runtime.taps.push_back(std::move(tap));
  }

  rebuild_fused_graph_state(runtime);
  return runtime;
}

void Graph::rebuild_fused_graph_state(RuntimeState & runtime) const
{
  runtime.fused_graph = build_fused_graph_state(runtime);
}

void Graph::sync_fused_current_outputs(RuntimeState & runtime) const
{
  auto * fused = runtime.fused_graph.get();
  if (fused == nullptr)
  {
    return;
  }

#ifdef EGRESS_PROFILE
  const auto sync_start = std::chrono::steady_clock::now();
  uint64_t output_copy_count = 0;
#endif
  for (uint32_t module_id = 0; module_id < runtime.modules.size(); ++module_id)
  {
    const auto & slot = runtime.modules[module_id];
    if (!slot.module || module_id >= fused->module_output_spans.size())
    {
      continue;
    }

    const auto & span = fused->module_output_spans[module_id];
    for (uint32_t offset = 0; offset < span.output_count; ++offset)
    {
      const uint32_t source_slot = span.first_output_slot + offset;
      if (source_slot >= fused->current_outputs.size() ||
          offset >= slot.module->outputs.size())
      {
        continue;
      }
      if (offset < slot.output_materialize_mask.size() &&
          !slot.output_materialize_mask[offset])
      {
        continue;
      }
      fused->current_outputs[source_slot] = slot.module->outputs[offset];
#ifdef EGRESS_PROFILE
      ++output_copy_count;
#endif
    }
  }
#ifdef EGRESS_PROFILE
  record_fused_sync_profile(
    false,
    static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - sync_start).count()),
    output_copy_count);
#endif
}

void Graph::sync_fused_prev_outputs(RuntimeState & runtime) const
{
  auto * fused = runtime.fused_graph.get();
  if (fused == nullptr)
  {
    return;
  }

#ifdef EGRESS_PROFILE
  const auto sync_start = std::chrono::steady_clock::now();
  uint64_t output_copy_count = 0;
#endif
  for (uint32_t module_id = 0; module_id < runtime.modules.size(); ++module_id)
  {
    const auto & slot = runtime.modules[module_id];
    if (!slot.module || module_id >= fused->module_output_spans.size())
    {
      continue;
    }

    const auto & span = fused->module_output_spans[module_id];
    for (uint32_t offset = 0; offset < span.output_count; ++offset)
    {
      const uint32_t source_slot = span.first_output_slot + offset;
      if (source_slot >= fused->prev_outputs.size() ||
          offset >= slot.module->prev_outputs.size())
      {
        continue;
      }
      if (offset < slot.output_prev_materialize_mask.size() &&
          !slot.output_prev_materialize_mask[offset])
      {
        continue;
      }
      fused->prev_outputs[source_slot] = slot.module->prev_outputs[offset];
#ifdef EGRESS_PROFILE
      ++output_copy_count;
#endif
    }
  }
#ifdef EGRESS_PROFILE
  record_fused_sync_profile(
    true,
    static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - sync_start).count()),
    output_copy_count);
#endif
}

bool Graph::run_fused_input_kernel(RuntimeState & runtime, bool allow_primitive_body_inputs) const
{
#ifndef EGRESS_LLVM_ORC_JIT
  (void)runtime;
  return false;
#else
  auto * fused = runtime.fused_graph.get();
  const bool allow_without_fusion_flag =
    allow_primitive_body_inputs &&
    fused != nullptr &&
    fused->primitive_body_available &&
    fused->kernel != nullptr;
  if ((!fusion_enabled_ && !allow_without_fusion_flag) ||
      fused == nullptr ||
      !fused->input_kernel.available ||
      fused->input_kernel.kernel == nullptr)
  {
    return false;
  }

  for (std::size_t source_slot = 0; source_slot < fused->source_outputs.size(); ++source_slot)
  {
    const auto & source = fused->source_outputs[source_slot];
    if (source.module_id < runtime.modules.size())
    {
      const auto & slot = runtime.modules[source.module_id];
      double numeric_scalar = 0.0;
      if (slot.module != nullptr &&
          slot.module->try_get_numeric_scalar_output(source.output_id, true, numeric_scalar))
      {
        fused->input_kernel.scalar_inputs[source_slot] = numeric_scalar;
        continue;
      }
    }
    if (source_slot >= fused->prev_outputs.size())
    {
      continue;
    }
    const Value & value = fused->prev_outputs[source_slot];
    fused->input_kernel.scalar_inputs[source_slot] =
      (value.type == ValueType::Array || value.type == ValueType::Matrix) ? 0.0 : expr::to_float64(value);
  }

  for (const auto & [key, scalar_slot] : fused->input_kernel.indexed_prev_scalar_slots)
  {
    const uint32_t source_slot = static_cast<uint32_t>(key >> 32);
    const uint32_t cache_slot = static_cast<uint32_t>(key & 0xffffffffu);
    if (scalar_slot >= fused->input_kernel.scalar_inputs.size() ||
        source_slot >= fused->indexed_prev_values.size() ||
        cache_slot >= fused->indexed_prev_values[source_slot].size())
    {
      continue;
    }
    fused->input_kernel.scalar_inputs[scalar_slot] =
      expr::to_float64(fused->indexed_prev_values[source_slot][cache_slot]);
  }

  for (const auto & [source_slot, array_slot] : fused->input_kernel.source_array_slots)
  {
    if (array_slot >= fused->input_kernel.array_storage.size())
    {
      continue;
    }
    auto & dst = fused->input_kernel.array_storage[array_slot];
    const auto & source = fused->source_outputs[source_slot];
    if (source.module_id < runtime.modules.size())
    {
      const auto & slot = runtime.modules[source.module_id];
      if (slot.module != nullptr)
      {
        const auto * numeric_values = slot.module->try_get_numeric_output_array_values(source.output_id, true);
        if (numeric_values != nullptr)
        {
          if (dst.size() != numeric_values->size())
          {
            dst.assign(numeric_values->size(), 0.0);
            fused->input_kernel.array_ptrs[array_slot] = dst.empty() ? nullptr : dst.data();
            fused->input_kernel.array_sizes[array_slot] = static_cast<uint64_t>(dst.size());
          }
          std::copy(numeric_values->begin(), numeric_values->end(), dst.begin());
          continue;
        }
      }
    }
    if (source_slot >= fused->prev_outputs.size())
    {
      continue;
    }
    const Value & value = fused->prev_outputs[source_slot];
    if (!expr::is_array(value))
    {
      continue;
    }
    if (dst.size() != value.array_items.size())
    {
      dst.assign(value.array_items.size(), 0.0);
      fused->input_kernel.array_ptrs[array_slot] = dst.empty() ? nullptr : dst.data();
      fused->input_kernel.array_sizes[array_slot] = static_cast<uint64_t>(dst.size());
    }
    for (std::size_t item_id = 0; item_id < value.array_items.size(); ++item_id)
    {
      dst[item_id] = expr::to_float64(value.array_items[item_id]);
    }
  }

  fused->input_kernel.kernel(
    fused->input_kernel.scalar_inputs.data(),
    nullptr,
    fused->input_kernel.array_ptrs.data(),
    fused->input_kernel.array_sizes.data(),
    fused->input_kernel.temps.data(),
    44100.0,
    0,
    fused->input_kernel.param_ptrs.data(),
    nullptr, nullptr, nullptr,
    fused->input_kernel.int_temps.empty() ? nullptr : fused->input_kernel.int_temps.data());

  for (const auto & binding : fused->input_kernel.input_bindings)
  {
    if (binding.module_id >= runtime.modules.size())
    {
      continue;
    }
    auto & slot = runtime.modules[binding.module_id];
    if (!slot.module || binding.input_id >= slot.module->inputs.size())
    {
      continue;
    }

    if (binding.value.kind == egress_graph_detail::FusedGraphValueKind::Scalar)
    {
      if (binding.value.scalar_register >= fused->input_kernel.temps.size())
      {
        continue;
      }
      const double scalar = fused->input_kernel.temps[binding.value.scalar_register];
      if (allow_primitive_body_inputs &&
          binding.fused_input_slot != std::numeric_limits<uint32_t>::max() &&
          binding.fused_input_slot < fused->inputs.size())
      {
        fused->inputs[binding.fused_input_slot] = scalar;
        continue;
      }
      slot.module->inputs[binding.input_id] =
        expr::float_value(scalar);
      continue;
    }

    if (binding.value.array_slot >= fused->input_kernel.array_storage.size())
    {
      continue;
    }
    const auto & src = fused->input_kernel.array_storage[binding.value.array_slot];
    if (allow_primitive_body_inputs &&
        binding.fused_input_array_slot != std::numeric_limits<uint32_t>::max() &&
        binding.fused_input_array_slot < fused->array_storage.size())
    {
      auto & dst = fused->array_storage[binding.fused_input_array_slot];
      dst = src;
      if (binding.fused_input_array_slot < fused->array_ptrs.size())
      {
        fused->array_ptrs[binding.fused_input_array_slot] = dst.empty() ? nullptr : dst.data();
        fused->array_sizes[binding.fused_input_array_slot] = static_cast<uint64_t>(dst.size());
      }
      continue;
    }
    std::vector<Value> items(src.size(), expr::float_value(0.0));
    for (std::size_t item_id = 0; item_id < src.size(); ++item_id)
    {
      items[item_id] = expr::float_value(src[item_id]);
    }
    slot.module->inputs[binding.input_id] = expr::array_value(std::move(items));
  }

  return true;
#endif
}

bool Graph::run_fused_primitive_body_kernel(RuntimeState & runtime) const
{
#ifndef EGRESS_LLVM_ORC_JIT
  (void)runtime;
  return false;
#else
  auto * fused = runtime.fused_graph.get();
  if (fused == nullptr ||
      !fused->primitive_body_available ||
      fused->kernel == nullptr)
  {
    return false;
  }
  auto fail = [&](const std::string & reason) -> bool
  {
    fused->primitive_body_status = reason;
    return false;
  };

  uint64_t sample_index = 0;
  bool have_sample_index = false;
  for (const auto & binding : fused->primitive_body_modules)
  {
    if (binding.module_id >= runtime.modules.size())
    {
      continue;
    }
    const auto & slot = runtime.modules[binding.module_id];
    if (!slot.module)
    {
      return fail("primitive body runtime missing module");
    }
    if (!have_sample_index)
    {
      sample_index = slot.module->sample_index_;
      have_sample_index = true;
      continue;
    }
    if (slot.module->sample_index_ != sample_index)
    {
      return fail("primitive body runtime sample index mismatch");
    }
  }

  fused->kernel(
    fused->inputs.empty() ? nullptr : fused->inputs.data(),
    fused->registers.empty() ? nullptr : fused->registers.data(),
    fused->array_ptrs.empty() ? nullptr : fused->array_ptrs.data(),
    fused->array_sizes.empty() ? nullptr : fused->array_sizes.data(),
    fused->temps.empty() ? nullptr : fused->temps.data(),
    fused->primitive_body_sample_rate,
    sample_index,
    fused->param_ptrs.data(),
    nullptr, nullptr, nullptr,
    fused->int_temps.empty() ? nullptr : fused->int_temps.data());

  for (const auto & binding : fused->primitive_body_modules)
  {
    if (binding.module_id >= runtime.modules.size())
    {
      return fail("primitive body runtime invalid module id");
    }
    auto & slot = runtime.modules[binding.module_id];
    if (!slot.module)
    {
      return fail("primitive body runtime missing output module");
    }
    Module & module = *slot.module;

    if (module.numeric_output_scalar_mask_.size() != module.outputs.size())
    {
      module.numeric_output_scalar_mask_.assign(module.outputs.size(), false);
      module.numeric_output_scalars_.assign(module.outputs.size(), 0.0);
    }
    else
    {
      std::fill(module.numeric_output_scalar_mask_.begin(), module.numeric_output_scalar_mask_.end(), false);
    }

    for (unsigned int output_id = 0; output_id < binding.output_registers.size(); ++output_id)
    {
      if (output_id >= module.outputs.size() || output_id >= binding.output_info.size())
      {
        return fail("primitive body runtime output metadata mismatch");
      }
      const auto output_kind =
        static_cast<egress_module_detail::NumericValueKind>(binding.output_info[output_id].kind);
      if (output_kind == egress_module_detail::NumericValueKind::Scalar)
      {
        const uint32_t output_reg = binding.output_registers[output_id];
        if (output_reg >= fused->temps.size())
        {
          return fail("primitive body runtime scalar output register out of range");
        }
        module.numeric_output_scalar_mask_[output_id] = true;
        module.numeric_output_scalars_[output_id] = fused->temps[output_reg];
      }
      else if (output_kind == egress_module_detail::NumericValueKind::Array)
      {
        const uint32_t global_array_slot = binding.array_base + binding.output_info[output_id].array_slot;
        if (global_array_slot >= fused->array_storage.size())
        {
          return fail("primitive body runtime array output slot out of range");
        }
        if (binding.output_info[output_id].array_slot >= module.numeric_array_storage_.size())
        {
          return fail("primitive body runtime module array output slot missing");
        }
        module.numeric_array_storage_[binding.output_info[output_id].array_slot] =
          fused->array_storage[global_array_slot];
        if (binding.output_info[output_id].array_slot < module.numeric_array_ptrs_.size())
        {
          auto & dst = module.numeric_array_storage_[binding.output_info[output_id].array_slot];
          module.numeric_array_ptrs_[binding.output_info[output_id].array_slot] = dst.empty() ? nullptr : dst.data();
          module.numeric_array_sizes_[binding.output_info[output_id].array_slot] = static_cast<uint64_t>(dst.size());
        }
      }

      if (output_id >= slot.output_materialize_mask.size() || slot.output_materialize_mask[output_id])
      {
        Module::assign_numeric_value_to(
          module.outputs[output_id],
          binding.output_info[output_id],
          binding.output_registers[output_id],
          fused->temps,
          fused->array_storage);
      }
    }

    if (module.numeric_registers_.size() < binding.register_targets.size())
    {
      module.numeric_registers_.resize(binding.register_targets.size(), 0.0);
    }
    if (module.numeric_next_registers_.size() < binding.register_targets.size())
    {
      module.numeric_next_registers_.resize(binding.register_targets.size(), 0.0);
    }
    if (module.registers_.size() < binding.register_targets.size())
    {
      module.registers_.resize(binding.register_targets.size(), expr::float_value(0.0));
    }
    if (module.next_registers_.size() < binding.register_targets.size())
    {
      module.next_registers_.resize(binding.register_targets.size(), expr::float_value(0.0));
    }
    if (module.numeric_register_arrays_.size() < binding.register_targets.size())
    {
      module.numeric_register_arrays_.resize(binding.register_targets.size());
    }

    for (unsigned int register_id = 0; register_id < binding.register_targets.size(); ++register_id)
    {
      if (register_id < binding.register_scalar_mask.size() &&
          !binding.register_scalar_mask[register_id])
      {
        continue;
      }
      const uint32_t fused_register_id = binding.register_base + register_id;
      if (fused_register_id >= fused->registers.size())
      {
        return fail("primitive body runtime scalar register out of range");
      }
      const int32_t target = binding.register_targets[register_id];
      const bool is_int = register_id < binding.register_int_mask.size() &&
                          binding.register_int_mask[register_id];
      const egress_jit::JitScalarType target_type =
        register_id < binding.register_target_types.size()
          ? binding.register_target_types[register_id]
          : egress_jit::JitScalarType::Float;
      double next_value;
      int64_t next_int_value = 0;
      if (target >= 0)
      {
        const auto t = static_cast<std::size_t>(target);
        if (is_int)
        {
          if (target_type == egress_jit::JitScalarType::Float)
          {
            next_int_value = (t < fused->temps.size())
              ? static_cast<int64_t>(fused->temps[t]) : 0;
          }
          else
          {
            next_int_value = (t < fused->int_temps.size())
              ? fused->int_temps[t] : 0;
          }
          next_value = static_cast<double>(next_int_value);
        }
        else
        {
          if (target_type == egress_jit::JitScalarType::Int || target_type == egress_jit::JitScalarType::Bool)
          {
            next_value = (t < fused->int_temps.size())
              ? static_cast<double>(fused->int_temps[t]) : fused->registers[fused_register_id];
          }
          else
          {
            next_value = (t < fused->temps.size())
              ? fused->temps[t] : fused->registers[fused_register_id];
          }
        }
      }
      else
      {
        next_value = fused->registers[fused_register_id];
        if (is_int && register_id < module.numeric_int_registers_.size())
          next_int_value = module.numeric_int_registers_[register_id];
      }
      fused->registers[fused_register_id] = next_value;
      module.numeric_registers_[register_id] = next_value;
      module.numeric_next_registers_[register_id] = next_value;
      if (is_int && register_id < module.numeric_int_registers_.size())
      {
        module.numeric_int_registers_[register_id] = next_int_value;
        module.numeric_next_int_registers_[register_id] = next_int_value;
      }
    }

    for (unsigned int register_id = 0; register_id < binding.array_register_targets.size(); ++register_id)
    {
      if (register_id >= binding.register_scalar_mask.size() || binding.register_scalar_mask[register_id])
      {
        continue;
      }
      const int32_t src_slot = binding.array_register_targets[register_id];
      if (src_slot < 0)
      {
        continue;
      }
      const uint32_t dst_slot = binding.array_base + binding.register_array_slot[register_id];
      const uint32_t global_src_slot = binding.array_base + static_cast<uint32_t>(src_slot);
      if (dst_slot >= fused->array_storage.size() || global_src_slot >= fused->array_storage.size())
      {
        return fail("primitive body runtime array register slot out of range");
      }
      auto & dst = fused->array_storage[dst_slot];
      auto & src = fused->array_storage[global_src_slot];
      if (dst.size() != src.size())
      {
        return fail("primitive body runtime array register size mismatch");
      }
      if (register_id < binding.array_register_can_swap.size() && binding.array_register_can_swap[register_id])
      {
        dst.swap(src);
        fused->array_ptrs[dst_slot] = dst.empty() ? nullptr : dst.data();
        fused->array_sizes[dst_slot] = static_cast<uint64_t>(dst.size());
        fused->array_ptrs[global_src_slot] = src.empty() ? nullptr : src.data();
        fused->array_sizes[global_src_slot] = static_cast<uint64_t>(src.size());
      }
      else
      {
        std::copy(src.begin(), src.end(), dst.begin());
      }
    }

    for (unsigned int register_id = 0; register_id < binding.register_array_slot.size(); ++register_id)
    {
      if (register_id >= binding.register_scalar_mask.size() || binding.register_scalar_mask[register_id])
      {
        continue;
      }
      const uint32_t local_array_slot = binding.register_array_slot[register_id];
      const uint32_t global_array_slot = binding.array_base + local_array_slot;
      if (global_array_slot >= fused->array_storage.size() ||
          local_array_slot >= module.numeric_array_storage_.size() ||
          register_id >= module.registers_.size() ||
          register_id >= module.next_registers_.size())
      {
        return fail("primitive body runtime local array register sync mismatch");
      }
      module.numeric_array_storage_[local_array_slot] = fused->array_storage[global_array_slot];
      if (local_array_slot < module.numeric_array_ptrs_.size())
      {
        auto & values = module.numeric_array_storage_[local_array_slot];
        module.numeric_array_ptrs_[local_array_slot] = values.empty() ? nullptr : values.data();
        module.numeric_array_sizes_[local_array_slot] = static_cast<uint64_t>(values.size());
      }
      module.numeric_register_arrays_[register_id] = module.numeric_array_storage_[local_array_slot];
    }

    module.value_registers_dirty_ = true;
    module.reset_inputs_after_process();
  }

  fused->primitive_body_status = "numeric JIT active";
  return true;
#endif
}

bool Graph::run_fused_mix_kernel(RuntimeState & runtime, double & mixed) const
{
#ifndef EGRESS_LLVM_ORC_JIT
  (void)runtime;
  (void)mixed;
  return false;
#else
  auto * fused = runtime.fused_graph.get();
  if (!fusion_enabled_ ||
      fused == nullptr ||
      !fused->mix_kernel.available ||
      fused->mix_kernel.kernel == nullptr)
  {
    return false;
  }

  for (std::size_t source_slot = 0; source_slot < fused->source_outputs.size(); ++source_slot)
  {
    const auto & source = fused->source_outputs[source_slot];
    if (source.module_id < runtime.modules.size())
    {
      const auto & slot = runtime.modules[source.module_id];
      double numeric_scalar = 0.0;
      if (slot.module != nullptr &&
          slot.module->try_get_numeric_scalar_output(source.output_id, false, numeric_scalar))
      {
        fused->mix_kernel.scalar_inputs[source_slot] = numeric_scalar;
        continue;
      }
    }
    if (source_slot >= fused->current_outputs.size())
    {
      continue;
    }
    const Value & value = fused->current_outputs[source_slot];
    fused->mix_kernel.scalar_inputs[source_slot] =
      (value.type == ValueType::Array || value.type == ValueType::Matrix) ? 0.0 : expr::to_float64(value);
  }

  for (const auto & [source_slot, array_slot] : fused->mix_kernel.source_array_slots)
  {
    if (array_slot >= fused->mix_kernel.array_storage.size())
    {
      continue;
    }
    auto & dst = fused->mix_kernel.array_storage[array_slot];

    if (source_slot < fused->source_outputs.size())
    {
      const auto & source = fused->source_outputs[source_slot];
      if (source.module_id < runtime.modules.size())
      {
        const auto & slot = runtime.modules[source.module_id];
        if (slot.module != nullptr)
        {
          const auto * numeric_values = slot.module->try_get_numeric_output_array_values(source.output_id);
          if (numeric_values != nullptr)
          {
            if (dst.size() != numeric_values->size())
            {
              dst.assign(numeric_values->size(), 0.0);
              fused->mix_kernel.array_ptrs[array_slot] = dst.empty() ? nullptr : dst.data();
              fused->mix_kernel.array_sizes[array_slot] = static_cast<uint64_t>(dst.size());
            }
            for (std::size_t item_id = 0; item_id < numeric_values->size(); ++item_id)
            {
              dst[item_id] = (*numeric_values)[item_id];
            }
            continue;
          }
        }
      }
    }

    if (source_slot >= fused->current_outputs.size())
    {
      continue;
    }
    const Value & value = fused->current_outputs[source_slot];
    if (!expr::is_array(value))
    {
      continue;
    }
    if (dst.size() != value.array_items.size())
    {
      dst.assign(value.array_items.size(), 0.0);
      fused->mix_kernel.array_ptrs[array_slot] = dst.empty() ? nullptr : dst.data();
      fused->mix_kernel.array_sizes[array_slot] = static_cast<uint64_t>(dst.size());
    }
    for (std::size_t item_id = 0; item_id < value.array_items.size(); ++item_id)
    {
      dst[item_id] = expr::to_float64(value.array_items[item_id]);
    }
  }

  fused->mix_kernel.kernel(
    fused->mix_kernel.scalar_inputs.data(),
    nullptr,
    fused->mix_kernel.array_ptrs.data(),
    fused->mix_kernel.array_sizes.data(),
    fused->mix_kernel.temps.data(),
    44100.0,
    0,
    fused->mix_kernel.param_ptrs.data(),
    nullptr, nullptr, nullptr,
    fused->mix_kernel.int_temps.empty() ? nullptr : fused->mix_kernel.int_temps.data());

  for (const auto & binding : fused->mix_kernel.mix_bindings)
  {
    if (binding.value.kind != egress_graph_detail::FusedGraphValueKind::Scalar ||
        binding.value.scalar_register >= fused->mix_kernel.temps.size())
    {
      continue;
    }
    mixed += fused->mix_kernel.temps[binding.value.scalar_register] / 20.0;
  }
  return true;
#endif
}

void Graph::refresh_runtime_ref_metadata(RuntimeState & runtime) const
{
  for (auto & slot : runtime.modules)
  {
    if (!slot.module)
    {
      continue;
    }
    slot.output_materialize_mask.assign(slot.module->outputs.size(), false);
    slot.output_prev_materialize_mask.assign(slot.module->outputs.size(), false);
    slot.indexed_output_indices.assign(slot.module->outputs.size(), {});
    slot.indexed_prev_output_values.assign(slot.module->outputs.size(), {});
  }

  auto refresh_program = [&](auto & program)
  {
    for (auto & instr : program.instructions)
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        continue;
      }

      ModuleSlot & source_slot = runtime.modules[instr.ref_module_id];
      if (instr.ref_output_id >= source_slot.indexed_output_indices.size())
      {
        continue;
      }

      if (instr.opcode != OpCode::RefIndex || instr.ref_index < 0)
      {
        continue;
      }

      auto & indices = source_slot.indexed_output_indices[instr.ref_output_id];
      auto it = std::find(indices.begin(), indices.end(), instr.ref_index);
      uint32_t cache_slot = 0;
      if (it == indices.end())
      {
        cache_slot = static_cast<uint32_t>(indices.size());
        indices.push_back(instr.ref_index);
        source_slot.indexed_prev_output_values[instr.ref_output_id].push_back(float_value(0.0));
      }
      else
      {
        cache_slot = static_cast<uint32_t>(std::distance(indices.begin(), it));
      }
      instr.src_a = cache_slot;
    }
  };

  auto mark_output_materialization = [&](const auto & program, bool previous_outputs)
  {
    for (const auto & instr : program.instructions)
    {
      if ((instr.opcode != OpCode::Ref && instr.opcode != OpCode::RefIndex) ||
          instr.ref_module_id >= runtime.modules.size())
      {
        continue;
      }

      if (previous_outputs && instr.opcode == OpCode::RefIndex)
      {
        continue;
      }

      ModuleSlot & source_slot = runtime.modules[instr.ref_module_id];
#ifdef EGRESS_LLVM_ORC_JIT
      if (source_slot.module != nullptr)
      {
        if (previous_outputs)
        {
          if (instr.opcode == OpCode::Ref)
          {
            egress_module_detail::NumericValueRef ref;
            if (source_slot.module->try_get_numeric_output_ref(instr.ref_output_id, ref) &&
                (ref.kind == egress_module_detail::NumericValueKind::Scalar ||
                 ref.kind == egress_module_detail::NumericValueKind::Array))
            {
              continue;
            }
          }
        }
        else
        {
          if (instr.opcode == OpCode::Ref)
          {
            egress_module_detail::NumericValueRef ref;
            if (source_slot.module->try_get_numeric_output_ref(instr.ref_output_id, ref) &&
                (ref.kind == egress_module_detail::NumericValueKind::Scalar ||
                 ref.kind == egress_module_detail::NumericValueKind::Array))
            {
              continue;
            }
          }
          else if (instr.opcode == OpCode::RefIndex &&
                   source_slot.module->try_get_numeric_output_array_values(instr.ref_output_id) != nullptr)
          {
            continue;
          }
        }
      }
#endif
      auto & mask = previous_outputs ? source_slot.output_prev_materialize_mask : source_slot.output_materialize_mask;
      if (instr.ref_output_id >= mask.size())
      {
        continue;
      }
      mask[instr.ref_output_id] = true;
    }
  };

  for (auto & consumer_slot : runtime.modules)
  {
    refresh_program(consumer_slot.input_program);
    mark_output_materialization(consumer_slot.input_program, true);
  }
  for (auto & mix_expr : runtime.mix_exprs)
  {
    refresh_program(mix_expr.program);
    mark_output_materialization(mix_expr.program, false);
  }

  for (auto & slot : runtime.modules)
  {
    if (slot.output_materialize_mask.size() != slot.output_prev_materialize_mask.size())
    {
      continue;
    }
    for (std::size_t output_id = 0; output_id < slot.output_materialize_mask.size(); ++output_id)
    {
      slot.output_materialize_mask[output_id] =
        slot.output_materialize_mask[output_id] || slot.output_prev_materialize_mask[output_id];
    }
  }
}

bool Graph::recompile_module_inputs_in_runtime(
  RuntimeState & runtime,
  const std::string & module_name,
  const std::vector<ExprSpecPtr> & exprs) const
{
  auto runtime_id_it = runtime.name_to_id.find(module_name);
  if (runtime_id_it == runtime.name_to_id.end() || runtime_id_it->second >= runtime.modules.size())
  {
    return false;
  }

  ModuleSlot & slot = runtime.modules[runtime_id_it->second];
  const unsigned int input_count = static_cast<unsigned int>(exprs.size());
  slot.input_program = compile_input_program(exprs, input_count, runtime);
  slot.input_registers.assign(slot.input_program.register_count, float_value(0.0));
  refresh_runtime_ref_metadata(runtime);
  rebuild_fused_graph_state(runtime);
  return true;
}

std::unique_ptr<Graph::FusedGraphState> Graph::build_fused_graph_state(const RuntimeState & runtime) const
{
  auto fused = std::make_unique<FusedGraphState>();
  fused->module_output_spans.resize(runtime.modules.size());

#ifndef EGRESS_LLVM_ORC_JIT
  fused->candidate_reason = "graph fusion requires LLVM ORC JIT";
  return fused;
#endif

  auto mark_ineligible = [&](const std::string & reason)
  {
    fused->numeric_candidate = false;
    fused->candidate_reason = reason;
  };

  for (uint32_t module_id = 0; module_id < runtime.modules.size(); ++module_id)
  {
    const auto & slot = runtime.modules[module_id];
    auto & span = fused->module_output_spans[module_id];
    span.first_output_slot = static_cast<uint32_t>(fused->source_outputs.size());
    if (!slot.module)
    {
      continue;
    }

    span.output_count = static_cast<uint32_t>(slot.module->outputs.size());
    for (unsigned int output_id = 0; output_id < slot.module->outputs.size(); ++output_id)
    {
      const uint32_t source_slot = static_cast<uint32_t>(fused->source_outputs.size());
      egress_graph_detail::FusedGraphSourceOutput output;
      output.module_id = module_id;
      output.output_id = output_id;
      output.materialized =
        output_id < slot.output_materialize_mask.size() ? slot.output_materialize_mask[output_id] : true;
      fused->source_outputs.push_back(output);
      fused->source_output_lookup.emplace(fused_source_output_key(module_id, output_id), source_slot);
      fused->current_outputs.push_back(
        output_id < slot.module->outputs.size() ? slot.module->outputs[output_id] : float_value(0.0));
      fused->prev_outputs.push_back(
        output_id < slot.module->prev_outputs.size() ? slot.module->prev_outputs[output_id] : float_value(0.0));
      fused->indexed_prev_indices.push_back(
        output_id < slot.indexed_output_indices.size() ? slot.indexed_output_indices[output_id] : std::vector<int64_t>{});
      fused->indexed_prev_values.push_back(
        output_id < slot.indexed_prev_output_values.size() ? slot.indexed_prev_output_values[output_id] : std::vector<Value>{});
    }



    if (!is_fused_numeric_candidate(slot.input_program))
    {
      mark_ineligible("graph fusion disabled by non-numeric top-level input expression");
      return fused;
    }

  }

  fused->mix_source_output_slots.reserve(runtime.mix.size());
  for (const auto & tap : runtime.mix)
  {
    const auto it = fused->source_output_lookup.find(fused_source_output_key(tap.module_id, tap.output_id));
    fused->mix_source_output_slots.push_back(
      it == fused->source_output_lookup.end() ? std::numeric_limits<uint32_t>::max() : it->second);
  }

  fused->tap_source_output_slots.reserve(runtime.taps.size());
  for (const auto & tap : runtime.taps)
  {
    if (!tap.valid)
    {
      fused->tap_source_output_slots.push_back(std::numeric_limits<uint32_t>::max());
      continue;
    }
    const auto it = fused->source_output_lookup.find(fused_source_output_key(tap.module_id, tap.output_id));
    fused->tap_source_output_slots.push_back(
      it == fused->source_output_lookup.end() ? std::numeric_limits<uint32_t>::max() : it->second);
  }

  for (const auto & mix_expr : runtime.mix_exprs)
  {
    if (!is_fused_numeric_candidate(mix_expr.program))
    {
      mark_ineligible("graph fusion disabled by non-numeric mix expression");
      return fused;
    }
  }

  if (runtime.modules.empty())
  {
    mark_ineligible("graph fusion requires at least one module");
    return fused;
  }

  std::vector<uint32_t> primitive_input_base_by_module(runtime.modules.size(), std::numeric_limits<uint32_t>::max());
  std::vector<std::vector<uint32_t>> primitive_array_input_slots_by_module(runtime.modules.size());

 #ifdef EGRESS_LLVM_ORC_JIT
  auto mark_primitive_body_unavailable = [&](const std::string & reason)
  {
    fused->primitive_body_available = false;
    fused->primitive_body_status = reason;
    fused->primitive_body_modules.clear();
    fused->primitive_body_module_mask.assign(runtime.modules.size(), false);
    fused->primitive_body_covers_all_modules = false;
    fused->program = egress_jit::NumericProgram{};
    fused->kernel = nullptr;
    fused->inputs.clear();
    fused->registers.clear();
    fused->temps.clear();
    fused->array_storage.clear();
    fused->array_ptrs.clear();
    fused->array_sizes.clear();
    std::fill(
      primitive_input_base_by_module.begin(),
      primitive_input_base_by_module.end(),
      std::numeric_limits<uint32_t>::max());
  };

  mark_primitive_body_unavailable("primitive body fusion unavailable");

  auto translate_body_instruction =
    [&](const egress_jit::NumericInstr & instr,
        uint32_t input_base,
        uint32_t register_base,
        uint32_t array_base,
        uint32_t reg_base,
        egress_jit::NumericInstr & out) -> bool
  {
    out = instr;
    switch (instr.op)
    {
      case egress_jit::NumericOp::Literal:
      case egress_jit::NumericOp::SampleRate:
      case egress_jit::NumericOp::SampleIndex:
        out.dst += reg_base;
        return true;
      case egress_jit::NumericOp::InputValue:
        out.dst += reg_base;
        out.slot_id = input_base + instr.slot_id;
        return true;
      case egress_jit::NumericOp::RegisterValue:
        out.dst += reg_base;
        out.slot_id = register_base + instr.slot_id;
        return true;
      case egress_jit::NumericOp::Not:
      case egress_jit::NumericOp::Less:
      case egress_jit::NumericOp::LessEqual:
      case egress_jit::NumericOp::Greater:
      case egress_jit::NumericOp::GreaterEqual:
      case egress_jit::NumericOp::Equal:
      case egress_jit::NumericOp::NotEqual:
      case egress_jit::NumericOp::Abs:
      case egress_jit::NumericOp::Add:
      case egress_jit::NumericOp::Sub:
      case egress_jit::NumericOp::Mul:
      case egress_jit::NumericOp::Div:
      case egress_jit::NumericOp::Pow:
      case egress_jit::NumericOp::Mod:
      case egress_jit::NumericOp::FloorDiv:
      case egress_jit::NumericOp::BitAnd:
      case egress_jit::NumericOp::BitOr:
      case egress_jit::NumericOp::BitXor:
      case egress_jit::NumericOp::LShift:
      case egress_jit::NumericOp::RShift:
      case egress_jit::NumericOp::Log:
      case egress_jit::NumericOp::IndexArray:
      case egress_jit::NumericOp::Sin:
      case egress_jit::NumericOp::Neg:
      case egress_jit::NumericOp::BitNot:
        out.dst += reg_base;
        out.src_a += reg_base;
        if (instr.op == egress_jit::NumericOp::IndexArray)
        {
          out.slot_id += array_base;
        }
        return true;
      case egress_jit::NumericOp::Clamp:
        out.dst += reg_base;
        out.src_a += reg_base;
        out.src_b += reg_base;
        out.src_c += reg_base;
        return true;
      case egress_jit::NumericOp::ArrayLessScalar:
      case egress_jit::NumericOp::ArrayLessEqualScalar:
      case egress_jit::NumericOp::ArrayGreaterScalar:
      case egress_jit::NumericOp::ArrayGreaterEqualScalar:
      case egress_jit::NumericOp::ArrayEqualScalar:
      case egress_jit::NumericOp::ArrayNotEqualScalar:
      case egress_jit::NumericOp::ArrayAddScalar:
      case egress_jit::NumericOp::ArrayMulScalar:
      case egress_jit::NumericOp::ArrayDivScalar:
      case egress_jit::NumericOp::ArrayModScalar:
        out.dst += array_base;
        out.src_a += array_base;
        out.src_b += reg_base;
        return true;
      case egress_jit::NumericOp::ArrayPack:
        out.dst += array_base;
        for (auto & arg : out.args)
        {
          arg += reg_base;
        }
        return true;
      case egress_jit::NumericOp::ArrayAdd:
      case egress_jit::NumericOp::ArraySub:
      case egress_jit::NumericOp::ArrayMul:
      case egress_jit::NumericOp::ArrayDiv:
        out.dst += array_base;
        out.src_a += reg_base;
        out.src_b += reg_base;
        out.src_a += (array_base - reg_base);
        out.src_b += (array_base - reg_base);
        return true;
      case egress_jit::NumericOp::MatMul:
        out.dst += array_base;
        out.src_a += array_base;
        out.src_b += array_base;
        return true;
      default:
        return false;
    }
  };

  {
    bool primitive_body_ok = true;
    uint64_t expected_sample_index = 0;
    bool have_sample_index = false;
    double expected_sample_rate = 44100.0;
    bool have_sample_rate = false;

    fused->program.instructions.clear();
    fused->program.register_count = 0;
    fused->primitive_body_modules.clear();
    fused->primitive_body_module_mask.assign(runtime.modules.size(), false);
    fused->primitive_body_covers_all_modules = false;
    fused->inputs.clear();
    fused->registers.clear();
    fused->temps.clear();
    fused->array_storage.clear();
    fused->array_ptrs.clear();
    fused->array_sizes.clear();
    uint32_t active_module_count = 0;
    for (uint32_t module_id = 0; module_id < runtime.modules.size(); ++module_id)
    {
      const auto & slot = runtime.modules[module_id];
      if (!slot.module)
      {
        continue;
      }
      ++active_module_count;

      Module & module = *slot.module;
      if (module.has_nested_modules_ || module.has_delay_states_)
      {
        continue;
      }

      if (!have_sample_index)
      {
        expected_sample_index = module.sample_index_;
        have_sample_index = true;
      }
      else if (module.sample_index_ != expected_sample_index)
      {
        continue;
      }

      if (!have_sample_rate)
      {
        expected_sample_rate = module.sample_rate_;
        have_sample_rate = true;
      }
      else if (module.sample_rate_ != expected_sample_rate)
      {
        continue;
      }

      Module::NumericJitState state;
      egress_jit::NumericProgram module_program;
      const Module::CompiledProgram compiled_program = module.program_;
      if (!module.build_numeric_program(compiled_program, state, module.inputs, module_program))
      {
        continue;
      }

      for (const auto & output_info : state.output_info)
      {
        if (static_cast<Module::NumericValueKind>(output_info.kind) == Module::NumericValueKind::Matrix)
        {
          primitive_body_ok = false;
          break;
        }
      }
      if (!primitive_body_ok)
      {
        break;
      }

      const uint32_t input_base = static_cast<uint32_t>(fused->inputs.size());
      const uint32_t register_base = static_cast<uint32_t>(fused->registers.size());
      const uint32_t array_base = static_cast<uint32_t>(fused->array_storage.size());
      const uint32_t reg_base = fused->program.register_count;

      primitive_input_base_by_module[module_id] = input_base;
      fused->inputs.resize(fused->inputs.size() + module.inputs.size(), 0.0);
      fused->registers.resize(fused->registers.size() + module.registers_.size(), 0.0);
      for (std::size_t reg_id = 0; reg_id < module.registers_.size(); ++reg_id)
      {
        if (reg_id < state.register_scalar_mask.size() && state.register_scalar_mask[reg_id])
        {
          fused->registers[register_base + reg_id] = expr::to_float64(module.registers_[reg_id]);
        }
      }
      primitive_array_input_slots_by_module[module_id].assign(module.inputs.size(), std::numeric_limits<uint32_t>::max());
      for (unsigned int input_id = 0; input_id < state.input_info.size(); ++input_id)
      {
        if (!state.input_info[input_id].is_scalar)
        {
          primitive_array_input_slots_by_module[module_id][input_id] = array_base + state.input_info[input_id].array_slot;
        }
      }
      fused->array_storage.insert(fused->array_storage.end(), state.array_storage.begin(), state.array_storage.end());

      egress_graph_detail::FusedPrimitiveBodyModule binding;
      binding.module_id = module_id;
      binding.input_base = input_base;
      binding.register_base = register_base;
      binding.array_base = array_base;
      binding.array_slot_count = static_cast<uint32_t>(state.array_storage.size());
      binding.input_info = state.input_info;
      binding.output_info = state.output_info;
      binding.register_scalar_mask = state.register_scalar_mask;
      binding.register_array_slot = state.register_array_slot;
      binding.array_register_targets = state.array_register_targets;
      binding.array_register_can_swap = state.array_register_can_swap;
      binding.register_int_mask = state.register_int_mask;
      binding.register_target_types = state.register_target_types;
      binding.output_registers.reserve(module.program_.output_targets.size());
      for (uint32_t output_reg : module.program_.output_targets)
      {
        binding.output_registers.push_back(reg_base + output_reg);
      }
      binding.register_targets.reserve(module.program_.register_targets.size());
      for (int32_t target : module.program_.register_targets)
      {
        binding.register_targets.push_back(target >= 0 ? static_cast<int32_t>(reg_base + static_cast<uint32_t>(target)) : -1);
      }

      for (const auto & instr : module_program.instructions)
      {
        egress_jit::NumericInstr translated;
        if (!translate_body_instruction(instr, input_base, register_base, array_base, reg_base, translated))
        {
          primitive_body_ok = false;
          break;
        }
        fused->program.instructions.push_back(std::move(translated));
      }
      if (!primitive_body_ok)
      {
        break;
      }

      fused->program.register_count += module_program.register_count;
      fused->primitive_body_module_mask[module_id] = true;
      fused->primitive_body_modules.push_back(std::move(binding));
    }

    if (primitive_body_ok && !fused->primitive_body_modules.empty())
    {
      auto & jit = egress_jit::OrcJitEngine::instance();
      auto kernel_or_err = jit.compile_numeric_program(fused->program);
      if (!kernel_or_err)
      {
        mark_primitive_body_unavailable(llvm::toString(kernel_or_err.takeError()));
      }
      else
      {
        fused->kernel = *kernel_or_err;
        fused->primitive_body_available = true;
        fused->primitive_body_status = "numeric JIT active";
        fused->primitive_body_sample_rate = expected_sample_rate;
        fused->param_ptrs.clear();
        {
          std::unordered_map<uint64_t, uint32_t> seen;
          for (const auto & instr : fused->program.instructions)
          {
            if (instr.op == egress_jit::NumericOp::SmoothedParam && instr.param_ptr != 0 &&
                seen.find(instr.param_ptr) == seen.end())
            {
              seen.emplace(instr.param_ptr, static_cast<uint32_t>(fused->param_ptrs.size()));
              fused->param_ptrs.push_back(instr.param_ptr);
            }
          }
        }
        fused->temps.assign(fused->program.register_count, 0.0);
        fused->int_temps.assign(fused->program.register_count, 0);
        fused->array_ptrs.resize(fused->array_storage.size(), nullptr);
        fused->array_sizes.resize(fused->array_storage.size(), 0);
        for (std::size_t array_id = 0; array_id < fused->array_storage.size(); ++array_id)
        {
          fused->array_ptrs[array_id] =
            fused->array_storage[array_id].empty() ? nullptr : fused->array_storage[array_id].data();
          fused->array_sizes[array_id] = static_cast<uint64_t>(fused->array_storage[array_id].size());
        }
        fused->primitive_body_covers_all_modules =
          fused->primitive_body_modules.size() == active_module_count;
      }
    }
    else
    {
      mark_primitive_body_unavailable("no eligible primitive modules for body fusion");
    }
  }

  auto build_expression_kernel = [&](
                                   const std::vector<const CompiledInputProgram *> & programs,
                                   bool use_prev_outputs,
                                   egress_graph_detail::FusedGraphKernelState & kernel_state,
                                   auto bind_result) -> bool
  {
    kernel_state = egress_graph_detail::FusedGraphKernelState{};
    if (programs.empty())
    {
      kernel_state.status = "empty";
      return true;
    }

    struct LoweredRegInfo
    {
      egress_graph_detail::FusedGraphValueKind kind =
        egress_graph_detail::FusedGraphValueKind::Scalar;
      uint32_t scalar_register = 0;
      uint32_t array_slot = 0;
      uint32_t array_size = 0;
      bool scalar_constant = false;
      double constant_value = 0.0;
    };

    auto make_scalar_ref = [](uint32_t reg, bool is_const = false, double constant = 0.0)
    {
      LoweredRegInfo info;
      info.kind = egress_graph_detail::FusedGraphValueKind::Scalar;
      info.scalar_register = reg;
      info.scalar_constant = is_const;
      info.constant_value = constant;
      return info;
    };

    auto make_array_ref = [](uint32_t slot, uint32_t size)
    {
      LoweredRegInfo info;
      info.kind = egress_graph_detail::FusedGraphValueKind::Array;
      info.array_slot = slot;
      info.array_size = size;
      return info;
    };

    auto append_array_values = [&](const std::vector<Value> & values) -> uint32_t
    {
      const uint32_t slot = static_cast<uint32_t>(kernel_state.array_storage.size());
      kernel_state.array_storage.emplace_back();
      auto & dst = kernel_state.array_storage.back();
      dst.reserve(values.size());
      for (const auto & item : values)
      {
        dst.push_back(expr::to_float64(item));
      }
      return slot;
    };

    auto allocate_array_slot = [&](uint32_t size) -> uint32_t
    {
      const uint32_t slot = static_cast<uint32_t>(kernel_state.array_storage.size());
      kernel_state.array_storage.emplace_back(size, 0.0);
      return slot;
    };

    auto ensure_source_array_slot = [&](uint32_t source_slot) -> uint32_t
    {
      const auto existing = kernel_state.source_array_slots.find(source_slot);
      if (existing != kernel_state.source_array_slots.end())
      {
        return existing->second;
      }

      if (source_slot >= (use_prev_outputs ? fused->prev_outputs.size() : fused->current_outputs.size()))
      {
        return std::numeric_limits<uint32_t>::max();
      }

      const Value & value =
        use_prev_outputs ? fused->prev_outputs[source_slot] : fused->current_outputs[source_slot];
      if (!expr::is_array(value))
      {
        return std::numeric_limits<uint32_t>::max();
      }

      const uint32_t slot = append_array_values(value.array_items);
      kernel_state.source_array_slots.emplace(source_slot, slot);
      return slot;
    };

    auto emit_constant_temp = [&](double constant) -> uint32_t
    {
      const uint32_t reg = kernel_state.program.register_count++;
      egress_jit::NumericInstr literal;
      literal.op = egress_jit::NumericOp::Literal;
      literal.dst = reg;
      literal.literal = constant;
      kernel_state.program.instructions.push_back(std::move(literal));
      return reg;
    };

    auto lower_literal = [&](const Value & literal, uint32_t dst_reg, LoweredRegInfo & out) -> bool
    {
      if (literal.type == ValueType::Matrix)
      {
        return false;
      }
      if (literal.type == ValueType::Array)
      {
        out = make_array_ref(append_array_values(literal.array_items),
                             static_cast<uint32_t>(literal.array_items.size()));
        return true;
      }

      egress_jit::NumericInstr instr;
      instr.op = egress_jit::NumericOp::Literal;
      instr.dst = dst_reg;
      instr.literal = expr::to_float64(literal);
      kernel_state.program.instructions.push_back(std::move(instr));
      out = make_scalar_ref(dst_reg, true, expr::to_float64(literal));
      return true;
    };

    auto lower_binary_scalar = [&](egress_jit::NumericOp op, uint32_t dst_reg, const LoweredRegInfo & lhs, const LoweredRegInfo & rhs)
    {
      egress_jit::NumericInstr instr;
      instr.op = op;
      instr.dst = dst_reg;
      instr.src_a = lhs.scalar_register;
      instr.src_b = rhs.scalar_register;
      kernel_state.program.instructions.push_back(std::move(instr));
    };

    kernel_state.program.instructions.clear();
    kernel_state.program.register_count = 0;

    for (std::size_t program_index = 0; program_index < programs.size(); ++program_index)
    {
      const CompiledInputProgram * program = programs[program_index];
      if (program == nullptr)
      {
        continue;
      }
      const uint32_t reg_base = kernel_state.program.register_count;
      kernel_state.program.register_count += program->register_count;
      std::vector<LoweredRegInfo> reg_info(program->register_count);

      for (const auto & instr : program->instructions)
      {
        if (instr.dst >= reg_info.size())
        {
          return false;
        }

        LoweredRegInfo result = make_scalar_ref(reg_base + instr.dst);
        const uint32_t dst_reg = reg_base + instr.dst;
        const auto load_const_rhs = [&](const Value & literal) {
          return make_scalar_ref(emit_constant_temp(expr::to_float64(literal)), true, expr::to_float64(literal));
        };

        switch (instr.opcode)
        {
          case OpCode::Literal:
            if (!lower_literal(instr.literal, dst_reg, result))
            {
              return false;
            }
            break;
          case OpCode::Ref:
          {
            const uint32_t source_slot = fused_source_output_key(instr.ref_module_id, instr.ref_output_id);
            const auto it = fused->source_output_lookup.find(source_slot);
            if (it == fused->source_output_lookup.end())
            {
              if (!lower_literal(float_value(0.0), dst_reg, result))
              {
                return false;
              }
              break;
            }

            const Value & source_value =
              use_prev_outputs ? fused->prev_outputs[it->second] : fused->current_outputs[it->second];
            if (source_value.type == ValueType::Matrix)
            {
              return false;
            }
            if (source_value.type == ValueType::Array)
            {
              const uint32_t array_slot = ensure_source_array_slot(it->second);
              if (array_slot == std::numeric_limits<uint32_t>::max())
              {
                return false;
              }
              result = make_array_ref(array_slot, static_cast<uint32_t>(source_value.array_items.size()));
              break;
            }

            egress_jit::NumericInstr numeric_instr;
            numeric_instr.op = egress_jit::NumericOp::InputValue;
            numeric_instr.dst = dst_reg;
            numeric_instr.slot_id = it->second;
            kernel_state.program.instructions.push_back(std::move(numeric_instr));
            result = make_scalar_ref(dst_reg);
            break;
          }
          case OpCode::RefIndex:
          {
            const auto source_it =
              fused->source_output_lookup.find(fused_source_output_key(instr.ref_module_id, instr.ref_output_id));
            if (source_it == fused->source_output_lookup.end() || instr.ref_index < 0)
            {
              if (!lower_literal(float_value(0.0), dst_reg, result))
              {
                return false;
              }
              break;
            }

            const Value & source_value =
              use_prev_outputs ? fused->prev_outputs[source_it->second] : fused->current_outputs[source_it->second];
            if (!expr::is_array(source_value))
            {
              if (use_prev_outputs)
              {
                const uint64_t scalar_key =
                  (static_cast<uint64_t>(source_it->second) << 32) | static_cast<uint64_t>(instr.src_a);
                auto existing = kernel_state.indexed_prev_scalar_slots.find(scalar_key);
                uint32_t scalar_slot = 0;
                if (existing != kernel_state.indexed_prev_scalar_slots.end())
                {
                  scalar_slot = existing->second;
                }
                else
                {
                  scalar_slot = static_cast<uint32_t>(kernel_state.scalar_inputs.size());
                  kernel_state.indexed_prev_scalar_slots.emplace(scalar_key, scalar_slot);
                  kernel_state.scalar_inputs.push_back(0.0);
                }
                egress_jit::NumericInstr numeric_instr;
                numeric_instr.op = egress_jit::NumericOp::InputValue;
                numeric_instr.dst = dst_reg;
                numeric_instr.slot_id = scalar_slot;
                kernel_state.program.instructions.push_back(std::move(numeric_instr));
                result = make_scalar_ref(dst_reg);
                break;
              }
              return false;
            }

            const uint32_t array_slot = ensure_source_array_slot(source_it->second);
            if (array_slot == std::numeric_limits<uint32_t>::max())
            {
              return false;
            }
            egress_jit::NumericInstr numeric_instr;
            numeric_instr.op = egress_jit::NumericOp::IndexArray;
            numeric_instr.dst = dst_reg;
            numeric_instr.src_a = emit_constant_temp(static_cast<double>(instr.ref_index));
            numeric_instr.slot_id = array_slot;
            kernel_state.program.instructions.push_back(std::move(numeric_instr));
            result = make_scalar_ref(dst_reg);
            break;
          }
          case OpCode::ArrayPack:
          {
            bool all_constant = true;
            std::vector<Value> packed_values;
            packed_values.reserve(instr.args.size());
            for (uint32_t arg : instr.args)
            {
              if (arg >= reg_info.size() ||
                  reg_info[arg].kind != egress_graph_detail::FusedGraphValueKind::Scalar)
              {
                return false;
              }
              all_constant = all_constant && reg_info[arg].scalar_constant;
              packed_values.push_back(float_value(reg_info[arg].constant_value));
            }

            if (all_constant)
            {
              result = make_array_ref(append_array_values(packed_values),
                                      static_cast<uint32_t>(packed_values.size()));
              break;
            }

            const uint32_t array_slot = allocate_array_slot(static_cast<uint32_t>(instr.args.size()));
            egress_jit::NumericInstr numeric_instr;
            numeric_instr.op = egress_jit::NumericOp::ArrayPack;
            numeric_instr.dst = array_slot;
            for (uint32_t arg : instr.args)
            {
              numeric_instr.args.push_back(reg_info[arg].scalar_register);
            }
            kernel_state.program.instructions.push_back(std::move(numeric_instr));
            result = make_array_ref(array_slot, static_cast<uint32_t>(instr.args.size()));
            break;
          }
          case OpCode::ArraySet:
          {
            if (instr.src_a >= reg_info.size() ||
                instr.src_b >= reg_info.size() ||
                instr.src_c >= reg_info.size())
            {
              return false;
            }
            const auto & array_value = reg_info[instr.src_a];
            const auto & index_value = reg_info[instr.src_b];
            const auto & item_value = reg_info[instr.src_c];
            if (array_value.kind != egress_graph_detail::FusedGraphValueKind::Array ||
                index_value.kind != egress_graph_detail::FusedGraphValueKind::Scalar ||
                item_value.kind != egress_graph_detail::FusedGraphValueKind::Scalar ||
                !index_value.scalar_constant)
            {
              return false;
            }
            const int64_t array_index = static_cast<int64_t>(index_value.constant_value);
            if (array_index < 0 || static_cast<uint32_t>(array_index) >= array_value.array_size)
            {
              return false;
            }

            const uint32_t array_slot = allocate_array_slot(array_value.array_size);
            egress_jit::NumericInstr pack_instr;
            pack_instr.op = egress_jit::NumericOp::ArrayPack;
            pack_instr.dst = array_slot;
            pack_instr.args.reserve(array_value.array_size);

            for (uint32_t item_index = 0; item_index < array_value.array_size; ++item_index)
            {
              if (item_index == static_cast<uint32_t>(array_index))
              {
                pack_instr.args.push_back(item_value.scalar_register);
                continue;
              }
              const uint32_t item_reg = kernel_state.program.register_count++;
              egress_jit::NumericInstr index_instr;
              index_instr.op = egress_jit::NumericOp::IndexArray;
              index_instr.dst = item_reg;
              index_instr.src_a = emit_constant_temp(static_cast<double>(item_index));
              index_instr.slot_id = array_value.array_slot;
              kernel_state.program.instructions.push_back(std::move(index_instr));
              pack_instr.args.push_back(item_reg);
            }

            kernel_state.program.instructions.push_back(std::move(pack_instr));
            result = make_array_ref(array_slot, array_value.array_size);
            break;
          }
          case OpCode::Index:
          {
            if (instr.src_a >= reg_info.size() || instr.src_b >= reg_info.size())
            {
              return false;
            }
            if (reg_info[instr.src_a].kind != egress_graph_detail::FusedGraphValueKind::Array ||
                reg_info[instr.src_b].kind != egress_graph_detail::FusedGraphValueKind::Scalar)
            {
              return false;
            }
            egress_jit::NumericInstr numeric_instr;
            numeric_instr.op = egress_jit::NumericOp::IndexArray;
            numeric_instr.dst = dst_reg;
            numeric_instr.src_a = reg_info[instr.src_b].scalar_register;
            numeric_instr.slot_id = reg_info[instr.src_a].array_slot;
            kernel_state.program.instructions.push_back(std::move(numeric_instr));
            result = make_scalar_ref(dst_reg);
            break;
          }
          case OpCode::Add:
          case OpCode::Sub:
          case OpCode::Mul:
          case OpCode::Div:
          case OpCode::Mod:
          {
            if (instr.src_a >= reg_info.size() || instr.src_b >= reg_info.size())
            {
              return false;
            }
            const auto & lhs = reg_info[instr.src_a];
            const auto & rhs = reg_info[instr.src_b];
            if (lhs.kind == egress_graph_detail::FusedGraphValueKind::Array ||
                rhs.kind == egress_graph_detail::FusedGraphValueKind::Array)
            {
              egress_jit::NumericOp array_op = egress_jit::NumericOp::Add;
              if (instr.opcode == OpCode::Add)
              {
                if (lhs.kind == egress_graph_detail::FusedGraphValueKind::Array &&
                    rhs.kind == egress_graph_detail::FusedGraphValueKind::Array)
                {
                  if (lhs.array_size != rhs.array_size)
                  {
                    return false;
                  }
                  array_op = egress_jit::NumericOp::ArrayAdd;
                }
                else if (lhs.kind == egress_graph_detail::FusedGraphValueKind::Array)
                {
                  array_op = egress_jit::NumericOp::ArrayAddScalar;
                }
                else if (rhs.kind == egress_graph_detail::FusedGraphValueKind::Array)
                {
                  array_op = egress_jit::NumericOp::ArrayAddScalar;
                }
                else
                {
                  return false;
                }
              }
              else if (instr.opcode == OpCode::Sub)
              {
                if (lhs.kind == egress_graph_detail::FusedGraphValueKind::Array &&
                    rhs.kind == egress_graph_detail::FusedGraphValueKind::Array &&
                    lhs.array_size == rhs.array_size)
                {
                  array_op = egress_jit::NumericOp::ArraySub;
                }
                else
                {
                  return false;
                }
              }
              else if (instr.opcode == OpCode::Mul)
              {
                if (lhs.kind == egress_graph_detail::FusedGraphValueKind::Array &&
                    rhs.kind == egress_graph_detail::FusedGraphValueKind::Array)
                {
                  if (lhs.array_size != rhs.array_size)
                  {
                    return false;
                  }
                  array_op = egress_jit::NumericOp::ArrayMul;
                }
                else if (lhs.kind == egress_graph_detail::FusedGraphValueKind::Array)
                {
                  array_op = egress_jit::NumericOp::ArrayMulScalar;
                }
                else if (rhs.kind == egress_graph_detail::FusedGraphValueKind::Array)
                {
                  array_op = egress_jit::NumericOp::ArrayMulScalar;
                }
                else
                {
                  return false;
                }
              }
              else if (instr.opcode == OpCode::Div)
              {
                if (lhs.kind == egress_graph_detail::FusedGraphValueKind::Array &&
                    rhs.kind == egress_graph_detail::FusedGraphValueKind::Array)
                {
                  if (lhs.array_size != rhs.array_size)
                  {
                    return false;
                  }
                  array_op = egress_jit::NumericOp::ArrayDiv;
                }
                else if (lhs.kind == egress_graph_detail::FusedGraphValueKind::Array &&
                         rhs.kind == egress_graph_detail::FusedGraphValueKind::Scalar)
                {
                  array_op = egress_jit::NumericOp::ArrayDivScalar;
                }
                else
                {
                  return false;
                }
              }
              else if (instr.opcode == OpCode::Mod)
              {
                if (lhs.kind == egress_graph_detail::FusedGraphValueKind::Array &&
                    rhs.kind == egress_graph_detail::FusedGraphValueKind::Scalar)
                {
                  array_op = egress_jit::NumericOp::ArrayModScalar;
                }
                else
                {
                  return false;
                }
              }

              const uint32_t dst_slot = allocate_array_slot(
                lhs.kind == egress_graph_detail::FusedGraphValueKind::Array ? lhs.array_size : rhs.array_size);
              egress_jit::NumericInstr numeric_instr;
              numeric_instr.op = array_op;
              numeric_instr.dst = dst_slot;
              if (lhs.kind == egress_graph_detail::FusedGraphValueKind::Array)
              {
                numeric_instr.src_a = lhs.array_slot;
                numeric_instr.src_b =
                  rhs.kind == egress_graph_detail::FusedGraphValueKind::Array ? rhs.array_slot : rhs.scalar_register;
              }
              else
              {
                numeric_instr.src_a = rhs.array_slot;
                numeric_instr.src_b = lhs.scalar_register;
              }
              kernel_state.program.instructions.push_back(std::move(numeric_instr));
              result = make_array_ref(dst_slot,
                                      lhs.kind == egress_graph_detail::FusedGraphValueKind::Array ? lhs.array_size : rhs.array_size);
              break;
            }

            switch (instr.opcode)
            {
              case OpCode::Add: lower_binary_scalar(egress_jit::NumericOp::Add, dst_reg, lhs, rhs); break;
              case OpCode::Sub: lower_binary_scalar(egress_jit::NumericOp::Sub, dst_reg, lhs, rhs); break;
              case OpCode::Mul: lower_binary_scalar(egress_jit::NumericOp::Mul, dst_reg, lhs, rhs); break;
              case OpCode::Div: lower_binary_scalar(egress_jit::NumericOp::Div, dst_reg, lhs, rhs); break;
              case OpCode::Mod: lower_binary_scalar(egress_jit::NumericOp::Mod, dst_reg, lhs, rhs); break;
              default: break;
            }
            result = make_scalar_ref(dst_reg);
            break;
          }
          case OpCode::AddConst:
          case OpCode::MulConst:
          case OpCode::SubConstRhs:
          case OpCode::SubConstLhs:
          case OpCode::DivConstLhs:
          {
            if (instr.src_a >= reg_info.size())
            {
              return false;
            }
            const auto & src = reg_info[instr.src_a];
            const auto lit = load_const_rhs(instr.literal);
            if (instr.opcode == OpCode::AddConst)
            {
              if (src.kind == egress_graph_detail::FusedGraphValueKind::Array)
              {
                const uint32_t dst_slot = allocate_array_slot(src.array_size);
                egress_jit::NumericInstr numeric_instr;
                numeric_instr.op = egress_jit::NumericOp::ArrayAddScalar;
                numeric_instr.dst = dst_slot;
                numeric_instr.src_a = src.array_slot;
                numeric_instr.src_b = lit.scalar_register;
                kernel_state.program.instructions.push_back(std::move(numeric_instr));
                result = make_array_ref(dst_slot, src.array_size);
                break;
              }
              lower_binary_scalar(egress_jit::NumericOp::Add, dst_reg, src, lit);
              result = make_scalar_ref(dst_reg);
              break;
            }
            if (instr.opcode == OpCode::MulConst)
            {
              if (src.kind == egress_graph_detail::FusedGraphValueKind::Array)
              {
                const uint32_t dst_slot = allocate_array_slot(src.array_size);
                egress_jit::NumericInstr numeric_instr;
                numeric_instr.op = egress_jit::NumericOp::ArrayMulScalar;
                numeric_instr.dst = dst_slot;
                numeric_instr.src_a = src.array_slot;
                numeric_instr.src_b = lit.scalar_register;
                kernel_state.program.instructions.push_back(std::move(numeric_instr));
                result = make_array_ref(dst_slot, src.array_size);
                break;
              }
              lower_binary_scalar(egress_jit::NumericOp::Mul, dst_reg, src, lit);
              result = make_scalar_ref(dst_reg);
              break;
            }
            if (src.kind != egress_graph_detail::FusedGraphValueKind::Scalar)
            {
              return false;
            }
            if (instr.opcode == OpCode::SubConstRhs)
            {
              lower_binary_scalar(egress_jit::NumericOp::Sub, dst_reg, src, lit);
            }
            else if (instr.opcode == OpCode::SubConstLhs)
            {
              lower_binary_scalar(egress_jit::NumericOp::Sub, dst_reg, lit, src);
            }
            else
            {
              lower_binary_scalar(egress_jit::NumericOp::Div, dst_reg, lit, src);
            }
            result = make_scalar_ref(dst_reg);
            break;
          }
          case OpCode::Pow:
          case OpCode::FloorDiv:
          case OpCode::BitAnd:
          case OpCode::BitOr:
          case OpCode::BitXor:
          case OpCode::LShift:
          case OpCode::RShift:
          case OpCode::Less:
          case OpCode::LessEqual:
          case OpCode::Greater:
          case OpCode::GreaterEqual:
          case OpCode::Equal:
          case OpCode::NotEqual:
          {
            if (instr.src_a >= reg_info.size() || instr.src_b >= reg_info.size())
            {
              return false;
            }
            const auto & lhs = reg_info[instr.src_a];
            const auto & rhs = reg_info[instr.src_b];
            if (lhs.kind != egress_graph_detail::FusedGraphValueKind::Scalar ||
                rhs.kind != egress_graph_detail::FusedGraphValueKind::Scalar)
            {
              return false;
            }
            egress_jit::NumericOp op = egress_jit::NumericOp::Pow;
            switch (instr.opcode)
            {
              case OpCode::Pow: op = egress_jit::NumericOp::Pow; break;
              case OpCode::FloorDiv: op = egress_jit::NumericOp::FloorDiv; break;
              case OpCode::BitAnd: op = egress_jit::NumericOp::BitAnd; break;
              case OpCode::BitOr: op = egress_jit::NumericOp::BitOr; break;
              case OpCode::BitXor: op = egress_jit::NumericOp::BitXor; break;
              case OpCode::LShift: op = egress_jit::NumericOp::LShift; break;
              case OpCode::RShift: op = egress_jit::NumericOp::RShift; break;
              case OpCode::Less: op = egress_jit::NumericOp::Less; break;
              case OpCode::LessEqual: op = egress_jit::NumericOp::LessEqual; break;
              case OpCode::Greater: op = egress_jit::NumericOp::Greater; break;
              case OpCode::GreaterEqual: op = egress_jit::NumericOp::GreaterEqual; break;
              case OpCode::Equal: op = egress_jit::NumericOp::Equal; break;
              case OpCode::NotEqual: op = egress_jit::NumericOp::NotEqual; break;
              default: break;
            }
            lower_binary_scalar(op, dst_reg, lhs, rhs);
            result = make_scalar_ref(dst_reg);
            break;
          }
          case OpCode::Abs:
          case OpCode::Log:
          case OpCode::Neg:
          case OpCode::Not:
          case OpCode::BitNot:
          case OpCode::Sin:
          {
            if (instr.src_a >= reg_info.size() ||
                reg_info[instr.src_a].kind != egress_graph_detail::FusedGraphValueKind::Scalar)
            {
              return false;
            }
            egress_jit::NumericInstr numeric_instr;
            switch (instr.opcode)
            {
              case OpCode::Abs: numeric_instr.op = egress_jit::NumericOp::Abs; break;
              case OpCode::Log: numeric_instr.op = egress_jit::NumericOp::Log; break;
              case OpCode::Neg: numeric_instr.op = egress_jit::NumericOp::Neg; break;
              case OpCode::Not: numeric_instr.op = egress_jit::NumericOp::Not; break;
              case OpCode::BitNot: numeric_instr.op = egress_jit::NumericOp::BitNot; break;
              case OpCode::Sin: numeric_instr.op = egress_jit::NumericOp::Sin; break;
              default: break;
            }
            numeric_instr.dst = dst_reg;
            numeric_instr.src_a = reg_info[instr.src_a].scalar_register;
            kernel_state.program.instructions.push_back(std::move(numeric_instr));
            result = make_scalar_ref(dst_reg);
            break;
          }
          case OpCode::Clamp:
          {
            if (instr.src_a >= reg_info.size() ||
                instr.src_b >= reg_info.size() ||
                instr.src_c >= reg_info.size())
            {
              return false;
            }
            const auto & value = reg_info[instr.src_a];
            const auto & lo = reg_info[instr.src_b];
            const auto & hi = reg_info[instr.src_c];
            if (value.kind != egress_graph_detail::FusedGraphValueKind::Scalar ||
                lo.kind != egress_graph_detail::FusedGraphValueKind::Scalar ||
                hi.kind != egress_graph_detail::FusedGraphValueKind::Scalar)
            {
              return false;
            }
            egress_jit::NumericInstr numeric_instr;
            numeric_instr.op = egress_jit::NumericOp::Clamp;
            numeric_instr.dst = dst_reg;
            numeric_instr.src_a = value.scalar_register;
            numeric_instr.src_b = lo.scalar_register;
            numeric_instr.src_c = hi.scalar_register;
            kernel_state.program.instructions.push_back(std::move(numeric_instr));
            result = make_scalar_ref(dst_reg);
            break;
          }
          case OpCode::MatMul:
            return false;
        }

        reg_info[instr.dst] = result;
      }

      if (!bind_result(program_index, *program, reg_info))
      {
        return false;
      }
    }

    auto & jit = egress_jit::OrcJitEngine::instance();
    if (!jit.available())
    {
      kernel_state.status = jit.init_error();
      return false;
    }
    auto kernel_or_err = jit.compile_numeric_program(kernel_state.program);
    if (!kernel_or_err)
    {
      kernel_state.status = llvm::toString(kernel_or_err.takeError());
      return false;
    }
    kernel_state.kernel = *kernel_or_err;
    kernel_state.available = true;
    kernel_state.status = "numeric JIT active";
    kernel_state.param_ptrs.clear();
    {
      std::unordered_map<uint64_t, uint32_t> seen;
      for (const auto & instr : kernel_state.program.instructions)
      {
        if (instr.op == egress_jit::NumericOp::SmoothedParam && instr.param_ptr != 0 &&
            seen.find(instr.param_ptr) == seen.end())
        {
          seen.emplace(instr.param_ptr, static_cast<uint32_t>(kernel_state.param_ptrs.size()));
          kernel_state.param_ptrs.push_back(instr.param_ptr);
        }
      }
    }
    if (kernel_state.scalar_inputs.size() < fused->source_outputs.size())
    {
      kernel_state.scalar_inputs.resize(fused->source_outputs.size(), 0.0);
    }
    kernel_state.temps.assign(kernel_state.program.register_count, 0.0);
    kernel_state.int_temps.assign(kernel_state.program.register_count, 0);
    kernel_state.array_ptrs.resize(kernel_state.array_storage.size(), nullptr);
    kernel_state.array_sizes.resize(kernel_state.array_storage.size(), 0);
    for (std::size_t array_id = 0; array_id < kernel_state.array_storage.size(); ++array_id)
    {
      kernel_state.array_ptrs[array_id] =
        kernel_state.array_storage[array_id].empty() ? nullptr : kernel_state.array_storage[array_id].data();
      kernel_state.array_sizes[array_id] = static_cast<uint64_t>(kernel_state.array_storage[array_id].size());
    }
    return true;
  };

  bool primitive_body_inputs_ok = true;
  std::vector<const CompiledInputProgram *> input_programs;
  std::vector<uint32_t> input_program_module_ids;
  input_programs.reserve(runtime.modules.size());
  input_program_module_ids.reserve(runtime.modules.size());
  for (uint32_t module_id = 0; module_id < runtime.modules.size(); ++module_id)
  {
    const auto & slot = runtime.modules[module_id];
    if (!slot.module)
    {
      continue;
    }
    input_programs.push_back(&slot.input_program);
    input_program_module_ids.push_back(module_id);
  }
  if (!build_expression_kernel(
        input_programs,
        true,
        fused->input_kernel,
        [&](std::size_t program_index, const CompiledInputProgram & program, const auto & reg_info) {
          if (program_index >= input_program_module_ids.size())
          {
            return false;
          }
          const uint32_t module_id = input_program_module_ids[program_index];
          for (unsigned int input_id = 0; input_id < program.result_registers.size(); ++input_id)
          {
            const uint32_t result_reg = program.result_registers[input_id];
            if (result_reg >= reg_info.size())
            {
              return false;
            }
            egress_graph_detail::FusedGraphInputBinding binding;
            binding.module_id = module_id;
            binding.input_id = input_id;
            if (module_id < primitive_input_base_by_module.size())
            {
              const uint32_t base = primitive_input_base_by_module[module_id];
              if (base != std::numeric_limits<uint32_t>::max())
              {
                if (reg_info[result_reg].kind == egress_graph_detail::FusedGraphValueKind::Scalar)
                {
                  binding.fused_input_slot = base + input_id;
                }
                else if (input_id < primitive_array_input_slots_by_module[module_id].size())
                {
                  const uint32_t array_slot = primitive_array_input_slots_by_module[module_id][input_id];
                  if (array_slot == std::numeric_limits<uint32_t>::max() ||
                      reg_info[result_reg].kind != egress_graph_detail::FusedGraphValueKind::Array)
                  {
                    primitive_body_inputs_ok = false;
                  }
                  else
                  {
                    binding.fused_input_array_slot = array_slot;
                  }
                }
                else
                {
                  primitive_body_inputs_ok = false;
                }
              }
            }
            binding.value.kind = reg_info[result_reg].kind;
            binding.value.scalar_register = reg_info[result_reg].scalar_register;
            binding.value.array_slot = reg_info[result_reg].array_slot;
            binding.value.array_size = reg_info[result_reg].array_size;
            fused->input_kernel.input_bindings.push_back(std::move(binding));
          }
          return true;
        }))
  {
    mark_ineligible("graph fusion input kernel build failed");
    return fused;
  }
  if (!primitive_body_inputs_ok)
  {
    mark_primitive_body_unavailable("primitive body fusion requires scalar top-level input bindings");
  }

  std::vector<const CompiledInputProgram *> mix_programs;
  mix_programs.reserve(runtime.mix_exprs.size());
  for (const auto & mix_expr : runtime.mix_exprs)
  {
    mix_programs.push_back(&mix_expr.program);
  }
  if (!build_expression_kernel(
        mix_programs,
        false,
        fused->mix_kernel,
        [&](std::size_t, const CompiledInputProgram & program, const auto & reg_info) {
          if (program.result_registers.empty())
          {
            return true;
          }
          const uint32_t result_reg = program.result_registers.front();
          if (result_reg >= reg_info.size() ||
              reg_info[result_reg].kind != egress_graph_detail::FusedGraphValueKind::Scalar)
          {
            return false;
          }
          egress_graph_detail::FusedGraphMixBinding binding;
          binding.value.kind = reg_info[result_reg].kind;
          binding.value.scalar_register = reg_info[result_reg].scalar_register;
          binding.value.array_slot = reg_info[result_reg].array_slot;
          binding.value.array_size = reg_info[result_reg].array_size;
          fused->mix_kernel.mix_bindings.push_back(std::move(binding));
          return true;
        }))
  {
    fused->mix_kernel = egress_graph_detail::FusedGraphKernelState{};
  }
 #endif

  fused->numeric_candidate = true;
  fused->candidate_reason = "graph-level lowering candidate";
  return fused;
}

bool Graph::supports_fused_numeric_opcode(OpCode opcode)
{
  switch (opcode)
  {
    case OpCode::Literal:
    case OpCode::Ref:
    case OpCode::RefIndex:
    case OpCode::ArrayPack:
    case OpCode::Index:
    case OpCode::Add:
    case OpCode::AddConst:
    case OpCode::Sub:
    case OpCode::SubConstRhs:
    case OpCode::SubConstLhs:
    case OpCode::Mul:
    case OpCode::MulConst:
    case OpCode::Div:
    case OpCode::DivConstLhs:
    case OpCode::MatMul:
    case OpCode::Pow:
    case OpCode::Mod:
    case OpCode::FloorDiv:
    case OpCode::BitAnd:
    case OpCode::BitOr:
    case OpCode::BitXor:
    case OpCode::LShift:
    case OpCode::RShift:
    case OpCode::Abs:
    case OpCode::Clamp:
    case OpCode::Log:
    case OpCode::Neg:
    case OpCode::Not:
    case OpCode::BitNot:
    case OpCode::Sin:
    case OpCode::Less:
    case OpCode::LessEqual:
    case OpCode::Greater:
    case OpCode::GreaterEqual:
    case OpCode::Equal:
    case OpCode::NotEqual:
      return true;
    case OpCode::ArraySet:
      return true;
  }
  return false;
}

bool Graph::is_fused_numeric_candidate(const CompiledInputProgram & program)
{
  for (const auto & instr : program.instructions)
  {
    if (!supports_fused_numeric_opcode(instr.opcode))
    {
      return false;
    }
  }
  return true;
}

bool Graph::program_uses_current_outputs(const CompiledInputProgram & program)
{
  for (const auto & instr : program.instructions)
  {
    if (instr.opcode == OpCode::Ref || instr.opcode == OpCode::RefIndex)
    {
      return true;
    }
  }
  return false;
}

void Graph::rebuild_and_publish_runtime_locked()
{
  const uint32_t active = active_runtime_index_.load(std::memory_order_acquire);
  const uint32_t inactive = 1U - active;
  wait_for_runtime_available(inactive);
  runtimes_[inactive] = build_runtime_locked();
  active_runtime_index_.store(inactive, std::memory_order_release);
}

void Graph::eval_instruction(const RuntimeState & runtime, const ExprInstr & instr, Value * registers)
{
  auto get_fused_source_slot = [&](uint32_t module_id, unsigned int output_id) -> uint32_t
  {
    if (runtime.fused_graph == nullptr)
    {
      return std::numeric_limits<uint32_t>::max();
    }
    const auto it = runtime.fused_graph->source_output_lookup.find(fused_source_output_key(module_id, output_id));
    return it == runtime.fused_graph->source_output_lookup.end()
      ? std::numeric_limits<uint32_t>::max()
      : it->second;
  };

  switch (instr.opcode)
  {
    case OpCode::Literal:
      registers[instr.dst] = instr.literal;
      break;
    case OpCode::Ref:
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      const auto & slot = runtime.modules[instr.ref_module_id];
      const uint32_t source_slot = get_fused_source_slot(instr.ref_module_id, instr.ref_output_id);
#ifdef EGRESS_LLVM_ORC_JIT
      if (slot.module != nullptr)
      {
        double scalar = 0.0;
        if (slot.module->try_get_numeric_scalar_output(instr.ref_output_id, true, scalar))
        {
          registers[instr.dst] = float_value(scalar);
          break;
        }
        const auto * numeric_values = slot.module->try_get_numeric_output_array_values(instr.ref_output_id, true);
        if (numeric_values != nullptr)
        {
          std::vector<Value> items(numeric_values->size(), float_value(0.0));
          for (std::size_t item_id = 0; item_id < numeric_values->size(); ++item_id)
          {
            items[item_id] = float_value((*numeric_values)[item_id]);
          }
          registers[instr.dst] = array_value(std::move(items));
          break;
        }
      }
#endif
      if (runtime.fused_graph != nullptr &&
          source_slot < runtime.fused_graph->prev_outputs.size())
      {
        registers[instr.dst] = runtime.fused_graph->prev_outputs[source_slot];
        break;
      }
      if (!slot.module || instr.ref_output_id >= slot.module->prev_outputs.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      registers[instr.dst] = slot.module->materialize_output_value(instr.ref_output_id, true);
      break;
    }
    case OpCode::RefIndex:
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      const auto & slot = runtime.modules[instr.ref_module_id];
      const uint32_t source_slot = get_fused_source_slot(instr.ref_module_id, instr.ref_output_id);
      if (runtime.fused_graph != nullptr &&
          source_slot < runtime.fused_graph->indexed_prev_values.size() &&
          instr.src_a < runtime.fused_graph->indexed_prev_values[source_slot].size())
      {
        registers[instr.dst] = runtime.fused_graph->indexed_prev_values[source_slot][instr.src_a];
        break;
      }
      if (!slot.module || instr.ref_output_id >= slot.module->prev_outputs.size() || instr.ref_index < 0)
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      if (instr.ref_output_id < slot.indexed_prev_output_values.size() &&
           instr.src_a < slot.indexed_prev_output_values[instr.ref_output_id].size())
      {
        registers[instr.dst] = slot.indexed_prev_output_values[instr.ref_output_id][instr.src_a];
        break;
      }
      const Value * output_ptr = &slot.module->materialize_output_value(instr.ref_output_id, true);
      if (runtime.fused_graph != nullptr &&
          source_slot < runtime.fused_graph->prev_outputs.size())
      {
        output_ptr = &runtime.fused_graph->prev_outputs[source_slot];
      }
      const Value & output = *output_ptr;
      const std::size_t item_index = static_cast<std::size_t>(instr.ref_index);
      if (expr::is_array(output))
      {
        if (item_index >= output.array_items.size())
        {
          throw std::out_of_range("Array index out of range.");
        }
        registers[instr.dst] = output.array_items[item_index];
        break;
      }
      if (expr::is_matrix(output))
      {
        registers[instr.dst] = expr::array_from_matrix_row(output, item_index);
        break;
      }
      registers[instr.dst] = float_value(0.0);
      break;
    }
    case OpCode::ArrayPack:
    {
      std::vector<Value> items;
      items.reserve(instr.args.size());
      for (uint32_t src : instr.args)
      {
        if (expr::is_array(registers[src]) || expr::is_matrix(registers[src]))
        {
          throw std::invalid_argument("Nested arrays are not supported.");
        }
        items.push_back(registers[src]);
      }
      registers[instr.dst] = expr::array_value(std::move(items));
      break;
    }
    case OpCode::Index:
    {
      const Value & array_value = registers[instr.src_a];
      const int64_t index = expr::to_int64(registers[instr.src_b]);
      if (index < 0)
      {
        throw std::out_of_range("Array index out of range.");
      }
      if (expr::is_array(array_value))
      {
        if (static_cast<std::size_t>(index) >= array_value.array_items.size())
        {
          throw std::out_of_range("Array index out of range.");
        }
        registers[instr.dst] = array_value.array_items[static_cast<std::size_t>(index)];
        break;
      }
      if (expr::is_matrix(array_value))
      {
        registers[instr.dst] = expr::array_from_matrix_row(array_value, static_cast<std::size_t>(index));
        break;
      }
      registers[instr.dst] = float_value(0.0);
      break;
    }
    case OpCode::ArraySet:
      registers[instr.dst] = expr_eval::array_set_value(
        registers[instr.src_a],
        registers[instr.src_b],
        registers[instr.src_c]);
      break;
    case OpCode::Add:
      registers[instr.dst] = expr_eval::add_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::AddConst:
      registers[instr.dst] = expr_eval::add_values(registers[instr.src_a], instr.literal);
      break;
    case OpCode::Sub:
      registers[instr.dst] = expr_eval::sub_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::SubConstRhs:
      registers[instr.dst] = expr_eval::sub_values(registers[instr.src_a], instr.literal);
      break;
    case OpCode::SubConstLhs:
      registers[instr.dst] = expr_eval::sub_values(instr.literal, registers[instr.src_a]);
      break;
    case OpCode::Mul:
      registers[instr.dst] = expr_eval::mul_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::MulConst:
      registers[instr.dst] = expr_eval::mul_values(registers[instr.src_a], instr.literal);
      break;
    case OpCode::Div:
      registers[instr.dst] = expr_eval::div_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::MatMul:
      registers[instr.dst] = expr_eval::matmul_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::DivConstLhs:
      registers[instr.dst] = expr_eval::div_values(instr.literal, registers[instr.src_a]);
      break;
    case OpCode::Mod:
      registers[instr.dst] = expr_eval::mod_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::FloorDiv:
      registers[instr.dst] = expr_eval::floor_div_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::BitAnd:
      registers[instr.dst] = expr_eval::bit_and_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::BitOr:
      registers[instr.dst] = expr_eval::bit_or_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::BitXor:
      registers[instr.dst] = expr_eval::bit_xor_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::LShift:
      registers[instr.dst] = expr_eval::lshift_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::RShift:
      registers[instr.dst] = expr_eval::rshift_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::Abs:
      registers[instr.dst] = expr_eval::abs_value(registers[instr.src_a]);
      break;
    case OpCode::Clamp:
      registers[instr.dst] = expr_eval::clamp_values(registers[instr.src_a], registers[instr.src_b], registers[instr.src_c]);
      break;
    case OpCode::Log:
      registers[instr.dst] = expr_eval::log_value(registers[instr.src_a]);
      break;
    case OpCode::Neg:
      registers[instr.dst] = expr_eval::neg_value(registers[instr.src_a]);
      break;
    case OpCode::Not:
      registers[instr.dst] = expr_eval::not_value(registers[instr.src_a]);
      break;
    case OpCode::BitNot:
      registers[instr.dst] = expr_eval::bit_not_value(registers[instr.src_a]);
      break;
    case OpCode::Sin:
      registers[instr.dst] = expr_eval::sin_value(registers[instr.src_a]);
      break;
    case OpCode::Pow:
      registers[instr.dst] = expr_eval::pow_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::Less:
      registers[instr.dst] = expr_eval::less_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::LessEqual:
      registers[instr.dst] = expr_eval::less_equal_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::Greater:
      registers[instr.dst] = expr_eval::greater_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::GreaterEqual:
      registers[instr.dst] = expr_eval::greater_equal_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::Equal:
      registers[instr.dst] = expr_eval::equal_values(registers[instr.src_a], registers[instr.src_b]);
      break;
    case OpCode::NotEqual:
      registers[instr.dst] = expr_eval::not_equal_values(registers[instr.src_a], registers[instr.src_b]);
      break;
  }
}

void Graph::eval_mix_instruction(const RuntimeState & runtime, const ExprInstr & instr, Value * registers)
{
  auto get_fused_source_slot = [&](uint32_t module_id, unsigned int output_id) -> uint32_t
  {
    if (runtime.fused_graph == nullptr)
    {
      return std::numeric_limits<uint32_t>::max();
    }
    const auto it = runtime.fused_graph->source_output_lookup.find(fused_source_output_key(module_id, output_id));
    return it == runtime.fused_graph->source_output_lookup.end()
      ? std::numeric_limits<uint32_t>::max()
      : it->second;
  };

  switch (instr.opcode)
  {
    case OpCode::Ref:
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      const auto & slot = runtime.modules[instr.ref_module_id];
#ifdef EGRESS_LLVM_ORC_JIT
      if (slot.module != nullptr)
      {
        double numeric_scalar = 0.0;
        if (slot.module->try_get_numeric_scalar_output(instr.ref_output_id, false, numeric_scalar))
        {
          registers[instr.dst] = float_value(numeric_scalar);
          break;
        }
      }
#endif
      const uint32_t source_slot = get_fused_source_slot(instr.ref_module_id, instr.ref_output_id);
      if (runtime.fused_graph != nullptr &&
          source_slot < runtime.fused_graph->current_outputs.size())
      {
        registers[instr.dst] = runtime.fused_graph->current_outputs[source_slot];
        break;
      }
      if (!slot.module || instr.ref_output_id >= slot.module->outputs.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      registers[instr.dst] = slot.module->materialize_output_value(instr.ref_output_id, false);
      break;
    }
    case OpCode::RefIndex:
    {
      if (instr.ref_module_id >= runtime.modules.size())
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
      const auto & slot = runtime.modules[instr.ref_module_id];
#ifdef EGRESS_LLVM_ORC_JIT
      if (slot.module != nullptr && instr.ref_index >= 0)
      {
        const auto * numeric_values = slot.module->try_get_numeric_output_array_values(instr.ref_output_id);
        if (numeric_values != nullptr)
        {
          if (static_cast<std::size_t>(instr.ref_index) >= numeric_values->size())
          {
            throw std::out_of_range("Array index out of range.");
          }
          registers[instr.dst] =
            expr::float_value((*numeric_values)[static_cast<std::size_t>(instr.ref_index)]);
          break;
        }
      }
#endif
      const uint32_t source_slot = get_fused_source_slot(instr.ref_module_id, instr.ref_output_id);
      if (runtime.fused_graph != nullptr &&
          source_slot < runtime.fused_graph->current_outputs.size())
      {
        const Value & output = runtime.fused_graph->current_outputs[source_slot];
        const std::size_t item_index = static_cast<std::size_t>(instr.ref_index);
        if (expr::is_array(output))
        {
          if (item_index >= output.array_items.size())
          {
            throw std::out_of_range("Array index out of range.");
          }
          registers[instr.dst] = output.array_items[item_index];
          break;
        }
        if (expr::is_matrix(output))
        {
          registers[instr.dst] = expr::array_from_matrix_row(output, item_index);
          break;
        }
      }
      if (!slot.module || instr.ref_output_id >= slot.module->outputs.size() || instr.ref_index < 0)
      {
        registers[instr.dst] = float_value(0.0);
        break;
      }
#ifdef EGRESS_LLVM_ORC_JIT
      const auto * numeric_values = slot.module->try_get_numeric_output_array_values(instr.ref_output_id);
      if (numeric_values != nullptr)
      {
        if (static_cast<std::size_t>(instr.ref_index) >= numeric_values->size())
        {
          throw std::out_of_range("Array index out of range.");
        }
        registers[instr.dst] =
          expr::float_value((*numeric_values)[static_cast<std::size_t>(instr.ref_index)]);
        break;
      }
#endif
      const Value & output = slot.module->materialize_output_value(instr.ref_output_id, false);
      const std::size_t item_index = static_cast<std::size_t>(instr.ref_index);
      if (expr::is_array(output))
      {
        if (item_index >= output.array_items.size())
        {
          throw std::out_of_range("Array index out of range.");
        }
        registers[instr.dst] = output.array_items[item_index];
        break;
      }
      if (expr::is_matrix(output))
      {
        registers[instr.dst] = expr::array_from_matrix_row(output, item_index);
        break;
      }
      registers[instr.dst] = float_value(0.0);
      break;
    }
    default:
      eval_instruction(runtime, instr, registers);
      break;
  }
}

uint32_t Graph::compile_expr_node(
  const ExprSpecPtr & expr,
  CompiledInputProgram & compiled,
  const RuntimeState & runtime) const
{
  if (!expr)
  {
    ExprInstr instr;
    instr.opcode = OpCode::Literal;
    instr.dst = compiled.register_count++;
    instr.literal = float_value(0.0);
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Literal)
  {
    ExprInstr instr;
    instr.opcode = OpCode::Literal;
    instr.dst = compiled.register_count++;
    instr.literal = expr->literal;
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Ref)
  {
    auto it = runtime.name_to_id.find(expr->module_name);
    if (it == runtime.name_to_id.end())
    {
      ExprInstr instr;
      instr.opcode = OpCode::Literal;
      instr.dst = compiled.register_count++;
      instr.literal = float_value(0.0);
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    ExprInstr instr;
    instr.opcode = OpCode::Ref;
    instr.dst = compiled.register_count++;
    instr.ref_module_id = it->second;
    instr.ref_output_id = expr->output_id;
    compiled.instructions.push_back(instr);

    return instr.dst;
  }

  if (expr->kind == ExprKind::ArrayPack)
  {
    ExprInstr instr;
    instr.opcode = OpCode::ArrayPack;
    instr.dst = compiled.register_count++;
    instr.args.reserve(expr->args.size());
    for (const auto & arg : expr->args)
    {
      instr.args.push_back(compile_expr_node(arg, compiled, runtime));
    }
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Index)
  {
    if (expr->lhs && expr->lhs->kind == ExprKind::Ref &&
        expr->rhs && expr->rhs->kind == ExprKind::Literal &&
        expr->rhs->literal.type != ValueType::Array &&
        expr->rhs->literal.type != ValueType::Matrix)
    {
      const int64_t raw_index = expr::to_int64(expr->rhs->literal);
      if (raw_index >= 0)
      {
        auto it = runtime.name_to_id.find(expr->lhs->module_name);
        if (it != runtime.name_to_id.end())
        {
          ExprInstr instr;
          instr.opcode = OpCode::RefIndex;
          instr.dst = compiled.register_count++;
          instr.ref_module_id = it->second;
          instr.ref_output_id = expr->lhs->output_id;
          instr.ref_index = raw_index;
          instr.src_a = std::numeric_limits<uint32_t>::max();
          compiled.instructions.push_back(instr);
          return instr.dst;
        }
      }
    }

    ExprInstr instr;
    instr.opcode = OpCode::Index;
    instr.dst = compiled.register_count++;
    instr.src_a = compile_expr_node(expr->lhs, compiled, runtime);
    instr.src_b = compile_expr_node(expr->rhs, compiled, runtime);
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::ArraySet)
  {
    ExprInstr instr;
    instr.opcode = OpCode::ArraySet;
    instr.dst = compiled.register_count++;
    instr.src_a = compile_expr_node(expr->lhs, compiled, runtime);
    instr.src_b = compile_expr_node(expr->rhs, compiled, runtime);
    instr.src_c = compile_expr_node(expr->args.empty() ? nullptr : expr->args.front(), compiled, runtime);
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Abs ||
      expr->kind == ExprKind::Neg ||
      expr->kind == ExprKind::Not ||
      expr->kind == ExprKind::BitNot ||
      expr->kind == ExprKind::Log ||
      expr->kind == ExprKind::Sin)
  {
    const uint32_t operand = compile_expr_node(expr->lhs, compiled, runtime);
    ExprInstr instr;
    instr.opcode = expr->kind == ExprKind::Abs
                     ? OpCode::Abs
                     : (expr->kind == ExprKind::Neg
                          ? OpCode::Neg
                          : (expr->kind == ExprKind::Not
                               ? OpCode::Not
                               : (expr->kind == ExprKind::BitNot
                                    ? OpCode::BitNot
                                    : (expr->kind == ExprKind::Log ? OpCode::Log : OpCode::Sin))));
    instr.dst = compiled.register_count++;
    instr.src_a = operand;
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Clamp)
  {
    ExprInstr instr;
    instr.opcode = OpCode::Clamp;
    instr.dst = compiled.register_count++;
    instr.src_a = compile_expr_node(expr->lhs, compiled, runtime);
    instr.src_b = compile_expr_node(expr->rhs, compiled, runtime);
    instr.src_c = compile_expr_node(expr->args.empty() ? nullptr : expr->args.front(), compiled, runtime);
    compiled.instructions.push_back(instr);
    return instr.dst;
  }

  if (expr->kind == ExprKind::Add)
  {
    if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
    {
      const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::AddConst;
      instr.dst = compiled.register_count++;
      instr.src_a = rhs;
      instr.literal = expr->lhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
    {
      const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::AddConst;
      instr.dst = compiled.register_count++;
      instr.src_a = lhs;
      instr.literal = expr->rhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }
  }

  if (expr->kind == ExprKind::Mul)
  {
    if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
    {
      const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::MulConst;
      instr.dst = compiled.register_count++;
      instr.src_a = rhs;
      instr.literal = expr->lhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
    {
      const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::MulConst;
      instr.dst = compiled.register_count++;
      instr.src_a = lhs;
      instr.literal = expr->rhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }
  }

  if (expr->kind == ExprKind::Sub)
  {
    if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
    {
      const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::SubConstRhs;
      instr.dst = compiled.register_count++;
      instr.src_a = lhs;
      instr.literal = expr->rhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
    {
      const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::SubConstLhs;
      instr.dst = compiled.register_count++;
      instr.src_a = rhs;
      instr.literal = expr->lhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }
  }

  if (expr->kind == ExprKind::Div)
  {
    if (expr->rhs && expr->rhs->kind == ExprKind::Literal)
    {
      const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::MulConst;
      instr.dst = compiled.register_count++;
      instr.src_a = lhs;
      instr.literal = to_float64(expr->rhs->literal) == 0.0
                        ? float_value(0.0)
                        : float_value(1.0 / to_float64(expr->rhs->literal));
      compiled.instructions.push_back(instr);
      return instr.dst;
    }

    if (expr->lhs && expr->lhs->kind == ExprKind::Literal)
    {
      const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);
      ExprInstr instr;
      instr.opcode = OpCode::DivConstLhs;
      instr.dst = compiled.register_count++;
      instr.src_a = rhs;
      instr.literal = expr->lhs->literal;
      compiled.instructions.push_back(instr);
      return instr.dst;
    }
  }

  const uint32_t lhs = compile_expr_node(expr->lhs, compiled, runtime);
  const uint32_t rhs = compile_expr_node(expr->rhs, compiled, runtime);

  ExprInstr instr;
  switch (expr->kind)
  {
    case ExprKind::Add:
      instr.opcode = OpCode::Add;
      break;
    case ExprKind::Sub:
      instr.opcode = OpCode::Sub;
      break;
    case ExprKind::Mul:
      instr.opcode = OpCode::Mul;
      break;
    case ExprKind::Div:
      instr.opcode = OpCode::Div;
      break;
    case ExprKind::MatMul:
      instr.opcode = OpCode::MatMul;
      break;
    case ExprKind::Pow:
      instr.opcode = OpCode::Pow;
      break;
    case ExprKind::Mod:
      instr.opcode = OpCode::Mod;
      break;
    case ExprKind::FloorDiv:
      instr.opcode = OpCode::FloorDiv;
      break;
    case ExprKind::BitAnd:
      instr.opcode = OpCode::BitAnd;
      break;
    case ExprKind::BitOr:
      instr.opcode = OpCode::BitOr;
      break;
    case ExprKind::BitXor:
      instr.opcode = OpCode::BitXor;
      break;
    case ExprKind::LShift:
      instr.opcode = OpCode::LShift;
      break;
    case ExprKind::RShift:
      instr.opcode = OpCode::RShift;
      break;
    case ExprKind::Less:
      instr.opcode = OpCode::Less;
      break;
    case ExprKind::LessEqual:
      instr.opcode = OpCode::LessEqual;
      break;
    case ExprKind::Greater:
      instr.opcode = OpCode::Greater;
      break;
    case ExprKind::GreaterEqual:
      instr.opcode = OpCode::GreaterEqual;
      break;
    case ExprKind::Equal:
      instr.opcode = OpCode::Equal;
      break;
    case ExprKind::NotEqual:
      instr.opcode = OpCode::NotEqual;
      break;
    default:
      instr.opcode = OpCode::Literal;
      instr.literal = float_value(0.0);
      break;
  }
  instr.dst = compiled.register_count++;
  instr.src_a = lhs;
  instr.src_b = rhs;
  compiled.instructions.push_back(instr);
  return instr.dst;
}

Graph::CompiledInputProgram Graph::compile_input_program(
  const std::vector<ExprSpecPtr> & exprs,
  unsigned int input_count,
  const RuntimeState & runtime) const
{
  CompiledInputProgram compiled;
  compiled.result_registers.assign(input_count, 0);

  for (unsigned int input_id = 0; input_id < input_count; ++input_id)
  {
    const ExprSpecPtr expr = input_id < exprs.size() ? exprs[input_id] : nullptr;
    ExprSpecPtr inlined = egress_expr_inline::inline_functions(expr);
    compiled.result_registers[input_id] = compile_expr_node(inlined, compiled, runtime);
  }

  return compiled;
}

void Graph::eval_input_program(
  const RuntimeState & runtime,
  const CompiledInputProgram & program,
  std::vector<Value> & registers,
  std::vector<Value> & inputs) const
{
  if (program.instructions.empty())
  {
    for (auto & input : inputs)
    {
      input = float_value(0.0);
    }
    return;
  }

  if (registers.size() < program.register_count)
  {
    registers.resize(program.register_count, float_value(0.0));
  }

  for (const auto & instr : program.instructions)
  {
    eval_instruction(runtime, instr, registers.data());
  }

  const unsigned int input_limit = std::min(
    static_cast<unsigned int>(inputs.size()),
    static_cast<unsigned int>(program.result_registers.size()));

  for (unsigned int input_id = 0; input_id < input_limit; ++input_id)
  {
    inputs[input_id] = registers[program.result_registers[input_id]];
  }
}
