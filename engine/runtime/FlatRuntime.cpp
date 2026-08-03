#include "runtime/FlatRuntime.hpp"
#include "runtime/NumericProgramParser.hpp"

#include <llvm/ADT/ArrayRef.h>
#include <llvm/Support/SHA256.h>

#ifdef TROPICAL_METAL
#include "metal/MetalKernel.hpp"
#endif

#include <algorithm>
#include <bit>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <unordered_set>

namespace tropical_runtime
{

namespace
{
constexpr std::size_t kImmutableAssetAlignment = 64;

std::size_t align_asset(std::size_t value)
{
  return (value + kImmutableAssetAlignment - 1)
       / kImmutableAssetAlignment * kImmutableAssetAlignment;
}

std::string sha256_hex(const std::vector<uint8_t> & bytes)
{
  const auto digest = llvm::SHA256::hash(
    llvm::ArrayRef<uint8_t>(bytes.data(), bytes.size()));
  std::ostringstream out;
  out << std::hex << std::setfill('0');
  for (uint8_t byte : digest)
    out << std::setw(2) << static_cast<unsigned>(byte);
  return out.str();
}

bool valid_sha256(const std::string & value)
{
  return value.size() == 64
      && std::all_of(value.begin(), value.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}

bool path_is_within(const std::filesystem::path & root,
                    const std::filesystem::path & candidate)
{
  auto root_it = root.begin();
  auto candidate_it = candidate.begin();
  for (; root_it != root.end(); ++root_it, ++candidate_it)
    if (candidate_it == candidate.end() || *root_it != *candidate_it)
      return false;
  return true;
}

std::filesystem::path resolve_asset_path(const std::string & text)
{
  if (text.empty())
    throw std::runtime_error("FlatRuntime: immutable asset path is empty");
  const std::filesystem::path relative(text);
  if (relative.is_absolute())
    throw std::runtime_error(
      "FlatRuntime: immutable asset path must be package-relative: '" + text + "'");
  for (const auto & component : relative)
    if (component == "..")
      throw std::runtime_error(
        "FlatRuntime: immutable asset path may not contain '..': '" + text + "'");

  const auto root = std::filesystem::canonical(std::filesystem::current_path());
  std::error_code ec;
  const auto candidate = std::filesystem::weakly_canonical(root / relative, ec);
  if (ec || !path_is_within(root, candidate))
    throw std::runtime_error(
      "FlatRuntime: immutable asset path escapes package root: '" + text + "'");
  return candidate;
}

std::vector<uint8_t> read_asset(
  const tropical_plan5::ParsedPlan5::ImmutableAsset & asset)
{
  if (asset.byte_count == 0
      || asset.byte_count > std::numeric_limits<std::size_t>::max()
      || asset.byte_count >
           static_cast<uint64_t>(std::numeric_limits<std::streamsize>::max()))
    throw std::runtime_error(
      "FlatRuntime: immutable asset byte_count is invalid for '" + asset.path + "'");
  if (!valid_sha256(asset.sha256))
    throw std::runtime_error(
      "FlatRuntime: immutable asset SHA-256 must be 64 lowercase hex digits");
  const auto path = resolve_asset_path(asset.path);
  std::error_code ec;
  const auto actual_size = std::filesystem::file_size(path, ec);
  if (ec)
    throw std::runtime_error(
      "FlatRuntime: immutable asset is missing or unreadable: '" + asset.path + "'");
  if (actual_size != asset.byte_count)
    throw std::runtime_error(
      "FlatRuntime: immutable asset byte_count mismatch for '" + asset.path + "'");

  std::vector<uint8_t> bytes(static_cast<std::size_t>(asset.byte_count));
  std::ifstream input(path, std::ios::binary);
  if (!input.read(
        reinterpret_cast<char *>(bytes.data()),
        static_cast<std::streamsize>(bytes.size())))
    throw std::runtime_error(
      "FlatRuntime: immutable asset read failed for '" + asset.path + "'");
  if (sha256_hex(bytes) != asset.sha256)
    throw std::runtime_error(
      "FlatRuntime: immutable asset SHA-256 mismatch for '" + asset.path + "'");
  return bytes;
}

void bind_immutable_assets(
  const tropical_plan5::ParsedPlan5 & parsed, KernelState & state)
{
  if (parsed.immutable_assets.empty()) return;

  auto storage = std::make_shared<ImmutableAssetPack>();
  std::unordered_set<uint32_t> bound_slots;
  for (const auto & asset : parsed.immutable_assets)
  {
    if (asset.sample_rate != parsed.sample_rate)
      throw std::runtime_error(
        "FlatRuntime: immutable asset sample_rate mismatch for '" + asset.path + "'");
    if (asset.arrays.empty())
      throw std::runtime_error(
        "FlatRuntime: immutable asset has no array_slots: '" + asset.path + "'");
    const auto bytes = read_asset(asset);
    if (storage->metal_bytes.size()
        > std::numeric_limits<std::size_t>::max()
            - (kImmutableAssetAlignment - 1))
      throw std::runtime_error("FlatRuntime: immutable asset alignment overflow");
    const std::size_t packed_base = align_asset(storage->metal_bytes.size());
    if (packed_base > std::numeric_limits<std::size_t>::max() - bytes.size())
      throw std::runtime_error("FlatRuntime: immutable asset pack size overflow");
    storage->metal_bytes.resize(packed_base + bytes.size(), 0);
    std::copy(bytes.begin(), bytes.end(),
              storage->metal_bytes.begin() + packed_base);

    for (const auto & binding : asset.arrays)
    {
      if (binding.slot >= state.array_ptrs.size())
        throw std::runtime_error(
          "FlatRuntime: immutable asset array slot is out of range");
      if (!bound_slots.insert(binding.slot).second)
        throw std::runtime_error(
          "FlatRuntime: immutable asset array slot is bound more than once");
      if (std::find(parsed.coeff_array_slots.begin(),
                    parsed.coeff_array_slots.end(), binding.slot)
          != parsed.coeff_array_slots.end())
        throw std::runtime_error(
          "FlatRuntime: immutable asset slot cannot be a coefficient column");
      if (binding.scalar_format != "float32")
        throw std::runtime_error(
          "FlatRuntime: immutable asset scalar_format must be 'float32'");
      if (binding.element_count == 0
          || binding.element_count != state.array_sizes[binding.slot])
        throw std::runtime_error(
          "FlatRuntime: immutable asset element_count does not match array slot size");
      if (binding.byte_offset % sizeof(float) != 0
          || binding.byte_offset > asset.byte_count
          || binding.element_count >
               (asset.byte_count - binding.byte_offset) / sizeof(float))
        throw std::runtime_error(
          "FlatRuntime: immutable asset array byte range is invalid");

      auto & converted = storage->jit_arrays.emplace_back();
      storage->array_slots.push_back(binding.slot);
      converted.resize(static_cast<std::size_t>(binding.element_count));
      for (std::size_t i = 0; i < converted.size(); ++i)
      {
        const std::size_t offset =
          static_cast<std::size_t>(binding.byte_offset) + i * sizeof(float);
        const uint32_t bits =
            static_cast<uint32_t>(bytes[offset])
          | (static_cast<uint32_t>(bytes[offset + 1]) << 8)
          | (static_cast<uint32_t>(bytes[offset + 2]) << 16)
          | (static_cast<uint32_t>(bytes[offset + 3]) << 24);
        const float value = std::bit_cast<float>(bits);
        if (!std::isfinite(value))
          throw std::runtime_error(
            "FlatRuntime: immutable asset array contains a non-finite value");
        converted[i] = std::bit_cast<int64_t>(static_cast<double>(value));
      }
      state.array_ptrs[binding.slot] = converted.data();
    }
  }
  state.immutable_assets = std::move(storage);
}

std::shared_ptr<const ScopeProgramImage> make_scope_program_image(
  const KernelState & state, uint64_t program_version)
{
  auto image = std::make_shared<ScopeProgramImage>();
  image->program_version = program_version;
  image->mode = state.mode;
  image->kernel = state.kernel;
  image->coeff_kernel = state.coeff_kernel;
  image->sample_rate = state.sample_rate;
  image->register_count = state.registers.size();
  image->temp_count = state.temps.size();
  image->coeff_register_count = state.coeff_registers.size();
  image->coeff_temp_count = state.coeff_temps.size();
  image->array_sizes = state.array_sizes;
  image->param_ptrs = state.param_ptrs;
  image->coeff_array_slots = state.coeff_array_slots;
  image->slot_names = state.slot_names;
  image->initial_slots = state.slots;
  image->immutable_assets = state.immutable_assets;
  image->slot_lookup.reserve(image->slot_names.size());
  for (uint32_t i = 0; i < image->slot_names.size(); ++i)
    image->slot_lookup.emplace(image->slot_names[i], i);
  return image;
}

std::shared_ptr<const ScopeControlImage> make_scope_control_image(
  const ScopeProgramImage & program, const KernelState & audio_state,
  uint64_t control_version,
  uint64_t effective_sample_index)
{
  auto image = std::make_shared<ScopeControlImage>();
  image->control_version = control_version;
  image->effective_sample_index = effective_sample_index;
  image->slots = program.initial_slots;
  std::unordered_map<std::string, double> audio_controls;
  audio_controls.reserve(audio_state.slot_names.size());
  for (std::size_t i = 0;
       i < audio_state.slot_names.size() && i < audio_state.slots.size(); ++i)
    audio_controls.emplace(audio_state.slot_names[i], audio_state.slots[i]);
  for (std::size_t i = 0;
       i < program.slot_names.size() && i < image->slots.size(); ++i)
    if (const auto found = audio_controls.find(program.slot_names[i]);
        found != audio_controls.end())
      image->slots[i] = found->second;
  return image;
}
uint64_t read_u64_limbs(
  const KernelState & state, const std::string & base,
  bool & complete)
{
  uint64_t value = 0;
  complete = true;
  for (uint32_t limb = 0; limb < 4; ++limb)
  {
    const std::string name =
      "param:" + base + "#u" + std::to_string(limb);
    auto it = std::find(
      state.slot_names.begin(), state.slot_names.end(), name);
    if (it == state.slot_names.end())
    {
      complete = false;
      return 0;
    }
    const std::size_t index =
      static_cast<std::size_t>(it - state.slot_names.begin());
    if (index >= state.slots.size())
    {
      complete = false;
      return 0;
    }
    const uint64_t part =
      static_cast<uint64_t>(state.slots[index]) & UINT64_C(0xffff);
    value |= part << (16 * limb);
  }
  return value;
}

void write_u64_limbs(
  KernelState & state, const std::string & base, uint64_t value)
{
  for (uint32_t limb = 0; limb < 4; ++limb)
  {
    const std::string name =
      "param:" + base + "#u" + std::to_string(limb);
    auto it = std::find(
      state.slot_names.begin(), state.slot_names.end(), name);
    if (it == state.slot_names.end()) return;
    const std::size_t index =
      static_cast<std::size_t>(it - state.slot_names.begin());
    if (index >= state.slots.size()) return;
    state.slots[index] =
      static_cast<double>((value >> (16 * limb)) & UINT64_C(0xffff));
  }
}

double signed_sample_delta(uint64_t now, uint64_t then)
{
  return now >= then
    ? static_cast<double>(now - then)
    : -static_cast<double>(then - now);
}

} // namespace

FlatRuntime::FlatRuntime(unsigned int buffer_length)
  : buffer_length_(buffer_length),
    outputBuffer(buffer_length, 0.0)
{
  if (buffer_length == 0)
    throw std::invalid_argument(
      "FlatRuntime: device buffer length must be nonzero");
}

FlatRuntime::~FlatRuntime() = default;

void FlatRuntime::process_offline()
{
#ifdef TROPICAL_METAL
  if (metal_audio_enabled_.load(std::memory_order_acquire))
  {
    wait_for_next_metal_tile("offline render");
    const uint64_t starvation_before =
      metal_tiles_->starvation_count();
    const uint64_t tags_before =
      metal_tiles_->tag_mismatch_count();
    process();
    if (metal_tiles_->starvation_count() != starvation_before
        || metal_tiles_->tag_mismatch_count() != tags_before)
      throw std::runtime_error(
        "FlatRuntime: offline Metal tile changed before consumption");
    return;
  }
#endif
  process();
}

void FlatRuntime::prepare_realtime_start(unsigned int process_cycles)
{
#ifdef TROPICAL_METAL
  if (metal_audio_enabled_.load(std::memory_order_acquire))
  {
    wait_for_next_metal_tile("real-time start");
    return;
  }
#endif
  for (unsigned int i = 0; i < process_cycles; ++i)
    process();
}

#ifdef TROPICAL_METAL
void FlatRuntime::wait_for_next_metal_tile(const char * purpose)
{
  const auto deadline =
    std::chrono::steady_clock::now() + std::chrono::seconds(30);
  while (!metal_tiles_->next_callback_ready_for_offline())
  {
    if (metal_tiles_->active_faulted())
      throw std::runtime_error(
        std::string("FlatRuntime: Metal epoch is faulted before ")
        + purpose);
    if (std::chrono::steady_clock::now() >= deadline)
      throw std::runtime_error(
        std::string("FlatRuntime: timed out waiting for Metal tile before ")
        + purpose);
    std::this_thread::yield();
  }
}

uint32_t FlatRuntime::configure_metal_render_tile_frames(
  uint32_t device_frames)
{
  static constexpr uint32_t kDefaultMinimumFrames = 512;
  static constexpr uint32_t kSafeAllocationCeiling = 16384;
  uint64_t frames = std::max<uint32_t>(
    kDefaultMinimumFrames, device_frames);
  frames =
    ((frames + device_frames - 1) / device_frames) * device_frames;

  if (const char * raw = std::getenv(
        "TROPICAL_METAL_RENDER_TILE_FRAMES");
      raw && *raw)
  {
    errno = 0;
    char * end = nullptr;
    const unsigned long parsed = std::strtoul(raw, &end, 10);
    if (errno != 0 || end == raw || *end != '\0'
        || parsed > std::numeric_limits<uint32_t>::max())
      throw std::invalid_argument(
        "FlatRuntime: TROPICAL_METAL_RENDER_TILE_FRAMES must be an integer");
    frames = parsed;
  }

  if (frames == 0 || frames < device_frames
      || frames % device_frames != 0
      || frames > kSafeAllocationCeiling)
    throw std::invalid_argument(
      "FlatRuntime: TROPICAL_METAL_RENDER_TILE_FRAMES must be a nonzero "
      "multiple of the device buffer length in [Bdev,16384]");
  return static_cast<uint32_t>(frames);
}

tropical_metal::RenderEpochRequest
FlatRuntime::make_metal_epoch_request(
  KernelState & state, uint64_t epoch_id,
  tropical_metal::EpochTransitionKind transition,
  uint64_t source_origin, uint32_t generation) const
{
  tropical_metal::RenderEpochRequest request;
  request.epoch_id = epoch_id;
  request.kernel = state.metal;
  request.sample_rate = state.sample_rate;
  request.source_origin = source_origin;
  request.transition = transition;
  if (generation == UINT32_MAX)
    generation = std::atomic_ref(state.control_published_gen)
                   .load(std::memory_order_acquire);
  const auto & slots = state.slot_generations[generation];
  request.slots.reserve(slots.size());
  for (double value : slots)
    request.slots.push_back(static_cast<float>(value));
  std::size_t column_count = 0;
  for (const auto & column : state.coeff_generations[generation])
    column_count += column.size();
  request.columns.reserve(column_count);
  for (const auto & column : state.coeff_generations[generation])
    for (int64_t value : column)
      request.columns.push_back(
        static_cast<float>(std::bit_cast<double>(value)));
  return request;
}

tropical_metal::EpochScheduleResult
FlatRuntime::schedule_metal_epoch_locked(
  KernelState & state,
  tropical_metal::EpochTransitionKind transition,
  uint64_t requested_source, uint32_t generation)
{
  const uint64_t epoch_id = next_metal_epoch_id_++;
  for (;;)
  {
    if (!metal_worker_->wait_for_admission())
      return {
        false, "MetalRenderWorker: shut down before admission",
        epoch_id, 0, 0
      };
    const auto reservation =
      metal_worker_->reserve(transition, requested_source);
    auto request = make_metal_epoch_request(
      state, epoch_id, transition,
      reservation.effective_sample_index, generation);
    request.fixed_activation = true;
    request.activation_frame = reservation.activation_frame;
    auto result = metal_worker_->schedule(std::move(request));
    if (result.retargeted) continue;
    return result;
  }
}

tropical_metal::EpochScheduleResult
FlatRuntime::publish_metal_control_snapshot_locked(
  KernelState & state, uint32_t state_index)
{
  auto generation = reserve_control_generation(state, state_index);
  materialize_control_snapshot(state, generation.target);
  auto scheduled = schedule_metal_epoch_locked(
    state, tropical_metal::EpochTransitionKind::Continuous, 0,
    generation.target);
  if (scheduled.ok)
    commit_control_snapshot(state, state_index, generation);
  else
    release_control_snapshot_reservation(state_index, generation);
  return scheduled;
}
#endif

void FlatRuntime::set_sample_index(uint64_t idx)
{
  ControlWaiterGuard priority(control_waiter_count_);
  std::lock_guard<std::mutex> lock(build_mutex_);
#ifdef TROPICAL_METAL
  const uint32_t state_index =
    active_state_.load(std::memory_order_acquire);
  KernelState & state = states_[state_index];
  if (state.metal)
  {
    (void)schedule_metal_epoch_locked(
      state, tropical_metal::EpochTransitionKind::ClockJump, idx);
    return;
  }
#endif
  sample_index_request_seq_.fetch_add(1, std::memory_order_acq_rel);
  requested_sample_index_.store(idx, std::memory_order_release);
  if (RuntimeOwnershipTestSeam * seam =
        ownership_test_seam_.load(std::memory_order_acquire);
      seam && seam->pause_after_clock_payload.load(std::memory_order_acquire))
  {
    seam->clock_payload_stored.store(true, std::memory_order_release);
    while (!seam->release_clock.load(std::memory_order_acquire))
      std::this_thread::yield();
  }
  sample_index_request_seq_.fetch_add(1, std::memory_order_release);
}

// Map a parsed plan's *metadata* (the manifest) into a fresh KernelState —
// everything except the kernel handle, which load_ir fills (via
// compile_ir_text) between this and publish_state. The plan's instruction
// graph is metadata-only now; codegen is Lean's (EmitLlvm).
KernelState FlatRuntime::build_kernel_state(const tropical_plan5::ParsedPlan5 & parsed)
{
  KernelState new_state;
  new_state.mode           = parsed.compilation_mode;
  new_state.sample_rate    = parsed.sample_rate;
  new_state.array_names    = parsed.array_slot_names;

  // Slot array
  new_state.slots.assign(parsed.slot_count, 0.0);
  new_state.slot_names = parsed.slot_names;
  for (std::size_t i = 0;
       i < parsed.slot_defaults.size() && i < new_state.slots.size();
       ++i)
  {
    new_state.slots[i] = parsed.slot_defaults[i];
  }
  for (auto & generation : new_state.slot_generations)
    generation = new_state.slots;
  new_state.audio_slots = new_state.slots;

  // Host-contract dispatch table — swaps with the plan it describes, like
  // slot_names. Field-copied because the runtime keeps its own struct (the
  // header stays free of the parser header).
  new_state.param_disciplines.reserve(parsed.param_disciplines.size());
  for (const auto & d : parsed.param_disciplines)
    new_state.param_disciplines.push_back(
      { d.name, d.discipline, d.glide_dur_sec, d.companions });

  // CF-only: there are no per-sample state registers. The kernel's
  // `%registers` argument survives in the calling convention but is never
  // read or written, so the backing buffer is empty. The temp pool
  // (`register_count`) is the SSA scratch the kernel actually uses.
  new_state.registers.clear();
  new_state.temps.assign(parsed.program.register_count, 0);
  new_state.coeff_registers = new_state.registers;
  new_state.coeff_temps.assign(parsed.program.register_count, 0);

  const auto & sizes = parsed.program.array_slot_sizes;
  new_state.array_storage.resize(sizes.size());
  for (std::size_t i = 0; i < sizes.size(); ++i)
    new_state.array_storage[i].assign(static_cast<std::size_t>(sizes[i]), 0);
  new_state.array_ptrs.resize(new_state.array_storage.size());
  new_state.array_sizes.resize(new_state.array_storage.size());
  for (std::size_t i = 0; i < new_state.array_storage.size(); ++i)
  {
    new_state.array_ptrs[i]  = new_state.array_storage[i].data();
    new_state.array_sizes[i] = new_state.array_storage[i].size();
  }

  bind_immutable_assets(parsed, new_state);

  // BANKS-AS-DATA: generation-buffer the coefficient columns the plan
  // advertises (see the KernelState comment for the protocol). Three
  // generations per column, sized like the slot's base storage; the
  // coefficient kernel's pointer view starts as a copy of the audio view
  // (publish_control_snapshot repoints column entries to its target generation).
  for (uint32_t s : parsed.coeff_array_slots)
    if (s < new_state.array_storage.size())
      new_state.coeff_array_slots.push_back(s);
  if (!new_state.coeff_array_slots.empty())
  {
    for (auto & gen : new_state.coeff_generations)
    {
      gen.resize(new_state.coeff_array_slots.size());
      for (std::size_t j = 0; j < new_state.coeff_array_slots.size(); ++j)
        gen[j].assign(
          new_state.array_storage[new_state.coeff_array_slots[j]].size(), 0);
    }
    new_state.coeff_array_ptrs = new_state.array_ptrs;
  }

  return new_state;
}

// Atomic double-buffer flip (CF-only: no by-name state transfer — see below).
// Assumes new_state already carries a populated kernel handle.
bool FlatRuntime::publish_state(
  KernelState && new_state, std::optional<KernelState> scope_state)
{
  ControlWaiterGuard priority(control_waiter_count_);
  std::lock_guard<std::mutex> lock(build_mutex_);
#ifdef TROPICAL_METAL
  if (new_state.metal)
  {
    if (!metal_worker_)
    {
      metal_worker_ =
        std::make_unique<tropical_metal::MetalRenderWorker>(
          *metal_tiles_);
      // The worker is created at most once and lives until FlatRuntime
      // destruction. Publish a stable observation pointer so telemetry never
      // waits behind a control transaction holding build_mutex_.
      metal_worker_view_.store(
        metal_worker_.get(), std::memory_order_release);
    }
    const uint64_t epoch_id = next_metal_epoch_id_++;
    const auto transition =
      metal_runtime_loaded_.load(std::memory_order_acquire)
        ? tropical_metal::EpochTransitionKind::HotSwap
        : tropical_metal::EpochTransitionKind::Fresh;
    auto request = make_metal_epoch_request(
      new_state, epoch_id, transition,
      published_sample_index_.load(std::memory_order_acquire));
    const auto scheduled = metal_worker_->schedule(std::move(request));
    if (!scheduled.ok)
      throw std::runtime_error(
        "FlatRuntime: initial Metal epoch failed: " + scheduled.error);
  }
#endif
  const uint32_t active   = active_state_.load(std::memory_order_acquire);
  const uint32_t inactive = 1U - active;
  for (;;)
  {
    StorageOwner expected = StorageOwner::Free;
    if (state_owners_[inactive].compare_exchange_strong(
          expected, StorageOwner::Writer,
          std::memory_order_acquire, std::memory_order_relaxed))
      break;
    if (RuntimeOwnershipTestSeam * seam =
          ownership_test_seam_.load(std::memory_order_acquire))
      seam->writer_waiting_for_state.store(true, std::memory_order_release);
    std::this_thread::yield();
  }

  // Build the read-side image before moving the state. Its program storage is
  // immutable and its Plan-6 pack is shared, so publication below never leaves
  // the scope pointing into either movable KernelState slot.
  const uint64_t program_version =
    recompile_version_.load(std::memory_order_relaxed) + 1;
  const KernelState & scope_source = scope_state
    ? *scope_state : new_state;
  auto scope_program =
    make_scope_program_image(scope_source, program_version);
  auto scope_controls = make_scope_control_image(
    *scope_program, new_state, next_scope_control_version_++,
    published_sample_index_.load(std::memory_order_acquire));
  auto scope_snapshot = std::make_shared<ScopeSnapshot>();
  scope_snapshot->program = std::move(scope_program);
  scope_snapshot->controls = std::move(scope_controls);

  // CF-only: no by-name state transfer on hot-swap. Registers/arrays/slots
  // zero-init from the fresh kernel. The audio-owned clock is runtime-global,
  // so a state flip cannot repeat, skip, or race its sample position.

  states_[inactive] = std::move(new_state);
  state_owners_[inactive].store(
    StorageOwner::Free, std::memory_order_release);

  // A new Metal kernel is unprimed (and JIT/sync Metal has the same E=C
  // mapping). Publish the state and its initial generation through the same
  // word audio captures. Audio never waits: if it is between its two phase
  // increments this control thread simply retries.
  state_requires_reprime_[inactive].store(true, std::memory_order_relaxed);
  const uint64_t state_bits = static_cast<uint64_t>(inactive) << 2;
  const uint64_t gen_bits =
    std::atomic_ref(states_[inactive].control_published_gen)
      .load(std::memory_order_acquire);
  for (;;)
  {
    uint64_t expected =
      capture_boundary_word_.load(std::memory_order_acquire);
    if ((expected & kCapturePhaseBit_) != 0)
    {
      std::this_thread::yield();
      continue;
    }
    const uint64_t desired =
      (expected & ~(kCaptureStateMask_ | kCaptureGenerationMask_))
      | state_bits | gen_bits;
    if (capture_boundary_word_.compare_exchange_weak(
          expected, desired,
          std::memory_order_release, std::memory_order_acquire))
      break;
  }
  active_state_.store(inactive, std::memory_order_release);
  published_slot_count_.store(
    static_cast<uint32_t>(states_[inactive].slots.size()),
    std::memory_order_release);
  recompile_version_.store(program_version, std::memory_order_release);
  std::atomic_store_explicit(
    &scope_snapshot_,
    std::shared_ptr<const ScopeSnapshot>(std::move(scope_snapshot)),
    std::memory_order_release);
#ifdef TROPICAL_METAL
  const bool metal = static_cast<bool>(states_[inactive].metal);
  metal_runtime_loaded_.store(metal, std::memory_order_release);
  metal_audio_enabled_.store(metal, std::memory_order_release);
#endif
  return true;
}

bool FlatRuntime::load_ir(const std::string & ir_text, const std::string & manifest_json)
{
  return load_ir_staged(ir_text, {}, {}, manifest_json);
}

bool FlatRuntime::load_ir_msl(const std::string & ir_text,
                              const std::string & msl_source,
                              const std::string & manifest_json)
{
  return load_ir_staged(ir_text, msl_source, {}, manifest_json);
}

bool FlatRuntime::load_ir_staged(const std::string & ir_text,
                                 const std::string & msl_source,
                                 const std::string & coeff_ir,
                                 const std::string & manifest_json)
{
  return load_ir_staged_with_scope(
    ir_text, msl_source, coeff_ir, manifest_json, {}, {}, {});
}

bool FlatRuntime::load_ir_staged_with_scope(
  const std::string & ir_text, const std::string & msl_source,
  const std::string & coeff_ir, const std::string & manifest_json,
  const std::string & scope_ir, const std::string & scope_coeff_ir,
  const std::string & scope_manifest_json)
{
  using json = nlohmann::json;
  auto compile_jit_state = [this](
      const std::string & kernel_ir, const std::string & coefficient_ir,
      const std::string & manifest_text) {
    const json manifest = json::parse(manifest_text);
    const std::string schema = manifest.value("schema", std::string{});
    if (schema != "tropical_plan_5" && schema != "tropical_plan_6")
      throw std::runtime_error(
        "FlatRuntime: load_ir unsupported manifest schema '" + schema +
        "'; expected 'tropical_plan_5' or 'tropical_plan_6'");
    auto parsed = tropical_plan5::parse_plan5(manifest);
    KernelState state = build_kernel_state(parsed);
    state.mode = tropical_jit::CompilationMode::Fused;

    auto kernel_result =
      tropical_jit::OrcJitEngine::instance().compile_ir_text(kernel_ir);
    if (!kernel_result)
    {
      std::string err;
      llvm::handleAllErrors(kernel_result.takeError(),
        [&err](const llvm::ErrorInfoBase & e) { err = e.message(); });
      throw std::runtime_error(
        "FlatRuntime: IR JIT compilation failed: " + err);
    }
    state.kernel = *kernel_result;

    if (!coefficient_ir.empty())
    {
      // O0: the coefficient kernel runs once per control write — codegen
      // quality is irrelevant, and O2 on its thousands-of-flops block is
      // exactly the compile wall the stage-0 split removes.
      auto coeff_result =
        tropical_jit::OrcJitEngine::instance().compile_ir_text(
          coefficient_ir, true);
      if (!coeff_result)
      {
        std::string err;
        llvm::handleAllErrors(coeff_result.takeError(),
          [&err](const llvm::ErrorInfoBase & e) { err = e.message(); });
        throw std::runtime_error(
          "FlatRuntime: coefficient IR JIT compilation failed: " + err);
      }
      state.coeff_kernel = *coeff_result;
    }
    // Initial coefficient slots/columns must exist before either artifact can
    // be published. This operates on an unpublished local state.
    publish_control_snapshot(state);
    return state;
  };

  KernelState new_state =
    compile_jit_state(ir_text, coeff_ir, manifest_json);

  if (!msl_source.empty())
  {
#ifdef TROPICAL_METAL
    if (!metal_queue_ready_.load(std::memory_order_acquire))
    {
      metal_render_tile_frames_ =
        configure_metal_render_tile_frames(buffer_length_);
      metal_tiles_ = std::make_unique<EpochTileQueue>(
        buffer_length_, metal_render_tile_frames_);
      metal_tiles_->synchronize_audio_coordinates(
        published_device_frame_.load(std::memory_order_acquire),
        published_sample_index_.load(std::memory_order_acquire));
      metal_queue_ready_.store(true, std::memory_order_release);
    }
    std::string err;
    std::size_t column_floats = 0;
    for (const auto & column :
         new_state.coeff_generations[
           std::atomic_ref(new_state.control_published_gen)
             .load(std::memory_order_relaxed)])
      column_floats += column.size();
    new_state.metal = tropical_metal::create(
      msl_source, metal_render_tile_frames_,
      static_cast<uint32_t>(new_state.slots.size()),
      static_cast<uint32_t>(column_floats),
      !new_state.immutable_assets
          || new_state.immutable_assets->metal_bytes.empty()
        ? nullptr : new_state.immutable_assets->metal_bytes.data(),
      new_state.immutable_assets
        ? new_state.immutable_assets->metal_bytes.size() : 0,
      err);
    if (!new_state.metal)
      throw std::runtime_error("FlatRuntime: " + err);
#else
    throw std::runtime_error(
      "FlatRuntime: MSL source supplied but the engine was built without TROPICAL_METAL");
#endif
  }

  std::optional<KernelState> scope_state;
  if (!scope_ir.empty())
  {
    if (scope_manifest_json.empty())
      throw std::invalid_argument(
        "FlatRuntime: scope IR requires a scope manifest");
    scope_state.emplace(compile_jit_state(
      scope_ir, scope_coeff_ir, scope_manifest_json));
  }

  return publish_state(std::move(new_state), std::move(scope_state));
}

FlatRuntime::ControlGenerationReservation
FlatRuntime::reserve_control_generation(
  KernelState & state, uint32_t state_index)
{
  const uint32_t pub = std::atomic_ref(state.control_published_gen)
                         .load(std::memory_order_acquire);
  if (state_index == UINT32_MAX)
  {
    uint32_t target = 0;
    while (target == pub) ++target;
    return {target, false};
  }

  // Control writers are serialized by build_mutex_. They may wait/yield off
  // the real-time thread until a non-published generation is free.
  for (;;)
  {
    const uint32_t published =
      std::atomic_ref(state.control_published_gen)
        .load(std::memory_order_acquire);
    for (uint32_t target = 0;
         target < control_generation_owners_[state_index].size();
         ++target)
    {
      if (target == published) continue;
      StorageOwner expected = StorageOwner::Free;
      if (control_generation_owners_[state_index][target]
            .compare_exchange_strong(
              expected, StorageOwner::Writer,
              std::memory_order_acquire, std::memory_order_relaxed))
        return {target, true};
    }
    std::this_thread::yield();
  }
}

void FlatRuntime::materialize_control_snapshot(
  KernelState & state, uint32_t target)
{
  int64_t ** arrays = state.array_ptrs.data();
  if (!state.coeff_array_slots.empty())
  {
    for (std::size_t j = 0; j < state.coeff_array_slots.size(); ++j)
      state.coeff_array_ptrs[state.coeff_array_slots[j]] =
        state.coeff_generations[target][j].data();
    arrays = state.coeff_array_ptrs.data();
  }
  if (state.coeff_kernel)
  {
    double scratch_out = 0.0;
    state.coeff_kernel(
      nullptr,
      state.coeff_registers.data(),
      arrays,
      state.array_sizes.data(),
      state.coeff_temps.data(),
      state.sample_rate,
      0,
      state.param_ptrs.data(),
      &scratch_out,
      1,
      state.slots.data());
  }
  std::copy(state.slots.begin(), state.slots.end(),
            state.slot_generations[target].begin());
}

void FlatRuntime::release_control_snapshot_reservation(
  uint32_t state_index, ControlGenerationReservation reservation)
{
  if (reservation.owned)
    control_generation_owners_[state_index][reservation.target].store(
      StorageOwner::Free, std::memory_order_release);
}

void FlatRuntime::commit_control_snapshot(
  KernelState & state, uint32_t state_index,
  ControlGenerationReservation reservation)
{
  if (state_index == UINT32_MAX)
  {
    std::atomic_ref(state.control_published_gen)
      .store(reservation.target, std::memory_order_release);
    return;
  }

  release_control_snapshot_reservation(state_index, reservation);
  for (;;)
  {
    uint64_t expected =
      capture_boundary_word_.load(std::memory_order_acquire);
    if ((expected & kCapturePhaseBit_) != 0)
    {
      std::this_thread::yield();
      continue;
    }
    const uint32_t captured_state =
      static_cast<uint32_t>((expected & kCaptureStateMask_) >> 2);
    if (captured_state != state_index)
    {
      // build_mutex_ excludes a state flip here; this is defensive.
      std::this_thread::yield();
      continue;
    }
    const uint64_t desired =
      (expected & ~kCaptureGenerationMask_) | reservation.target;
    if (capture_boundary_word_.compare_exchange_weak(
          expected, desired,
          std::memory_order_release, std::memory_order_acquire))
      break;
  }
  std::atomic_ref(state.control_published_gen)
    .store(reservation.target, std::memory_order_release);
  publish_scope_controls_locked(state);
}

void FlatRuntime::publish_scope_controls_locked(
  const KernelState & state, uint64_t effective_sample_index)
{
  const auto current = std::atomic_load_explicit(
    &scope_snapshot_, std::memory_order_acquire);
  if (!current || !current->program) return;
  if (effective_sample_index == UINT64_MAX)
    effective_sample_index =
      published_sample_index_.load(std::memory_order_acquire);
  auto next = std::make_shared<ScopeSnapshot>();
  next->program = current->program;
  next->controls = make_scope_control_image(
    *current->program, state, next_scope_control_version_++,
    effective_sample_index);
  std::atomic_store_explicit(
    &scope_snapshot_,
    std::shared_ptr<const ScopeSnapshot>(std::move(next)),
    std::memory_order_release);
}

ParamDispatchResult FlatRuntime::dispatch_param_sync(
  const std::string & name, double value)
{
  if (!std::isfinite(value))
  {
    const uint64_t observed =
      published_sample_index_.load(std::memory_order_acquire);
    return {
      false, "set_param: value must be finite", {},
      observed, observed
    };
  }

  ControlWaiterGuard priority(control_waiter_count_);
  std::lock_guard<std::mutex> lock(build_mutex_);
  const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
  KernelState & state = states_[state_idx];
  const uint64_t observed_sample_index =
    published_sample_index_.load(std::memory_order_acquire);
  const std::vector<double> base_slots = state.slots;

#ifdef TROPICAL_METAL
  if (state.metal)
  {
    // Validate without reserving either a control generation or an activation
    // epoch. Raw parameter coefficient work is epoch-independent, so perform
    // it now in local storage while an earlier activation may still be
    // awaiting acknowledgement.
    state.slots = base_slots;
    auto preflight = dispatch_param_sync_locked(
      state, state_idx, name, value, observed_sample_index);
    preflight.observed_sample_index = observed_sample_index;
    if (!preflight.ok)
    {
      state.slots = base_slots;
      return preflight;
    }
    const bool raw = preflight.discipline == "raw";
    std::vector<double> raw_slots;
    std::vector<std::vector<int64_t>> raw_columns;
    if (raw)
    {
      raw_slots = state.slots;
      const uint32_t published_generation =
        std::atomic_ref(state.control_published_gen)
          .load(std::memory_order_acquire);
      raw_columns = state.coeff_generations[published_generation];
      std::vector<int64_t *> arrays = state.array_ptrs;
      for (std::size_t j = 0; j < state.coeff_array_slots.size(); ++j)
        arrays[state.coeff_array_slots[j]] = raw_columns[j].data();
      if (state.coeff_kernel)
      {
        double scratch_out = 0.0;
        state.coeff_kernel(
          nullptr,
          state.coeff_registers.data(),
          arrays.data(),
          state.array_sizes.data(),
          state.coeff_temps.data(),
          state.sample_rate,
          0,
          state.param_ptrs.data(),
          &scratch_out,
          1,
          raw_slots.data());
      }
    }
    state.slots = base_slots;

    if (!metal_worker_->wait_for_admission())
    {
      preflight.ok = false;
      preflight.error = "set_param: Metal worker shut down before admission";
      return preflight;
    }
    auto reservation = reserve_control_generation(state, state_idx);
    const uint64_t epoch_id = next_metal_epoch_id_++;
    for (;;)
    {
      const auto epoch = metal_worker_->reserve(
        tropical_metal::EpochTransitionKind::Continuous);
      ParamDispatchResult result = preflight;
      result.observed_sample_index = observed_sample_index;
      result.effective_sample_index = epoch.effective_sample_index;
      state.slots = base_slots;
      if (raw)
      {
        state.slots = raw_slots;
        state.slot_generations[reservation.target] = raw_slots;
        state.coeff_generations[reservation.target] = raw_columns;
      }
      else
      {
        result = dispatch_param_sync_locked(
          state, state_idx, name, value,
          epoch.effective_sample_index);
        result.observed_sample_index = observed_sample_index;
        result.effective_sample_index = epoch.effective_sample_index;
        if (!result.ok)
        {
          state.slots = base_slots;
          release_control_snapshot_reservation(
            state_idx, reservation);
          return result;
        }
        materialize_control_snapshot(state, reservation.target);
      }

      auto request = make_metal_epoch_request(
        state, epoch_id,
        tropical_metal::EpochTransitionKind::Continuous,
        epoch.effective_sample_index, reservation.target);
      request.fixed_activation = true;
      request.activation_frame = epoch.activation_frame;
      const auto scheduled =
        metal_worker_->schedule(std::move(request));
      if (scheduled.retargeted)
      {
        state.slots = base_slots;
        if (!metal_worker_->wait_for_admission())
        {
          release_control_snapshot_reservation(
            state_idx, reservation);
          result.ok = false;
          result.error =
            "set_param: Metal worker shut down during retarget";
          return result;
        }
        continue;
      }
      if (!scheduled.ok)
      {
        state.slots = base_slots;
        release_control_snapshot_reservation(
          state_idx, reservation);
        result.ok = false;
        result.error =
          "set_param: Metal epoch preparation failed: "
          + scheduled.error;
        return result;
      }
      commit_control_snapshot(state, state_idx, reservation);
      result.epoch_id = scheduled.epoch_id;
      result.device_activation_frame = scheduled.activation_frame;
      result.activation_published = true;
      result.activation_audible =
        metal_tiles_->activation_acknowledged() == scheduled.epoch_id;
      return result;
    }
  }
#endif

  auto reservation = reserve_control_generation(state, state_idx);

  for (;;)
  {
    // Read C/E and the reset/fresh qualifiers under the audio boundary's
    // seqlock. Any capture crossing this snapshot changes the word and makes
    // the eventual publication CAS lose.
    uint64_t boundary_word = 0;
    uint64_t effective_sample_index = 0;
    for (;;)
    {
      boundary_word =
        capture_boundary_word_.load(std::memory_order_acquire);
      if ((boundary_word & kCapturePhaseBit_) != 0)
      {
        std::this_thread::yield();
        continue;
      }
      const uint64_t next_capture =
        next_capture_sample_index_.load(std::memory_order_relaxed);
      const uint64_t steady_effective =
        next_capture_effective_sample_index_.load(
          std::memory_order_relaxed);
      const uint64_t request_seq =
        sample_index_request_seq_.load(std::memory_order_acquire);
      const uint64_t applied_request_seq =
        audio_applied_sample_index_seq_.load(std::memory_order_relaxed);
      if ((request_seq & 1U) == 0 && request_seq != applied_request_seq)
        effective_sample_index =
          requested_sample_index_.load(std::memory_order_acquire);
      else if (state_requires_reprime_[state_idx].load(
                 std::memory_order_relaxed))
        effective_sample_index = next_capture;
      else
        effective_sample_index = steady_effective;
      if (capture_boundary_word_.load(std::memory_order_acquire)
          == boundary_word)
        break;
    }

    state.slots = base_slots;
    auto result = dispatch_param_sync_locked(
      state, state_idx, name, value, effective_sample_index);
    result.observed_sample_index = observed_sample_index;
    result.effective_sample_index = effective_sample_index;
    if (!result.ok)
    {
      state.slots = base_slots;
      release_control_snapshot_reservation(state_idx, reservation);
      return result;
    }
    materialize_control_snapshot(state, reservation.target);
    if (RuntimeOwnershipTestSeam * seam =
          ownership_test_seam_.load(std::memory_order_acquire);
        seam
        && seam->pause_after_dispatch_materialization.load(
          std::memory_order_acquire))
    {
      seam->dispatch_materialized.store(true, std::memory_order_release);
      while (!seam->release_dispatch.load(std::memory_order_acquire))
        std::this_thread::yield();
    }

    // Make the completed target capturable, then atomically replace only the
    // generation bits in the exact stable boundary word sampled above.
    release_control_snapshot_reservation(state_idx, reservation);
    const uint64_t desired =
      (boundary_word & ~kCaptureGenerationMask_) | reservation.target;
    if (capture_boundary_word_.compare_exchange_strong(
          boundary_word, desired,
          std::memory_order_release, std::memory_order_acquire))
    {
      std::atomic_ref(state.control_published_gen)
        .store(reservation.target, std::memory_order_release);
      publish_scope_controls_locked(state, result.effective_sample_index);
      return result;
    }

    // Audio crossed the candidate boundary (or another boundary-qualified
    // state transition won). The unpublished target cannot be captured;
    // reacquire it, recompute temporal companions at the new E, and retry.
    for (;;)
    {
      StorageOwner expected = StorageOwner::Free;
      if (control_generation_owners_[state_idx][reservation.target]
            .compare_exchange_strong(
              expected, StorageOwner::Writer,
              std::memory_order_acquire, std::memory_order_relaxed))
        break;
      std::this_thread::yield();
    }
  }
}

ParamDispatchResult FlatRuntime::dispatch_param_sync_at_sample_index(
  const std::string & name, double value, uint64_t sample_index)
{
  if (!std::isfinite(value))
    return {
      false, "set_param: value must be finite", {},
      sample_index, sample_index
    };

  ControlWaiterGuard priority(control_waiter_count_);
  std::lock_guard<std::mutex> lock(build_mutex_);
  const uint32_t state_idx = active_state_.load(std::memory_order_acquire);
  KernelState & state = states_[state_idx];
  auto result = dispatch_param_sync_locked(
    state, state_idx, name, value, sample_index);
  result.observed_sample_index = sample_index;
  result.effective_sample_index = sample_index;
  if (result.ok)
    publish_control_snapshot(state, state_idx);
  return result;
}

ParamDispatchResult FlatRuntime::dispatch_param_sync_locked(
  KernelState & state, uint32_t state_idx, const std::string & name,
  double value, uint64_t sample_index)
{
  auto slot_of = [&state](const std::string & slot_name) -> uint32_t {
    for (uint32_t i = 0; i < state.slot_names.size(); ++i)
      if (state.slot_names[i] == slot_name) return i;
    return UINT32_MAX;
  };
  auto read = [&state](uint32_t i) {
    return i < state.slots.size() ? state.slots[i] : 0.0;
  };
  auto write = [&state](uint32_t i, double v) {
    if (i < state.slots.size()) state.slots[i] = v;
  };
  const ParamDiscipline * pd = state.find_discipline(name);
  const std::string discipline = pd ? pd->discipline : std::string{"raw"};
  auto fail = [&](std::string error) {
    return ParamDispatchResult{
      false, std::move(error), discipline, sample_index, sample_index
    };
  };
  auto commit = [&]() {
    return ParamDispatchResult{
      true, {}, discipline, sample_index, sample_index
    };
  };
  const double now = static_cast<double>(sample_index);

  if (discipline == "glide")
  {
    const uint32_t v0i = slot_of("param:" + name + "#v0");
    const uint32_t v1i = slot_of("param:" + name + "#v1");
    const uint32_t t0i = slot_of("param:" + name + "#t0");
    if (v0i == UINT32_MAX)
      return fail("set_param: no glide slots for '" + name + "'");
    const double dur_sec = pd->glide_dur_sec > 0.0
      ? pd->glide_dur_sec : 0.02;
    const double dur = state.sample_rate * dur_sec;
    const double v0 = read(v0i);
    const double v1 = read(v1i);
    const double t0 = read(t0i);
    bool exact_t0_available = false;
    const uint64_t exact_t0 =
      read_u64_limbs(state, name + "#t0", exact_t0_available);
    const double elapsed = exact_t0_available
      ? signed_sample_delta(sample_index, exact_t0)
      : now - t0;
    const double r = elapsed / dur;
    const double s = r < 0.0 ? 0.0 : (r > 1.0 ? 1.0 : r);
    write(v0i, v0 + (v1 - v0) * (s * s * (3.0 - 2.0 * s)));
    write(v1i, value);
    write(t0i, now);
    write_u64_limbs(state, name + "#t0", sample_index);
    return commit();
  }

  if (discipline == "anchor")
  {
    const uint32_t fi = slot_of("param:" + name);
    if (fi == UINT32_MAX)
      return fail("set_param: unknown param '" + name + "'");
    const uint32_t pi = slot_of("param:" + name + "#phase");
    if (pi == UINT32_MAX)
    {
      write(fi, value);
      return commit();
    }
    const double inc0 =
      std::floor(read(fi) * 4294967296.0 / state.sample_rate);
    const double inc1 =
      std::floor(value * 4294967296.0 / state.sample_rate);
    const double dcyc = ((inc0 - inc1) * now) / 4294967296.0;
    const double phase = read(pi) + dcyc;
    write(pi, phase - std::floor(phase));
    write(fi, value);
    return commit();
  }

  if (discipline == "velocity")
  {
    const uint32_t vi = slot_of("param:" + name);
    if (vi == UINT32_MAX)
      return fail("set_param: unknown param '" + name + "'");
    std::string tau;
    if (!pd->companions.empty()) tau = pd->companions.front();
    else
    {
      tau = name;
      const std::size_t pos = tau.find("velocity");
      if (pos != std::string::npos) tau.replace(pos, 8, "tau_base");
    }
    const uint32_t ti = slot_of("param:" + tau);
    if (ti == UINT32_MAX)
      return fail("set_param: no origin slot '" + tau + "'");
    write(ti, read(ti) + (read(vi) - value) * now / state.sample_rate);
    write(vi, value);
    return commit();
  }

  const uint32_t bi = slot_of("param:" + name);
  if (bi == UINT32_MAX)
    return fail("set_param: unknown param '" + name + "'");
  write(bi, value);
  return commit();
}

} // namespace tropical_runtime
