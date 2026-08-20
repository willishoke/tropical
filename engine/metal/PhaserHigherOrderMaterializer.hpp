#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tropical_metal
{

// Transform the self-describing primitive image emitted by the tile JIT into
// the immutable float-crossing image consumed by Metal.  The vector stores
// f64 values as raw i64 bits because it is also an ordinary Tropical array
// slot.  Failure leaves admission to the worker's exact fallback.
bool materialize_higher_order_phaser_image(
  std::vector<int64_t> & image,
  uint32_t interval_frames,
  std::string * error = nullptr);

} // namespace tropical_metal
