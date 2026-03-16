#include "graph/Module.hpp"
#include "graph/ModuleMethods.hpp"
#include "graph/ModuleNumericJitMethods.hpp"

#ifndef EGRESS_LLVM_ORC_JIT
// Non-JIT fallback: outputs are already evaluated; just return them directly.
const Value & Module::materialize_output_value(unsigned int output_id, bool previous)
{
  auto & destinations = previous ? prev_outputs : outputs;
  if (output_id >= destinations.size())
  {
    static Value zero = egress_expr::float_value(0.0);
    return zero;
  }
  return destinations[output_id];
}
#endif
