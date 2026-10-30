/**
 * @tropical/runtime-wasm (in-repo for now) — the browser-side consumer of
 * tropical's wasm codegen. Depends only on the `tropical_kernel_1` contract
 * (design/typed-kernel.md); knows nothing of the compiler. Extract to its own package when
 * a second consumer appears (see design notes / CLAUDE.md).
 */
export type { KernelManifest, KernelPort, KernelElem, KernelRole } from './manifest.js'
export {
  kernelFrames, kernelParams, kernelOutputChannelCount,
  MAX_KERNEL_OUTPUT_CHANNELS,
} from './manifest.js'
export type { KernelLayout } from './layout.js'
export { computeLayout } from './layout.js'
export { WasmKernel, importedMemoryMinPages } from './kernel.js'
