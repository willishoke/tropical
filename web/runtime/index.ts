/**
 * @tropical/runtime-wasm (in-repo for now) — the browser-side consumer of
 * tropical's wasm codegen. Depends only on the KernelManifest contract; knows
 * nothing of tropical_plan_6 or the compiler. Extract to its own package when
 * a second consumer appears (see design notes / CLAUDE.md).
 */
export type { KernelManifest } from './manifest.js'
export { kernelOutputChannelCount } from './manifest.js'
export type { KernelLayout } from './layout.js'
export { computeLayout } from './layout.js'
export { WasmKernel } from './kernel.js'
