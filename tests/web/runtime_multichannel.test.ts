import { describe, expect, test } from 'bun:test'
import {
  WasmKernel, computeLayout, kernelFrames, kernelOutputChannelCount,
  type KernelLayout, type KernelManifest,
} from '../../web/runtime/index'

const FRAMES = 8

/** A `tropical_kernel_1` manifest: tau[FRAMES], no params, one output. */
function manifest(outputChannelCount: number, frames: number = FRAMES): KernelManifest {
  return {
    schema: 'tropical_kernel_1',
    rate: 44100,
    inputs: [{ name: 'tau', elem: 'i64q32', shape: [frames], role: 'signal' }],
    outputs: [{ name: 'out', elem: 'f64', shape: [frames, outputChannelCount] }],
  }
}

function fakeKernel(
  outputChannelCount: number,
  sample: (frame: number, channel: number, absoluteFrame: number) => number,
): WasmKernel {
  const memory = new WebAssembly.Memory({ initial: 1 })
  const kernelManifest = manifest(outputChannelCount)
  const layout: KernelLayout = computeLayout(kernelManifest, 0)
  // The port: argv = [tau, ...params, out]; the frame count is the kernel's.
  const kernel = (...pointers: number[]) => {
    const tau = new BigInt64Array(memory.buffer, pointers[0]!, FRAMES)
    const view = new Float64Array(
      memory.buffer, pointers[pointers.length - 1]!, FRAMES * outputChannelCount)
    for (let frame = 0; frame < FRAMES; frame++)
      for (let channel = 0; channel < outputChannelCount; channel++)
        view[frame * outputChannelCount + channel] =
          sample(frame, channel, Number(tau[frame]! >> 32n))
  }

  const value = Object.create(WasmKernel.prototype) as WasmKernel & {
    memory: WebAssembly.Memory
    kernel: typeof kernel
    layout: KernelLayout
    argv: number[]
    frameIdx: number
    fadeInRem: number
    fadeOutRem: number
  }
  Object.assign(value, {
    memory,
    kernel,
    layout,
    argv: [layout.tau, ...layout.params, layout.output],
    outputChannelCount,
    frameIdx: 0,
    fadeInRem: 0,
    fadeOutRem: 0,
  })
  return value
}

describe('multichannel wasm runtime contract', () => {
  test('the manifest is the type: tau first, one output, bounded width', () => {
    expect(kernelFrames(manifest(1))).toBe(FRAMES)
    expect(kernelOutputChannelCount(manifest(1))).toBe(1)
    expect(() => kernelOutputChannelCount(manifest(0))).toThrow()
    expect(() => kernelOutputChannelCount(manifest(1.5))).toThrow()
    expect(() => kernelOutputChannelCount(manifest(65))).toThrow(/\[1, 64\]/)
    const noTau = { ...manifest(1), inputs: [] }
    expect(() => kernelFrames(noTau)).toThrow(/tau/)
    const mismatched = { ...manifest(1), outputs: [{ name: 'out', elem: 'f64', shape: [4, 1] }] }
    expect(() => kernelOutputChannelCount(mismatched as KernelManifest)).toThrow(/frame count/)
  })

  test('linear memory reserves frames times compact channel width', () => {
    const layout = computeLayout(manifest(3), 0)
    expect(layout.frames).toBe(FRAMES)
    expect(layout.endByte - layout.output).toBe(FRAMES * 3 * 8)
    expect(layout.params).toEqual([])
  })

  test('render exposes frame-major interleaved samples', () => {
    const kernel = fakeKernel(2, (frame, channel) => frame * 10 + channel)
    expect(Array.from(kernel.render(3))).toEqual([0, 1, 10, 11, 20, 21])
  })

  test('the absolute sample clock advances in frames, not scalar samples', () => {
    const kernel = fakeKernel(2, (_frame, channel, absoluteFrame) =>
      absoluteFrame * 10 + channel)
    kernel.render(2)
    expect(Array.from(kernel.render(1))).toEqual([20, 21])
  })

  test('a window wider than the block is chunked over the same clock', () => {
    const kernel = fakeKernel(1, (_frame, _channel, absoluteFrame) => absoluteFrame)
    expect(Array.from(kernel.render(FRAMES + 3)))
      .toEqual(Array.from({ length: FRAMES + 3 }, (_, i) => i))
    expect(Array.from(kernel.render(2))).toEqual([FRAMES + 3, FRAMES + 4])
  })

  test('tau is the BigInt64 image of idx << 32 across the 2³² frame wrap', () => {
    const kernel = fakeKernel(1, (_frame, _channel, absoluteFrame) => absoluteFrame)
    const start = 2 ** 32 - 3
    ;(kernel as unknown as { frameIdx: number }).frameIdx = start
    expect(Array.from(kernel.render(FRAMES)))
      .toEqual(Array.from({ length: FRAMES }, (_, s) =>
        Number(BigInt.asIntN(64, BigInt(start + s) << 32n) >> 32n)))
    const { memory, layout } = kernel as unknown as
      { memory: WebAssembly.Memory, layout: KernelLayout }
    const tau = new BigInt64Array(memory.buffer, layout.tau, FRAMES)
    for (let s = 0; s < FRAMES; s++)
      expect(tau[s]).toBe(BigInt.asIntN(64, BigInt(start + s) << 32n))
  })

  test('process spans blocks on the same clock as render', () => {
    const kernel = fakeKernel(1, (_frame, _channel, absoluteFrame) => absoluteFrame)
    const out = new Float32Array(FRAMES + 3)
    kernel.process([out], FRAMES + 3)
    expect(Array.from(out)).toEqual(Array.from({ length: FRAMES + 3 }, (_, i) => i))
    kernel.process([out], 2)
    expect(Array.from(out.subarray(0, 2))).toEqual([FRAMES + 3, FRAMES + 4])
    expect(Array.from(kernel.render(1))).toEqual([FRAMES + 5])
  })

  test('stereo maps independently, clamps, and clears surplus host channels', () => {
    const kernel = fakeKernel(2, (frame, channel) =>
      channel === 0 ? frame + 1 : -300 - frame)
    const left = new Float32Array(2)
    const right = new Float32Array(2)
    const surplus = new Float32Array([9, 9])
    kernel.process([left, right, surplus], 2)
    expect(Array.from(left)).toEqual([1, 2])
    expect(Array.from(right)).toEqual([-256, -256])
    expect(Array.from(surplus)).toEqual([0, 0])
  })

  test('mono remains upmixed and a narrow stereo host is refused', () => {
    const mono = fakeKernel(1, (frame) => frame + 0.25)
    const left = new Float32Array(2)
    const right = new Float32Array(2)
    mono.process([left, right], 2)
    expect(Array.from(left)).toEqual([0.25, 1.25])
    expect(Array.from(right)).toEqual([0.25, 1.25])

    const stereo = fakeKernel(2, () => 0)
    expect(() => stereo.process([left], 2)).toThrow(/requires 2/)
  })

  test('one fade gain is shared by every channel in a frame', () => {
    const kernel = fakeKernel(2, (_frame, channel) => channel + 1)
    const left = new Float32Array(2)
    const right = new Float32Array(2)
    kernel.beginFadeIn()
    kernel.process([left, right], 2)
    expect(left[0]).toBe(0)
    expect(right[0]).toBe(0)
    expect(right[1]! / left[1]!).toBeCloseTo(2, 6)
  })
})
