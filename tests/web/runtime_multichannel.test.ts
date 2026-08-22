import { describe, expect, test } from 'bun:test'
import {
  WasmKernel, computeLayout, kernelOutputChannelCount,
  type KernelLayout, type KernelManifest,
} from '../../web/runtime/index'

function manifest(outputChannelCount?: number): KernelManifest {
  return {
    sampleRate: 44100,
    registerCount: 0,
    arraySlotSizes: [],
    slotCount: 0,
    slotDefaults: [],
    ...(outputChannelCount === undefined ? {} : { outputChannelCount }),
  }
}

function fakeKernel(
  outputChannelCount: number,
  sample: (frame: number, channel: number, absoluteFrame: number) => number,
): WasmKernel {
  const memory = new WebAssembly.Memory({ initial: 1 })
  const kernelManifest = manifest(outputChannelCount)
  const layout: KernelLayout = computeLayout(kernelManifest, 8, 0)
  const kernel = (
    _inputs: number, _registers: number, _arrays: number,
    _arraySizes: number, _temps: number, _sampleRate: number,
    startIdx: bigint, _paramPtrs: number, output: number,
    bufferLen: bigint,
  ) => {
    const frames = Number(bufferLen)
    const view = new Float64Array(
      memory.buffer, output, frames * outputChannelCount)
    for (let frame = 0; frame < frames; frame++)
      for (let channel = 0; channel < outputChannelCount; channel++)
        view[frame * outputChannelCount + channel] =
          sample(frame, channel, Number(startIdx) + frame)
  }

  const value = Object.create(WasmKernel.prototype) as WasmKernel & {
    memory: WebAssembly.Memory
    kernel: typeof kernel
    layout: KernelLayout
    manifest: KernelManifest
    startIdx: bigint
    fadeInRem: number
    fadeOutRem: number
  }
  Object.assign(value, {
    memory,
    kernel,
    layout,
    manifest: kernelManifest,
    maxBlockSize: 8,
    outputChannelCount,
    startIdx: 0n,
    fadeInRem: 0,
    fadeOutRem: 0,
  })
  return value
}

describe('multichannel wasm runtime contract', () => {
  test('old manifests default to mono and invalid widths are refused', () => {
    expect(kernelOutputChannelCount(manifest())).toBe(1)
    expect(() => kernelOutputChannelCount(manifest(0))).toThrow()
    expect(() => kernelOutputChannelCount(manifest(1.5))).toThrow()
  })

  test('linear memory reserves frames times compact channel width', () => {
    const layout = computeLayout(manifest(3), 8, 0)
    expect(layout.endByte - layout.output).toBe(8 * 3 * 8)
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
