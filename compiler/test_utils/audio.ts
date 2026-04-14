/**
 * test_utils/audio.ts — Buffer renderer and signal analysis utilities for integration testing.
 *
 * The buffer backend is the testing analogue of `runtime.createDAC().start()`:
 * instead of letting RtAudio drive process() asynchronously, renderFrames() drives
 * it synchronously and captures the output for assertion or file output.
 *
 * Usage:
 *   const samples = renderFrames(session.runtime, 174)   // ~1 second at 44100 Hz
 *   expect(peak(samples)).toBeGreaterThan(0.9)
 *   expect(dominantFrequency(samples, 44100)).toBeCloseTo(440, -1)
 *   await writeWav('/tmp/out.wav', samples, 44100)
 */

import { writeFile } from 'node:fs/promises'
import type { Runtime } from '../runtime/runtime'

// ─── Buffer backend ───────────────────────────────────────────────────────────

/**
 * Drive runtime.process() for nCalls iterations and return all collected samples.
 * Total samples = nCalls * runtime.bufferLength.
 *
 * This is the synchronous "buffer backend" — use instead of createDAC().start()
 * when you want deterministic, device-free audio rendering for tests.
 */
export function renderFrames(runtime: Runtime, nCalls: number): Float64Array {
  const bufLen = runtime.bufferLength
  const result = new Float64Array(nCalls * bufLen)
  for (let i = 0; i < nCalls; i++) {
    runtime.process()
    result.set(runtime.outputBuffer, i * bufLen)
  }
  return result
}

// ─── Signal analysis ──────────────────────────────────────────────────────────

/** Maximum absolute sample value. */
export function peak(samples: Float64Array): number {
  let p = 0
  for (let i = 0; i < samples.length; i++) {
    const a = Math.abs(samples[i])
    if (a > p) p = a
  }
  return p
}

/** Root-mean-square amplitude. */
export function rms(samples: Float64Array): number {
  let sum = 0
  for (let i = 0; i < samples.length; i++) sum += samples[i] * samples[i]
  return Math.sqrt(sum / samples.length)
}

// ─── Spectrum analysis ────────────────────────────────────────────────────────

function nextPow2(n: number): number {
  let p = 1
  while (p < n) p <<= 1
  return p
}

/**
 * In-place Cooley-Tukey radix-2 DIT FFT.
 * buf is interleaved [re0, im0, re1, im1, ...], length must be 2 * (power of 2).
 */
function fftInPlace(buf: Float64Array): void {
  const n = buf.length >> 1
  // Bit-reversal permutation
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1
    for (; j & bit; bit >>= 1) j ^= bit
    j ^= bit
    if (i < j) {
      let t = buf[2 * i]; buf[2 * i] = buf[2 * j]; buf[2 * j] = t
      t = buf[2 * i + 1]; buf[2 * i + 1] = buf[2 * j + 1]; buf[2 * j + 1] = t
    }
  }
  // Butterfly stages
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len
    const wr = Math.cos(ang)
    const wi = Math.sin(ang)
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0
      for (let j = 0; j < len >> 1; j++) {
        const ur = buf[2 * (i + j)]
        const ui = buf[2 * (i + j) + 1]
        const vr = buf[2 * (i + j + (len >> 1))] * cr - buf[2 * (i + j + (len >> 1)) + 1] * ci
        const vi = buf[2 * (i + j + (len >> 1))] * ci + buf[2 * (i + j + (len >> 1)) + 1] * cr
        buf[2 * (i + j)] = ur + vr
        buf[2 * (i + j) + 1] = ui + vi
        buf[2 * (i + j + (len >> 1))] = ur - vr
        buf[2 * (i + j + (len >> 1)) + 1] = ui - vi
        const newCr = cr * wr - ci * wi
        ci = cr * wi + ci * wr
        cr = newCr
      }
    }
  }
}

/**
 * Compute the magnitude spectrum of a real-valued signal.
 * Zero-pads the input to the next power of 2.
 * Returns bins [0 .. n/2], length = n/2 + 1, where n is the padded size.
 */
export function magnitudeSpectrum(samples: Float64Array): Float64Array {
  const n = nextPow2(samples.length)
  const buf = new Float64Array(2 * n) // interleaved re/im, im = 0
  for (let i = 0; i < samples.length; i++) buf[2 * i] = samples[i]
  fftInPlace(buf)
  const out = new Float64Array(n / 2 + 1)
  for (let i = 0; i <= n / 2; i++) {
    const re = buf[2 * i]
    const im = buf[2 * i + 1]
    out[i] = Math.sqrt(re * re + im * im)
  }
  return out
}

/**
 * Index of the highest-magnitude bin, excluding DC (bin 0).
 */
export function dominantBin(spectrum: Float64Array): number {
  let maxVal = -Infinity
  let maxIdx = 1
  for (let i = 1; i < spectrum.length; i++) {
    if (spectrum[i] > maxVal) { maxVal = spectrum[i]; maxIdx = i }
  }
  return maxIdx
}

/**
 * Convert a FFT bin index to Hz.
 * nSamples is the original (pre-pad) sample count; sampleRate is in Hz.
 */
export function binToHz(bin: number, nSamples: number, sampleRate: number): number {
  return (bin * sampleRate) / nextPow2(nSamples)
}

/**
 * Find the dominant frequency (Hz) in a real-valued signal, excluding DC.
 */
export function dominantFrequency(samples: Float64Array, sampleRate: number): number {
  const spectrum = magnitudeSpectrum(samples)
  return binToHz(dominantBin(spectrum), samples.length, sampleRate)
}

// ─── WAV output ───────────────────────────────────────────────────────────────

/**
 * Write a mono 32-bit float PCM WAV file.
 * Format: RIFF/WAVE, IEEE_FLOAT (0x0003), 1 channel, 18-byte fmt chunk.
 * Uses Bun.write — only available in Bun runtime.
 */
export async function writeWav(
  path: string,
  samples: Float64Array,
  sampleRate: number,
): Promise<void> {
  const dataBytes = samples.length * 4  // float32 per sample

  // File layout:
  //   RIFF header:  12 bytes ("RIFF" + size + "WAVE")
  //   fmt chunk:    26 bytes ("fmt " + 18 + 18 bytes of data)
  //   data chunk:   8 + dataBytes bytes ("data" + size + samples)
  // Total:          46 + dataBytes
  const totalBytes = 46 + dataBytes
  const riffSize = totalBytes - 8  // everything after "RIFF" + size field

  const buf = new ArrayBuffer(totalBytes)
  const v = new DataView(buf)
  let off = 0

  const cc = (s: string) => { for (let i = 0; i < 4; i++) v.setUint8(off + i, s.charCodeAt(i)); off += 4 }
  const u32 = (x: number) => { v.setUint32(off, x, true); off += 4 }
  const u16 = (x: number) => { v.setUint16(off, x, true); off += 2 }

  cc('RIFF'); u32(riffSize); cc('WAVE')
  cc('fmt '); u32(18)           // fmt chunk: 18-byte body (IEEE_FLOAT requires cbSize)
  u16(3)                        // AudioFormat: IEEE_FLOAT
  u16(1)                        // NumChannels: mono
  u32(sampleRate)               // SampleRate
  u32(sampleRate * 4)           // ByteRate = SampleRate * BlockAlign
  u16(4)                        // BlockAlign = channels * bytesPerSample
  u16(32)                       // BitsPerSample
  u16(0)                        // cbSize: no extra bytes
  cc('data'); u32(dataBytes)
  for (let i = 0; i < samples.length; i++) { v.setFloat32(off, samples[i], true); off += 4 }

  await writeFile(path, new Uint8Array(buf))
}
