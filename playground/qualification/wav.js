'use strict'

const { writeFileSync } = require('node:fs')

function writeMonoFloat64Pcm24(path, samples, sampleRate) {
  let peak = 0
  for (const value of samples) {
    if (!Number.isFinite(value)) throw new Error('cannot write non-finite WAV sample')
    peak = Math.max(peak, Math.abs(value))
  }
  // Evidence is never limited. Refuse an overflowing capture instead of hiding it.
  if (peak >= 1) throw new Error(`capture peak ${peak} does not fit PCM24`)
  const dataBytes = samples.length * 3
  const buffer = Buffer.alloc(44 + dataBytes)
  buffer.write('RIFF', 0)
  buffer.writeUInt32LE(36 + dataBytes, 4)
  buffer.write('WAVEfmt ', 8)
  buffer.writeUInt32LE(16, 16)
  buffer.writeUInt16LE(1, 20)
  buffer.writeUInt16LE(1, 22)
  buffer.writeUInt32LE(sampleRate, 24)
  buffer.writeUInt32LE(sampleRate * 3, 28)
  buffer.writeUInt16LE(3, 32)
  buffer.writeUInt16LE(24, 34)
  buffer.write('data', 36)
  buffer.writeUInt32LE(dataBytes, 40)
  samples.forEach((value, index) => {
    let word = Math.round(value * ((1 << 23) - 1))
    if (word < 0) word += 1 << 24
    const offset = 44 + index * 3
    buffer[offset] = word & 0xff
    buffer[offset + 1] = (word >>> 8) & 0xff
    buffer[offset + 2] = (word >>> 16) & 0xff
  })
  writeFileSync(path, buffer)
  return { peak, sampleCount: samples.length }
}

function assembleCaptureTimeline(blocks) {
  if (!blocks.length) return { samples: [], start: 0, gapSamples: 0, overlapSamples: 0 }
  const sorted = [...blocks].sort((a, b) => a.start_sample_index - b.start_sample_index)
  const start = sorted[0].start_sample_index
  const end = Math.max(...sorted.map((block) => block.start_sample_index + block.samples.length))
  const samples = new Array(end - start).fill(0)
  const written = new Uint8Array(samples.length)
  let overlapSamples = 0
  for (const block of sorted) {
    block.samples.forEach((value, index) => {
      const target = block.start_sample_index - start + index
      if (written[target]) overlapSamples += 1
      samples[target] = value
      written[target] = 1
    })
  }
  let gapSamples = 0
  written.forEach((value) => { if (!value) gapSamples += 1 })
  return { samples, start, gapSamples, overlapSamples }
}

module.exports = { assembleCaptureTimeline, writeMonoFloat64Pcm24 }
