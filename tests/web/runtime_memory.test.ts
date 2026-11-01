import { describe, expect, test } from 'bun:test'
import { importedMemoryMinPages } from '../../web/runtime/index'

const leb = (n: number): number[] => {
  const out: number[] = []
  do { let b = n & 0x7f; n >>>= 7; if (n) b |= 0x80; out.push(b) } while (n)
  return out
}
const name = (s: string): number[] => [...leb(s.length), ...new TextEncoder().encode(s)]
const section = (id: number, body: number[]): number[] => [id, ...leb(body.length), ...body]
const module = (...sections: number[][]): Uint8Array =>
  new Uint8Array([0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00, ...sections.flat()])

describe('imported memory minimum', () => {
  // A function import before the memory, multi-byte LEB sizes, and a max.
  const withMemory = module(
    section(1, [1, 0x60, 0, 0]),                         // type: () -> ()
    section(2, [2,
      ...name('env'), ...name('round'), 0, 0,            // func import, type 0
      ...name('env'), ...name('memory'), 2, 1, ...leb(300), ...leb(400)]))

  test('reads the declared minimum past other imports', () => {
    expect(WebAssembly.validate(withMemory)).toBe(true)
    expect(importedMemoryMinPages(withMemory)).toBe(300)
  })

  test('accepts any BufferSource view', () => {
    const padded = new Uint8Array(withMemory.length + 3)
    padded.set(withMemory, 3)
    expect(importedMemoryMinPages(padded.subarray(3))).toBe(300)
    expect(importedMemoryMinPages(withMemory.buffer)).toBe(300)
  })

  test('a module that imports no memory declares nothing', () => {
    const empty = module()
    expect(WebAssembly.validate(empty)).toBe(true)
    expect(importedMemoryMinPages(empty)).toBe(0)
  })
})
