/**
 * markdown.test.ts — coverage for the .trop markdown extractor (Phase B1).
 */

import { describe, test, expect } from 'bun:test'
import { extractMarkdown, joinBlocks } from './markdown.js'

describe('markdown extractor — basic', () => {
  test('empty document', () => {
    const ext = extractMarkdown('')
    expect(ext.blocks).toEqual([])
  })

  test('document with no fenced blocks', () => {
    const src = '# Heading\n\nSome prose here.\n'
    const ext = extractMarkdown(src)
    expect(ext.blocks).toEqual([])
    expect(ext.lines.length).toBe(4)
  })

  test('single tropical block', () => {
    const src = [
      '# Title',
      '',
      '```tropical',
      'program Foo() -> () { }',
      '```',
      '',
      'After.',
    ].join('\n')
    const ext = extractMarkdown(src)
    expect(ext.blocks).toHaveLength(1)
    expect(ext.blocks[0].source).toBe('program Foo() -> () { }')
    expect(ext.blocks[0].lineOffset).toBe(3)  // line right after the opening fence (0-indexed: line 3)
  })

  test('multiple tropical blocks preserve source order', () => {
    const src = [
      '```tropical',
      'A',
      '```',
      '',
      'prose',
      '',
      '```tropical',
      'B',
      'C',
      '```',
    ].join('\n')
    const ext = extractMarkdown(src)
    expect(ext.blocks.map(b => b.source)).toEqual(['A', 'B\nC'])
  })

  test('lineOffset points to the first content line of each block', () => {
    const src = [
      '# Heading',           // line 0
      '',                    // line 1
      '```tropical',         // line 2 (opening fence)
      'first',                // line 3 ← lineOffset of block 0
      'second',               // line 4
      '```',                 // line 5 (closing fence)
      '',                    // line 6
      'prose',               // line 7
      '```tropical',         // line 8 (opening fence)
      'third',               // line 9 ← lineOffset of block 1
      '```',                 // line 10
    ].join('\n')
    const ext = extractMarkdown(src)
    expect(ext.blocks[0].lineOffset).toBe(3)
    expect(ext.blocks[1].lineOffset).toBe(9)
  })
})

describe('markdown extractor — fence variations', () => {
  test('non-tropical fenced blocks are ignored', () => {
    const src = [
      '```mermaid',
      'graph LR',
      'a --> b',
      '```',
      '',
      '```tropical',
      'X',
      '```',
    ].join('\n')
    const ext = extractMarkdown(src)
    expect(ext.blocks).toHaveLength(1)
    expect(ext.blocks[0].source).toBe('X')
  })

  test('untagged fences (no language) are ignored', () => {
    const src = [
      '```',
      'just a code block',
      '```',
      '',
      '```tropical',
      'Y',
      '```',
    ].join('\n')
    const ext = extractMarkdown(src)
    expect(ext.blocks).toHaveLength(1)
    expect(ext.blocks[0].source).toBe('Y')
  })

  test('extra info string after language tag is allowed', () => {
    const src = [
      '```tropical title="OnePole"',
      'program Z() -> () { }',
      '```',
    ].join('\n')
    const ext = extractMarkdown(src)
    expect(ext.blocks).toHaveLength(1)
    expect(ext.blocks[0].source).toBe('program Z() -> () { }')
  })

  test('4-backtick fences are recognized and matched', () => {
    const src = [
      '````tropical',
      'has ``` inside',
      '````',
    ].join('\n')
    const ext = extractMarkdown(src)
    expect(ext.blocks).toHaveLength(1)
    expect(ext.blocks[0].source).toBe('has ``` inside')
  })

  test('closing fence must match or exceed opening backtick count', () => {
    // Opening with 4 backticks; a 3-backtick line is NOT the closer.
    const src = [
      '````tropical',
      'A',
      '```',  // not the closer (only 3 backticks)
      'B',
      '````',  // closer (4 backticks)
    ].join('\n')
    const ext = extractMarkdown(src)
    expect(ext.blocks).toHaveLength(1)
    expect(ext.blocks[0].source).toBe('A\n```\nB')
  })

  test('unterminated fence extends to end of document (CommonMark)', () => {
    const src = [
      'prose',
      '```tropical',
      'X',
      'Y',
      // no closing fence
    ].join('\n')
    const ext = extractMarkdown(src)
    expect(ext.blocks).toHaveLength(1)
    expect(ext.blocks[0].source).toBe('X\nY')
  })
})

describe('markdown extractor — preserves structure', () => {
  test('lines field preserves the original document line count', () => {
    const src = 'a\nb\nc\n'
    const ext = extractMarkdown(src)
    // 'a\nb\nc\n'.split('\n') is ['a','b','c',''] — 4 entries
    expect(ext.lines).toEqual(['a', 'b', 'c', ''])
  })

  test('blocks ignore tildes (only backtick fences supported)', () => {
    const src = [
      '~~~tropical',
      'should not be extracted',
      '~~~',
    ].join('\n')
    const ext = extractMarkdown(src)
    expect(ext.blocks).toEqual([])
  })
})

describe('markdown extractor — joinBlocks', () => {
  test('concatenates with blank-line separators', () => {
    const src = [
      '```tropical',
      'first',
      '```',
      '',
      '```tropical',
      'second',
      '```',
    ].join('\n')
    const joined = joinBlocks(extractMarkdown(src))
    expect(joined).toBe('first\n\nsecond')
  })
})
