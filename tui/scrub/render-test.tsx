// Headless UI test: mount the real App (engine spawns, patch loads, audio
// OFF), drive some input, assert it renders + survives. Run:
//   TUI_NO_AUDIO=1 bun tui/render-test.tsx
import React from 'react'
import { render } from 'ink-testing-library'
import { App } from './app'

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))

const main = async () => {
  process.env.TUI_NO_AUDIO = '1'
  const { lastFrame, stdin, unmount } = render(<App />)
  await sleep(15000) // let engine cold-boot + elaborate stdlib + patch load (arrow path)

  let frame = lastFrame() ?? ''
  if (!frame.includes('reversible')) throw new Error('missing title:\n' + frame)
  if (!frame.includes('loaded (audio off)')) throw new Error('patch did not load:\n' + frame)
  if (!frame.includes('Voice pitch')) throw new Error('params not rendered:\n' + frame)

  // Voice pitch is row 0 — scrub it and confirm the value display changes.
  const before = lastFrame() ?? ''
  for (let i = 0; i < 30; i++) stdin.write('\x1B[D') // left arrow ×30
  await sleep(300)
  frame = lastFrame() ?? ''
  if (frame === before) throw new Error('param did not respond to input:\n' + frame)

  // toggle reset
  stdin.write('r')
  await sleep(200)

  unmount()
  console.log('UI OK — mounted, loaded patch, rendered params, handled input')
  await sleep(100)
  process.exit(0)
}
main().catch((e) => { console.error('UI FAIL:', e.message); process.exit(1) })
