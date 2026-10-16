/**
 * The desktop repair-lock poll asks "is the lock holder us?" through
 * `lockHolderIsLiveSync`, whose `ourCt` is `ownCreateTimeSync`. Under Electron
 * `process.getCreationTime()` answers without a spawn, but under plain Node
 * (tests, tooling) on Windows every call fell through to a blocking powershell
 * CIM query (0.33-1.15 s each) for a value that cannot change while the process
 * lives. The own-pid creation time is read once and kept for the life of the
 * process; a failed read is never kept, so a transient powershell failure does
 * not pin "unknown" forever.
 *
 * The Windows path is exercised on any host: `process.platform` is spoofed to
 * win32 and `execFileSync` is stubbed, so the spawn ledger is exact.
 */

import assert from 'node:assert/strict'
import type * as ChildProcess from 'node:child_process'

import { afterEach, beforeEach, test, vi } from 'vitest'

const spawns = vi.hoisted(() => ({ calls: [] as string[], answers: [] as Array<string | Error> }))

// .NET ticks for 2026-01-01T00:00:00Z: (unix seconds * 1e7) + 621355968000000000.
const TICKS_2026 = '639028224000000000'
const UNIX_2026 = 1767225600

vi.mock('node:child_process', async importOriginal => {
  const actual = await importOriginal<typeof ChildProcess>()

  return {
    ...actual,
    execFileSync: ((file: string) => {
      spawns.calls.push(file)
      const next = spawns.answers.shift() ?? TICKS_2026

      if (next instanceof Error) {
        throw next
      }

      return next
    }) as typeof actual.execFileSync
  }
})

const platform = Object.getOwnPropertyDescriptor(process, 'platform')!

beforeEach(() => {
  vi.resetModules()
  spawns.calls.length = 0
  spawns.answers.length = 0
  Object.defineProperty(process, 'platform', { ...platform, value: 'win32' })
})

afterEach(() => {
  Object.defineProperty(process, 'platform', platform)
  delete (process as { getCreationTime?: unknown }).getCreationTime
})

test('40 repair-lock polls under plain Node spawn powershell once, not 40 times', async () => {
  const { lockHolderIsLiveSync } = await import('./update-marker')
  const nowS = Math.floor(Date.now() / 1000)

  for (let poll = 0; poll < 40; poll++) {
    assert.equal(lockHolderIsLiveSync(process.pid, UNIX_2026, nowS), true)
  }

  assert.equal(spawns.calls.length, 1, `powershell spawned ${spawns.calls.length} times`)
})

test('ownCreateTimeSync returns the same creation time on every call', async () => {
  const { ownCreateTimeSync } = await import('./update-marker')
  const first = ownCreateTimeSync()

  assert.equal(first, UNIX_2026)

  for (let i = 0; i < 5; i++) {
    assert.equal(ownCreateTimeSync(), first)
  }

  assert.equal(spawns.calls.length, 1)
})

test('a failed read is not kept: the next call retries and then memoizes', async () => {
  const { ownCreateTimeSync } = await import('./update-marker')

  spawns.answers.push(new Error('powershell.exe not responding'))
  assert.equal(ownCreateTimeSync(), null)
  assert.equal(ownCreateTimeSync(), UNIX_2026)
  assert.equal(ownCreateTimeSync(), UNIX_2026)
  assert.equal(spawns.calls.length, 2)
})

test('Electron (process.getCreationTime present) never spawns', async () => {
  ;(process as { getCreationTime?: () => number }).getCreationTime = () => UNIX_2026 * 1000

  const { ownCreateTimeSync } = await import('./update-marker')

  for (let i = 0; i < 5; i++) {
    assert.equal(ownCreateTimeSync(), UNIX_2026)
  }

  assert.equal(spawns.calls.length, 0)
})
