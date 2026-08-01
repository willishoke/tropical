// One immediate request per parameter, with a bounded turning-point + latest
// tail while that request is in flight. This is backpressure, not debounce:
// the first movement reaches the engine immediately, an out-and-back remains
// audible, and obsolete drag positions never form an unbounded Metal FIFO.
(function installLatestValueSender(root, factory) {
  const api = factory()
  if (typeof module === 'object' && module.exports) module.exports = api
  if (root) root.LatestValueSender = api.LatestValueSender
}(typeof globalThis === 'undefined' ? this : globalThis, () => {
  class LatestValueSender {
    constructor(send, onError = () => {}, scalar = null, onBusyChange = () => {}) {
      this.send = send
      this.onError = onError
      this.scalar = scalar
      this.onBusyChange = onBusyChange
      this.states = new Map()
    }

    submit(key, value) {
      let state = this.states.get(key)
      if (!state) {
        state = {
          busy: false,
          hasLatest: false,
          latest: undefined,
          hasTurning: false,
          turning: undefined,
          lastScalar: undefined,
          direction: 0,
          drain: null,
        }
        this.states.set(key, state)
      }
      if (this.scalar) {
        const nextScalar = this.scalar(value)
        if (Number.isFinite(nextScalar) && Number.isFinite(state.lastScalar)) {
          const delta = nextScalar - state.lastScalar
          const nextDirection = Math.sign(delta)
          if (nextDirection !== 0
            && state.direction !== 0
            && nextDirection !== state.direction
            && state.hasLatest
            && !state.hasTurning) {
            // Preserve one reversal point as well as the final position. A
            // fast out-and-back gesture must remain audible even when its
            // endpoint equals the value whose epoch is currently in flight.
            state.turning = state.latest
            state.hasTurning = true
          }
          if (nextDirection !== 0) state.direction = nextDirection
        }
        if (Number.isFinite(nextScalar)) state.lastScalar = nextScalar
      }
      state.latest = value
      state.hasLatest = true
      if (!state.busy) {
        const wasBusy = this.isBusy()
        state.busy = true
        if (!wasBusy) try { this.onBusyChange(true) } catch {}
        state.drain = this.drain(key, state)
      }
      return state.drain
    }

    async drain(key, state) {
      while (state.hasTurning || state.hasLatest) {
        const value = state.hasTurning ? state.turning : state.latest
        if (state.hasTurning) {
          state.turning = undefined
          state.hasTurning = false
        } else {
          state.latest = undefined
          state.hasLatest = false
        }
        try {
          await this.send(key, value)
        } catch (error) {
          try { this.onError(error, key, value) } catch {}
        }
      }
      state.busy = false
      state.direction = 0
      state.drain = null
      if (!this.isBusy()) try { this.onBusyChange(false) } catch {}
    }

    async whenIdle(key) {
      for (;;) {
        const state = this.states.get(key)
        if (!state?.busy) return
        await state.drain
      }
    }

    isBusy(key = undefined) {
      if (key !== undefined) return Boolean(this.states.get(key)?.busy)
      for (const state of this.states.values()) if (state.busy) return true
      return false
    }
  }

  return { LatestValueSender }
}))
