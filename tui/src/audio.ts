/**
 * DAC — Digital-to-Analog Converter. Port of egress/audio.py.
 */

import * as b from './bindings.js'

export interface DacStats {
  callbackCount:  number
  avgCallbackMs:  number
  maxCallbackMs:  number
  underrunCount:  number
  overrunCount:   number
}

export interface DeviceInfo {
  id:                   number
  name:                 string
  outputChannels:       number
  inputChannels:        number
  isDefaultOutput:      boolean
  preferredSampleRate:  number
  sampleRates:          number[]
}

const _registry = new FinalizationRegistry((handle: unknown) => {
  b.egress_dac_free(handle)
})

export class DAC {
  _h: unknown
  private _graph: unknown

  constructor(graphHandle: unknown, sampleRate = 44100, channels = 2) {
    this._graph = graphHandle
    this._h = b.check(b.egress_dac_new(graphHandle, sampleRate, channels), 'dac_new')
    _registry.register(this, this._h, this)
  }

  /** Create a DAC backed by a FlatRuntime instead of a Graph. */
  static fromRuntime(runtimeHandle: unknown, sampleRate = 44100, channels = 2): DAC {
    const dac = Object.create(DAC.prototype) as DAC
    dac._graph = runtimeHandle
    dac._h = b.check(b.egress_dac_new_runtime(runtimeHandle, sampleRate, channels), 'dac_new_runtime')
    _registry.register(dac, dac._h, dac)
    return dac
  }

  start(): void {
    b.egress_dac_start(this._h)
  }

  stop(): void {
    b.egress_dac_stop(this._h)
  }

  get isRunning(): boolean {
    return b.egress_dac_is_running(this._h) as boolean
  }

  get isReconnecting(): boolean {
    return b.egress_dac_is_reconnecting(this._h) as boolean
  }

  callbackStats(): DacStats {
    const s: Record<string, unknown> = {}
    b.egress_dac_get_stats(this._h, s)
    return {
      callbackCount: s.callback_count as number,
      avgCallbackMs: s.avg_callback_ms as number,
      maxCallbackMs: s.max_callback_ms as number,
      underrunCount: s.underrun_count as number,
      overrunCount:  s.overrun_count as number,
    }
  }

  resetStats(): void {
    b.egress_dac_reset_stats(this._h)
  }

  get activeDevice(): number {
    return b.egress_dac_get_active_device(this._h) as number
  }

  switchDevice(deviceId: number): boolean {
    return b.egress_dac_switch_device(this._h, deviceId) as boolean
  }

  dispose(): void {
    _registry.unregister(this)
    b.egress_dac_free(this._h)
    this._h = null
  }

  // ---------- Static device enumeration ----------

  static listDevices(): DeviceInfo[] {
    const count = b.egress_audio_device_count() as number
    if (count === 0) return []

    const ids = new Uint32Array(count)
    b.egress_audio_get_device_ids(ids, count)

    const devices: DeviceInfo[] = []
    for (const deviceId of ids) {
      const info: Record<string, unknown> = {}
      if (b.egress_audio_get_device_info(deviceId, info)) {
        const name = Array.isArray(info.name)
          ? b.decodeCharArray(info.name as number[])
          : (info.name as string)
        const rateCount = info.sample_rate_count as number
        // koffi returns fixed C arrays as objects with numeric keys, not JS arrays
        const ratesRaw = info.sample_rates as Record<number, number>
        const sampleRates: number[] = Array.from({ length: rateCount }, (_, i) => ratesRaw[i])
        devices.push({
          id:                  info.id as number,
          name,
          outputChannels:      info.output_channels as number,
          inputChannels:       info.input_channels as number,
          isDefaultOutput:     info.is_default_output as boolean,
          preferredSampleRate: info.preferred_sample_rate as number,
          sampleRates,
        })
      }
    }
    return devices
  }

  static defaultDevice(): number {
    return b.egress_audio_default_output_device() as number
  }
}
