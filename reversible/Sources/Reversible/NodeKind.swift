import SwiftUI

struct KnobSpec {
    let name: String
    let min: Double
    let max: Double
    let def: Double
    var log = false
    var unit = ""
}

/// Node vocabulary, 1:1 with the Lean arrow patch-graph `Node` cases
/// (ported from playground/renderer/app.js KINDS). You patch DOWNSTREAM
/// only; the frontend's lowering threads each effect's (possibly
/// signal-modulated) warp onto the generators' clocks — totally
/// composable, so effects stack freely.
enum NodeKind: String, CaseIterable, Codable {
    case source, knob, flange, sflange, fm, delay, reverse, mix, ring
    case resonator, reverb, modalmix
    case scope
    case out
}

struct NodeSpec {
    let title: String
    let accent: Color
    let summing: Bool
    let inlets: [String]
    let outlets: [String]
    let knobs: [KnobSpec]
    var modal = false
    var fixed = false
    /// A monitor lives only on the surface: it is never serialized into the
    /// engine graph (its connections select scope taps), and its knobs are
    /// local view state, not live param slots.
    var monitor = false
    var gridOverride: (w: Int, h: Int)? = nil

    /// Module footprint in grid units (VCV-style: every module has a defined
    /// width and height on the grid). Derived from content: 54px per knob +
    /// chrome for width; a two-line identity header, knob block, and jacks for
    /// height. Six units is the minimum that can show title, truth, and id
    /// without SwiftUI compressing each label into an ellipsis.
    var gridSize: (w: Int, h: Int) {
        if let o = gridOverride { return o }
        let w = max(6, Int(ceil((Double(knobs.count) * 54 + Double(max(0, knobs.count - 1)) * 8 + 18) / Grid.unit)))
        let h = knobs.isEmpty ? 4 : 7
        return (w, h)
    }
}

enum Grid {
    static let unit: Double = 22

    static func snap(_ p: CGPoint) -> CGPoint {
        CGPoint(
            x: (p.x / unit).rounded() * unit,
            y: (p.y / unit).rounded() * unit)
    }
}

extension NodeKind {
    var spec: NodeSpec {
        switch self {
        case .source:
            // Always a MorphOsc: morph = 0 is saw, morph = 1 is sine, so the
            // morph knob is always meaningful. `freq` is a CONTROL inlet —
            // patch a Knob to drive pitch from a live slot (only a knob;
            // audio-rate into the pitch port is not FM).
            return NodeSpec(
                title: "Osc", accent: Color(hex: 0x6CC7FF), summing: false,
                inlets: ["freq"], outlets: ["out"],
                knobs: [
                    // spans sub-Hz LFO → audio (log): dial it below ~1 Hz and
                    // it's a slow ramp you can patch into a Reson `addr` to
                    // scrub-trigger the resonance. A phasor is closed-form at
                    // any rate, so there's no low-frequency floor in the engine.
                    KnobSpec(name: "freq", min: 0.02, max: 2000, def: 220, log: true, unit: "Hz"),
                    KnobSpec(name: "morph", min: 0, max: 1, def: 0),
                ])
        case .knob:
            // A program that is nothing but a param with one output — for a
            // value you want to PATCH (fan it out, or into a `freq`/`mod`
            // control inlet). Every knob is live; this one is also a
            // first-class wire source.
            return NodeSpec(
                title: "Knob", accent: Color(hex: 0xB7A4FF), summing: false,
                inlets: [], outlets: ["out"],
                knobs: [KnobSpec(name: "value", min: 0, max: 1000, def: 220)])
        case .flange:
            // static δ comb — the pure plain-warp slide. `depth` is the GLIDE
            // prototype: the engine-owned discipline drives a closed-form
            // ramp, so A/B it against a raw parameter to hear the difference.
            return NodeSpec(
                title: "Flange", accent: Color(hex: 0xFFCC66), summing: false,
                inlets: ["in"], outlets: ["out"],
                knobs: [KnobSpec(name: "depth", min: 0.0001, max: 0.01, def: 0.002, log: true, unit: "s")])
        case .sflange:
            // SIGNAL-modulated comb. Patch a signal into `mod` to sweep it;
            // leave it open and the built-in LFO at `rate` drives the sweep.
            // The signal-warp composes.
            return NodeSpec(
                title: "SwFlange", accent: Color(hex: 0xFFD089), summing: false,
                inlets: ["in", "mod"], outlets: ["out"],
                knobs: [
                    KnobSpec(name: "depth", min: 0.0002, max: 0.02, def: 0.005, log: true, unit: "s"),
                    KnobSpec(name: "rate", min: 0.02, max: 12, def: 0.3, log: true, unit: "Hz"),
                ])
        case .fm:
            return NodeSpec(
                title: "FM", accent: Color(hex: 0xFF9EC7), summing: false,
                inlets: ["in"], outlets: ["out"],
                knobs: [
                    KnobSpec(name: "carrier", min: 20, max: 2000, def: 330, log: true, unit: "Hz"),
                    KnobSpec(name: "depth", min: 1, max: 400, def: 60, log: true),
                ])
        case .delay:
            return NodeSpec(
                title: "Delay", accent: Color(hex: 0x86E8C0), summing: false,
                inlets: ["in"], outlets: ["out"],
                knobs: [KnobSpec(name: "amount", min: 0.0001, max: 0.02, def: 0.004, log: true, unit: "s")])
        case .reverse:
            // clk -> -clk : the moat op, no parameter.
            return NodeSpec(
                title: "Reverse", accent: Color(hex: 0x7AD0AA), summing: false,
                inlets: ["in"], outlets: ["out"], knobs: [])
        case .mix:
            return NodeSpec(
                title: "Mix", accent: Color(hex: 0xC7CED9), summing: true,
                inlets: ["in"], outlets: ["out"], knobs: [])
        case .ring:
            // multiplicative fan-in (⊗) — the ring-product twin of Mix's sum.
            // Two inputs is ring modulation; input × an oscillator/LFO is a
            // VCA. A downstream warp reclocks every factor (the slide
            // distributes over the product).
            return NodeSpec(
                title: "Ring ⊗", accent: Color(hex: 0xD0A0E0), summing: true,
                inlets: ["in"], outlets: ["out"], knobs: [])

        // ── MODAL ISLAND ──────────────────────────────────────────────────
        // The second arena: pole / exp-poly banks that compose by the residue
        // calculus at BUILD time and realize to a Sig at their boundary. A
        // modal OUTLET carries poles, not audio — it may feed a Sig inlet
        // (realized at the seam by `lowerInput`) or another modal node; a
        // modal INLET (Reverb/Modal∪) accepts ONLY a modal node (a Sig source
        // there is ill-typed — see `wantsModal`).
        case .resonator:
            // A struck resonator: 6 harmonics of `freq` decaying at rate
            // `decay`. A modal SOURCE struck at τ=0 — it rings then decays, so
            // scrub the master clock to re-strike. Poles stay LIVE (the
            // residue is emitted symbolically). The `addr` inlet is TEMPORAL,
            // not spectral: patch a CF signal (LFO/env/osc) and its value
            // BECOMES the bank's time-address (seconds into the impulse
            // response) — an ABSOLUTE warp, so the resonance TRIGGERS as the
            // signal crosses zero and scrubs/pitches with its slope. Unlike a
            // downstream sflange (a ±δ vibrato around the master clock), this
            // relocates the strike to the signal. Unpatched ⇒ reads the
            // master clock as before.
            return NodeSpec(
                title: "Reson", accent: Color(hex: 0x4FD6C4), summing: false,
                inlets: ["addr"], outlets: ["out"],
                knobs: [
                    KnobSpec(name: "freq", min: 20, max: 2000, def: 220, log: true, unit: "Hz"),
                    KnobSpec(name: "decay", min: 0.5, max: 50, def: 4, log: true),
                ],
                modal: true)
        case .reverb:
            // A room bank (32 log-spaced modes, 60–6000 Hz) composed with its
            // MODAL input. Modal in → modal out, feeding another modal stage or
            // a Sig inlet where the complete value is realized.
            //
            // For the room kernel h, `dir` selects
            // h[dir] = (1-dir)·h + dir·T(h), and this node convolves h[dir] with
            // its input. Direction is local to this room: it never reverses the
            // upstream modal value or the complete composed output.
            return NodeSpec(
                title: "Reverb", accent: Color(hex: 0x3FB8B0), summing: false,
                inlets: ["in"], outlets: ["out"],
                knobs: [
                    KnobSpec(name: "rt60", min: 0.2, max: 12, def: 2, log: true, unit: "sec"),
                    // dir crossfades this room kernel: 0 = forward, 1 =
                    // reversed. It keeps σ/ω fixed and is independent of
                    // complete-output reversal by the master clock.
                    KnobSpec(name: "dir", min: 0, max: 1, def: 0),
                    // decay SWAY: the room breathes — σ modulated on the
                    // envelope clock only, so the tail's decay time undulates
                    // while pitch holds. Continuous, stateless.
                    KnobSpec(name: "sway", min: 0, max: 0.9, def: 0),
                    KnobSpec(name: "rate", min: 0.05, max: 8, def: 0.3, log: true, unit: "Hz"),
                ],
                modal: true)
        case .modalmix:
            // Pole UNION (∪) of its modal inputs — the modal twin of Mix's
            // sum. Many modal in → one modal out; no knobs (structural).
            return NodeSpec(
                title: "Modal ∪", accent: Color(hex: 0x6FE0D0), summing: true,
                inlets: ["in"], outlets: ["out"], knobs: [],
                modal: true)
        case .scope:
            // A MONITOR, not a processor: never enters the engine graph.
            // Each channel selects a node whose tap slot the poller reads via
            // `render_window` — random-access closed-form evaluation on the
            // C++ data plane, so tracing is free of the audio path and stays
            // live through compiles. Patching a channel to a non-Out node
            // asks the next relower for `taps: true` (each tap re-emits its
            // upstream cone, so taps are paid for only while a scope looks).
            // `window` is view state (the trace's time span), not a param.
            return NodeSpec(
                title: "Scope", accent: Color(hex: 0x7FE08C), summing: false,
                inlets: ["ch1", "ch2", "ch3", "ch4"], outlets: [],
                knobs: [KnobSpec(name: "window", min: 0.002, max: 0.35, def: 0.02, log: true, unit: "s")],
                monitor: true, gridOverride: (w: 16, h: 12))
        case .out:
            return NodeSpec(
                title: "Out · dac", accent: Color(hex: 0xFF8A8A), summing: true,
                inlets: ["in"], outlets: [], knobs: [],
                fixed: true)
        }
    }

    /// Control inlets accept a Knob node's output (a control value, not audio).
    static let controlInlets: Set<String> = ["freq", "mod"]

    /// Inlets that carry POLES, not audio: a modal node's input (Reverb's
    /// `in`, Modal∪'s `in`). They accept ONLY a modal-output node — the
    /// residue calculus composes pole banks at build time (`lowerModal`), and
    /// a Sig source there would throw at compile. A Sig inlet, by contrast,
    /// accepts either: a modal source realizes to a Sig at the seam
    /// (`lowerInput`), but never the reverse.
    static let modalInlets: Set<String> = ["reverb:in", "modalmix:in"]

    func wantsModal(inlet: String) -> Bool {
        Self.modalInlets.contains("\(rawValue):\(inlet)")
    }
}

// ── Knob value ↔ normalized position math ─────────────────────────────────
extension KnobSpec {
    func toNorm(_ v: Double) -> Double {
        if log { return (Foundation.log(v) - Foundation.log(min)) / (Foundation.log(max) - Foundation.log(min)) }
        return (v - min) / (max - min)
    }

    func fromNorm(_ t: Double) -> Double {
        let t = Swift.min(1, Swift.max(0, t))
        if log { return exp(Foundation.log(min) + t * (Foundation.log(max) - Foundation.log(min))) }
        return min + t * (max - min)
    }

    func format(_ v: Double) -> String {
        switch unit {
        case "s": return String(format: "%.\(v < 0.001 ? 2 : 1)fms", v * 1000)
        case "sec": return String(format: "%.2fs", v)   // whole-second times (rt60)
        case "Hz": return v >= 100 ? String(format: "%.0fHz", v) : String(format: "%.2fHz", v)
        default: return String(format: "%.2f", v)
        }
    }
}
