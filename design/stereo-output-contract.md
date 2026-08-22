# Independent output-channel contract

Status: accepted for the proof-initiatives sprint.

## Decision

Tropical plans support independent scalar output channels. A `SinkSpec` is one
continuously produced output channel, not an observation tap:

- `SinkSpec.target` is a zero-based logical output-channel index;
- all routes for one channel are collected into one sink;
- sink targets are unique and strictly below `FlatPlan.outputChannelCount`;
- each sink folds its input slots in authored order and then applies `gain`;
- channels without a sink produce zero.

`FlatPlan.outputChannelCount` is positive. Plan-6 encodes it as the optional
`output_channel_count` field, omitted when its value is one. Decoders treat an
absent field as one, so existing mono Plan-6 documents retain their meaning.
This is a correction of the previously unrealized `SinkSpec.target` contract,
not a Plan schema-version bump.

## Kernel output image

The canonical kernel buffer is frame-major interleaved:

```text
output[frame * outputChannelCount + channel]
```

`buffer_length` continues to count sample frames, and the sample clock advances
by frames rather than scalar buffer cells. Fused, microkernel, wasm, and Metal
interpret the same layout.

Mono remains the compatibility case: `outputChannelCount = 1` gives the
existing `output[frame]` layout. Native and browser hosts may broadcast a mono
plan to a wider device. A multichannel plan maps channels independently; a host
must not silently duplicate channel zero over the remaining declared channels.

Runtime output storage is preallocated to a documented maximum and is never
resized by a hot-swap or from the audio callback. The channel count belongs to
the atomically published program generation. Existing mono output APIs remain
channel-zero compatibility views; explicit channel-count and interleaved-output
APIs expose the full image.

## Authoring

Session DAC routes carry an explicit nonnegative channel. Channel zero is the
default and is omitted on the wire, preserving existing `dac.out` documents and
tool calls. Multiple routes to one channel are legal fan-in and compile to one
sink. The compiler orders sinks by ascending channel while preserving authored
route order inside each sink.

Independent stereo is therefore two pushed outputs:

```text
source A -> dac.out, channel 0
source B -> dac.out, channel 1
```

## Scopes remain pull observations

A scope tap is not a sink. It is a named module slot retained in the compiled
plan. `render_window` evaluates the fused kernel at requested coordinates in
observer-owned storage and projects those slots. This keeps the distinction
explicit:

- sinks are mandatory push outputs produced during ordinary execution;
- scope taps are pull queries over an execution state.

The Plan semantics should expose both the target-indexed sink image and the
final slot state. Scope semantics is repeated execution plus slot projection;
it does not participate in sink routing, channel gains, or device negotiation.

## Proof obligations

`FlatPlanWellFormed` must establish:

- `0 < outputChannelCount`;
- every sink target is in range and targets are unique;
- sink inputs address valid scalar module slots;
- the output image has exactly `outputChannelCount` values per frame;
- sink input folding and channel publication are deterministic.

Backend refinements must preserve the frame-major cell mapping. Device buffer
layout conversion, channel negotiation, callback publication, and physical
device behavior remain host/backend obligations rather than part of Plan
denotation.
