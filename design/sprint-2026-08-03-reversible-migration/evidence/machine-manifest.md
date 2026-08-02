# Sprint machine manifest

Recorded at sprint kickoff on 2026-08-02 in `America/Los_Angeles`.

```text
candidate_worktree=/private/tmp/tropical-reversible-migration-corrected
integration_baseline=07a6b2517f8d24c8822d67e0337c0fd99d016bd8
standalone_source_head=dea822ea1062a749ea1d7a76af1e2bd28194dfa1
os=macOS 26.3 (25D125)
kernel=Darwin 25.3.0 arm64
cpu_gpu=Apple M1 Pro; 16 GPU cores; Metal supported
swift=Apple Swift 6.2.3 (swiftlang-6.2.3.3.21 clang-1700.6.3.2)
swift_target=arm64-apple-macosx26.0
clang=Apple clang 17.0.0 (clang-1700.6.3.2)
lean=4.29.1 f72c35b3f637c8c6571d353742168ab66cc22c00
lake=5.0.0-src+f72c35b
bun=1.3.12
cmake=4.2.3
developer_directory=/Library/Developer/CommandLineTools
xcode_application=not selected/available at kickoff
audio_device=not reported by system_profiler in the execution environment
display_refresh=not reported by system_profiler in the execution environment
```

The command-line Swift toolchain is available. Full Xcode is not selected, so
Finder launch/UI automation and any gate needing Xcode-specific tools must be
run later on the qualified Mac or after selecting an installed Xcode. Audio
device, render buffer, device buffer, and refresh-rate evidence are still
required before a release-candidate label.
