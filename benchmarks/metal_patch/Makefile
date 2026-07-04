# ModalHeavy64 patch on Metal — fitness (vs tropical f64 render) + realtime latency.
# Reference mh64_ref.bin is produced by:
#   diffcli compile modal_heavy64.json | diffcli render-bytes - (see README).
CXX        ?= clang++
CXXFLAGS   ?= -O3 -std=c++20 -fobjc-arc -Wall -Wno-deprecated-declarations
FRAMEWORKS := -framework Metal -framework Foundation

modal_patch: modal_patch.mm modal_params.h
	$(CXX) $(CXXFLAGS) $(FRAMEWORKS) modal_patch.mm -o modal_patch

run: modal_patch
	./modal_patch

clean:
	rm -f modal_patch

.PHONY: run clean
