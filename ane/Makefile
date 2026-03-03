# Build the ANE bridge library for Python ctypes access
# Requires macOS 15+ with Apple Silicon (M1/M2/M3/M4)

DYLIB = libane_bridge.dylib
SRC = ane_bridge.m

.PHONY: all clean test bench

all: $(DYLIB)

$(DYLIB): $(SRC) ane_bridge.h
	xcrun clang -O2 -shared -o $@ $< -framework Foundation -framework IOSurface -fobjc-arc

test: $(DYLIB)
	python3 test_dyn_matmul.py

bench: $(DYLIB)
	python3 bench_full.py

clean:
	rm -f $(DYLIB)
