// msl_check — compile MSL from stdin through the RUNTIME Metal compiler
// (newLibraryWithSource, the same path the engine uses; no Xcode needed).
// Exit 0 on success; prints compiler diagnostics on failure.
//
//   clang++ -O2 -std=c++20 -fobjc-arc -framework Metal -framework Foundation \
//       engine/metal/msl_check.mm -o build/msl_check
//   diffcli emit-msl patch.json | build/msl_check
#import <Metal/Metal.h>
#import <Foundation/Foundation.h>
#include <iostream>
#include <sstream>

int main() {
    @autoreleasepool {
        std::stringstream ss;
        ss << std::cin.rdbuf();
        const std::string src = ss.str();
        if (src.empty()) { std::cerr << "msl_check: empty input\n"; return 2; }
        id<MTLDevice> dev = MTLCreateSystemDefaultDevice();
        if (!dev) { std::cerr << "msl_check: no Metal device\n"; return 3; }
        MTLCompileOptions* opts = [MTLCompileOptions new];
        opts.mathMode = MTLMathModeSafe;   // the engine's contract: IEEE-ordered f32
        NSError* err = nil;
        id<MTLLibrary> lib = [dev newLibraryWithSource:
                                  [NSString stringWithUTF8String:src.c_str()]
                                              options:opts
                                                error:&err];
        if (!lib) {
            std::cerr << "msl_check: compile FAILED\n"
                      << [[err localizedDescription] UTF8String] << "\n";
            return 1;
        }
        id<MTLFunction> fn = [lib newFunctionWithName:@"tropical_kernel"];
        if (!fn) { std::cerr << "msl_check: no tropical_kernel function\n"; return 1; }
        NSError* perr = nil;
        id<MTLComputePipelineState> pso =
            [dev newComputePipelineStateWithFunction:fn error:&perr];
        if (!pso) {
            std::cerr << "msl_check: PSO FAILED\n"
                      << [[perr localizedDescription] UTF8String] << "\n";
            return 1;
        }
        std::cerr << "msl_check: OK (maxTotalThreadsPerThreadgroup="
                  << pso.maxTotalThreadsPerThreadgroup << ")\n";
        return 0;
    }
}
