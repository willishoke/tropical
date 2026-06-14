cmake_minimum_required(VERSION 3.20)
project(tropical LANGUAGES C CXX)

# Default to RelWithDebInfo (-O2 -g -DNDEBUG): optimized but with debug symbols
# for profiling tools (Instruments, perf, etc.).  Override with e.g.
# -DCMAKE_BUILD_TYPE=Release for maximum throughput or =Debug for debugging.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

option(TROPICAL_BUILD_PYTHON "Build ctypes-based Python package support (installs tropical/ package)" ON)
option(TROPICAL_LLVM_STATIC "Link LLVM statically into libtropical for self-contained wheel distribution" OFF)
option(TROPICAL_WASM_EMIT "Build the in-process IR→wasm emitter (links the LLVM WebAssembly backend + lld as a library). Needed by the build-time compile-wasm tool, not by the shipped audio runtime." OFF)

# nlohmann/json — header-only JSON library (for plan loading)
set(JSON_BuildTests OFF CACHE INTERNAL "")
set(JSON_Install OFF CACHE INTERNAL "")
add_subdirectory(lib/json)

# RtAudio library target
add_library(rtaudio STATIC lib/rtaudio/RtAudio.cpp)
target_include_directories(rtaudio SYSTEM PUBLIC lib/rtaudio)
target_compile_options(rtaudio PRIVATE -Wno-vla-cxx-extension)
set_target_properties(rtaudio PROPERTIES POSITION_INDEPENDENT_CODE ON)
if(APPLE)
    target_compile_definitions(rtaudio PUBLIC __MACOSX_CORE__)
    target_link_libraries(rtaudio PUBLIC
        "-framework CoreAudio"
        "-framework CoreFoundation"
        pthread)
elseif(UNIX)
    target_compile_definitions(rtaudio PUBLIC __LINUX_ALSA__)
    find_package(ALSA REQUIRED)
    target_link_libraries(rtaudio PUBLIC ALSA::ALSA pthread)
endif()

# Core C++ sources shared by all targets
set(TROPICAL_CORE_SOURCES
    engine/runtime/FlatRuntime.cpp
    engine/c_api/tropical_c.cpp)

# libtropical — stable C shared library (no Python dependency)
add_library(tropical_core SHARED ${TROPICAL_CORE_SOURCES})

find_package(LLVM CONFIG REQUIRED)
if(LLVM_PACKAGE_VERSION VERSION_LESS 19)
    message(FATAL_ERROR "LLVM >= 19 required (getOrInsertDeclaration); found ${LLVM_PACKAGE_VERSION}")
endif()
message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION}")

target_sources(tropical_core PRIVATE engine/jit/OrcJitEngine.cpp)
target_include_directories(tropical_core PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(tropical_core PRIVATE ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(LLVM_ORC_JIT_LIBS
    Core OrcJIT ExecutionEngine native nativecodegen Support)
target_link_libraries(tropical_core PRIVATE ${LLVM_ORC_JIT_LIBS})

if(TROPICAL_WASM_EMIT)
    if(LLVM_PACKAGE_VERSION VERSION_LESS 21)
        message(FATAL_ERROR "TROPICAL_WASM_EMIT needs LLVM >= 21 (modern Triple / "
            "createTargetMachine API); found ${LLVM_PACKAGE_VERSION}. Build with "
            "-DTROPICAL_WASM_EMIT=OFF for the JIT-only runtime.")
    endif()
    # In-process IR→wasm: the WebAssembly codegen backend (the same libLLVM
    # the ORC JIT already links) plus lld's wasm driver as a library. No
    # subprocess, no PATH, no version skew — the wasm comes out of the exact
    # LLVM the JIT uses, which is what makes native≡wasm bit-exactness
    # structural rather than coincidental. ~+3-4MB; gated so the shipped audio
    # runtime can exclude it and only the build-time `diffcli` carries it.
    llvm_map_components_to_libnames(LLVM_WASM_LIBS
        WebAssemblyCodeGen WebAssemblyDesc WebAssemblyInfo)
    target_link_libraries(tropical_core PRIVATE ${LLVM_WASM_LIBS})

    # lld ships in the `lld` keg on Homebrew and in llvm-N's libdir on apt.llvm.org,
    # so hint both the Homebrew prefix and LLVM's own include/lib dirs (Linux).
    find_path(LLD_INCLUDE_DIR lld/Common/Driver.h
        HINTS /opt/homebrew/opt/lld/include ${LLVM_INCLUDE_DIRS} REQUIRED)
    find_library(LLD_WASM_LIB lldWasm
        HINTS /opt/homebrew/opt/lld/lib ${LLVM_LIBRARY_DIRS} REQUIRED)
    find_library(LLD_COMMON_LIB lldCommon
        HINTS /opt/homebrew/opt/lld/lib ${LLVM_LIBRARY_DIRS} REQUIRED)
    target_include_directories(tropical_core PRIVATE ${LLD_INCLUDE_DIR})
    if(APPLE)
        # Homebrew's lld dylibs record their own libLLVM dependency, so linking
        # the two lld libs is enough.
        target_link_libraries(tropical_core PRIVATE ${LLD_WASM_LIB} ${LLD_COMMON_LIB})
    else()
        # apt.llvm.org ships lld + LLVM as *static* archives, and lld references
        # the full target-init + LTO surface (99 LLVMInitialize* + 12 llvm::lto::*).
        # Pull every LLVM static lib from llvm-config as raw -l flags — CMake
        # preserves those in order, whereas it relocates imported targets (e.g.
        # the names from llvm_map_components_to_libnames) out of an explicit
        # --start-group. Wrap lld + LLVM in --start-group so the circular static
        # archives resolve regardless of order.
        execute_process(
            COMMAND ${LLVM_TOOLS_BINARY_DIR}/llvm-config --link-static --libs all-targets lto
            OUTPUT_VARIABLE TROPICAL_LLVM_ALL_LIBS
            OUTPUT_STRIP_TRAILING_WHITESPACE)
        separate_arguments(TROPICAL_LLVM_ALL_LIBS)
        # `lto` → LLVMPasses references Polly (apt's LLVM is built with Polly
        # linked into the pass pipeline), so -lPolly/-lPollyISL must resolve;
        # CI installs libpolly-N-dev for the static archives.
        target_link_libraries(tropical_core PRIVATE
            "-L${LLVM_LIBRARY_DIRS}"
            -Wl,--start-group
            ${LLD_WASM_LIB} ${LLD_COMMON_LIB} ${TROPICAL_LLVM_ALL_LIBS}
            -Wl,--end-group)
    endif()
    target_compile_definitions(tropical_core PRIVATE TROPICAL_WASM_EMIT)
endif()

if(TROPICAL_LLVM_STATIC)
    # Hide LLVM symbols so they don't leak out of libtropical and conflict
    # with any other LLVM loaded in the same process.
    if(APPLE)
        target_link_options(tropical_core PRIVATE
            -Wl,-dead_strip
            -Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/cmake/tropical_exports.txt)
    elseif(UNIX)
        target_link_options(tropical_core PRIVATE
            -Wl,--gc-sections
            -Wl,--exclude-libs,ALL
            -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/cmake/tropical.map)
    endif()
endif()

target_compile_features(tropical_core PRIVATE cxx_std_20)
target_link_libraries(tropical_core PRIVATE rtaudio nlohmann_json::nlohmann_json)
target_include_directories(tropical_core PRIVATE engine)
set_target_properties(tropical_core PROPERTIES OUTPUT_NAME "tropical")

# ---- Tests -------------------------------------------------------------------

option(TROPICAL_BUILD_TESTS "Build C++ process tests (no audio device required)" ON)

if(TROPICAL_BUILD_TESTS)
    enable_testing()
    add_executable(test_module_process engine/tests/test_module_process.cpp)
    target_include_directories(test_module_process PRIVATE engine)
    target_compile_features(test_module_process PRIVATE cxx_std_20)
    target_link_libraries(test_module_process PRIVATE tropical_core)
    add_test(NAME module_process COMMAND test_module_process)
endif()

# Install the public C header
install(FILES engine/c_api/tropical_c.h DESTINATION include)
if(TROPICAL_BUILD_PYTHON)
    # Place libtropical adjacent to the Python package so _bindings.py finds it
    # via its path-1 lookup (next to the tropical/ dir in site-packages).
    install(TARGETS tropical_core DESTINATION tropical)
else()
    install(TARGETS tropical_core DESTINATION lib)
endif()

