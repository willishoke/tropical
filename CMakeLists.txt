cmake_minimum_required(VERSION 3.20)
project(egress LANGUAGES C CXX)

# Default to RelWithDebInfo (-O2 -g -DNDEBUG): optimized but with debug symbols
# for profiling tools (Instruments, perf, etc.).  Override with e.g.
# -DCMAKE_BUILD_TYPE=Release for maximum throughput or =Debug for debugging.
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

option(EGRESS_BUILD_PYTHON "Build ctypes-based Python package support (installs egress/ package)" ON)
# Profiling overhead is minimal (atomic increments + rdtsc, gated by #ifdef).
# ON by default — pass -DEGRESS_PROFILE=OFF to disable.
option(EGRESS_PROFILE "Enable profiling counters and timing instrumentation" ON)
option(EGRESS_LLVM_STATIC "Link LLVM statically into libegress for self-contained wheel distribution" OFF)

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
set(EGRESS_CORE_SOURCES
    engine/expr/ExprStructural.cpp
    engine/expr/ExprRewrite.cpp
    engine/runtime/FlatRuntime.cpp
    engine/c_api/egress_c.cpp)

# libegress — stable C shared library (no Python dependency)
add_library(egress_core SHARED ${EGRESS_CORE_SOURCES})

find_package(LLVM CONFIG REQUIRED)
if(LLVM_PACKAGE_VERSION VERSION_LESS 15)
    message(FATAL_ERROR "LLVM >= 15 required; found ${LLVM_PACKAGE_VERSION}")
endif()
message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION}")

target_sources(egress_core PRIVATE engine/jit/OrcJitEngine.cpp)
target_include_directories(egress_core PRIVATE ${LLVM_INCLUDE_DIRS})
target_compile_definitions(egress_core PRIVATE ${LLVM_DEFINITIONS})

llvm_map_components_to_libnames(LLVM_ORC_JIT_LIBS
    Core OrcJIT ExecutionEngine native nativecodegen Support)
target_link_libraries(egress_core PRIVATE ${LLVM_ORC_JIT_LIBS})

if(EGRESS_LLVM_STATIC)
    # Hide LLVM symbols so they don't leak out of libegress and conflict
    # with any other LLVM loaded in the same process.
    if(APPLE)
        target_link_options(egress_core PRIVATE
            -Wl,-dead_strip
            -Wl,-exported_symbols_list,${CMAKE_CURRENT_SOURCE_DIR}/cmake/egress_exports.txt)
    elseif(UNIX)
        target_link_options(egress_core PRIVATE
            -Wl,--gc-sections
            -Wl,--exclude-libs,ALL
            -Wl,--version-script,${CMAKE_CURRENT_SOURCE_DIR}/cmake/egress.map)
    endif()
endif()

target_compile_features(egress_core PRIVATE cxx_std_20)
if(EGRESS_PROFILE)
    target_compile_definitions(egress_core PRIVATE EGRESS_PROFILE=1)
endif()
target_link_libraries(egress_core PRIVATE rtaudio nlohmann_json::nlohmann_json)
target_include_directories(egress_core PRIVATE engine)
set_target_properties(egress_core PROPERTIES OUTPUT_NAME "egress")

# ---- Tests -------------------------------------------------------------------

option(EGRESS_BUILD_TESTS "Build C++ process tests (no audio device required)" ON)

if(EGRESS_BUILD_TESTS)
    enable_testing()
    add_executable(test_module_process engine/tests/test_module_process.cpp)
    target_include_directories(test_module_process PRIVATE engine)
    target_compile_features(test_module_process PRIVATE cxx_std_20)
    target_link_libraries(test_module_process PRIVATE egress_core)
    add_test(NAME module_process COMMAND test_module_process)
endif()

# Install the public C header
install(FILES engine/c_api/egress_c.h DESTINATION include)
if(EGRESS_BUILD_PYTHON)
    # Place libegress adjacent to the Python package so _bindings.py finds it
    # via its path-1 lookup (next to the egress/ dir in site-packages).
    install(TARGETS egress_core DESTINATION egress)
else()
    install(TARGETS egress_core DESTINATION lib)
endif()

