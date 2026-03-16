cmake_minimum_required(VERSION 3.20)
project(egress LANGUAGES C CXX)

option(EGRESS_BUILD_PYTHON "Build ctypes-based Python package support (installs egress/ package)" ON)
option(EGRESS_PROFILE "Enable profiling counters and timing instrumentation" OFF)
option(EGRESS_LLVM_ORC_JIT "Enable LLVM ORC JIT backend in C++ runtime" OFF)

# RtAudio library target
add_library(rtaudio STATIC lib/rtaudio/RtAudio.cpp)
target_include_directories(rtaudio SYSTEM PUBLIC lib/rtaudio)
target_compile_definitions(rtaudio PUBLIC __MACOSX_CORE__)
target_compile_options(rtaudio PRIVATE -Wno-vla-cxx-extension)
target_link_libraries(rtaudio PUBLIC
    "-framework CoreAudio"
    "-framework CoreFoundation"
    pthread)

# Core C++ sources shared by all targets
set(EGRESS_CORE_SOURCES
    src/graph/Graph.cpp
    src/graph/Module.cpp
    src/expr/ExprStructural.cpp
    src/expr/ExprRewrite.cpp
    src/c_api/egress_c.cpp)

# libegress — stable C shared library (no Python dependency)
add_library(egress_core SHARED ${EGRESS_CORE_SOURCES})

if(EGRESS_LLVM_ORC_JIT)
    find_package(LLVM CONFIG REQUIRED)
    if(LLVM_PACKAGE_VERSION VERSION_LESS 15)
        message(FATAL_ERROR "EGRESS_LLVM_ORC_JIT requires LLVM >= 15; found ${LLVM_PACKAGE_VERSION}")
    endif()
    message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION}")

    target_sources(egress_core PRIVATE src/jit/OrcJitEngine.cpp)
    target_include_directories(egress_core PRIVATE ${LLVM_INCLUDE_DIRS})
    target_compile_definitions(egress_core PRIVATE ${LLVM_DEFINITIONS} EGRESS_LLVM_ORC_JIT=1)

    llvm_map_components_to_libnames(LLVM_ORC_JIT_LIBS
        Core OrcJIT ExecutionEngine native nativecodegen Support)
    target_link_libraries(egress_core PRIVATE ${LLVM_ORC_JIT_LIBS})
endif()

target_compile_features(egress_core PRIVATE cxx_std_17)
if(EGRESS_PROFILE)
    target_compile_definitions(egress_core PRIVATE EGRESS_PROFILE=1)
endif()
target_link_libraries(egress_core PRIVATE rtaudio)
target_include_directories(egress_core PRIVATE src)
set_target_properties(egress_core PROPERTIES OUTPUT_NAME "egress")

# Install the public C header
install(FILES src/c_api/egress_c.h DESTINATION include)
install(TARGETS egress_core DESTINATION lib)

# Fade-logic unit test (no audio hardware required)
add_executable(test_fade test/test_fade.cpp)
target_compile_features(test_fade PRIVATE cxx_std_17)
target_link_libraries(test_fade PRIVATE egress_core)
target_include_directories(test_fade PRIVATE src)
