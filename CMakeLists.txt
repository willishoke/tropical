cmake_minimum_required(VERSION 3.20)
project(egress LANGUAGES C CXX)

option(EGRESS_BUILD_PYTHON "Build Python bindings with pybind11" ON)
option(EGRESS_PROFILE "Enable profiling counters and timing instrumentation" OFF)
option(EGRESS_LLVM_ORC_JIT "Enable LLVM ORC JIT backend in C++ runtime" OFF)

# RtAudio as a library target
add_library(rtaudio STATIC lib/rtaudio/RtAudio.cpp)
target_include_directories(rtaudio SYSTEM PUBLIC lib/rtaudio)
target_compile_definitions(rtaudio PUBLIC __MACOSX_CORE__)
target_compile_options(rtaudio PRIVATE -Wno-vla-cxx-extension)
target_link_libraries(rtaudio PUBLIC
    "-framework CoreAudio"
    "-framework CoreFoundation"
    pthread)

if(EGRESS_BUILD_PYTHON)
    set(PYBIND11_FINDPYTHON ON)
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)

    execute_process(
        COMMAND ${Python3_EXECUTABLE} -m pybind11 --cmakedir
        OUTPUT_VARIABLE PYBIND11_CMAKE_DIR
        RESULT_VARIABLE PYBIND11_CMAKE_DIR_RESULT
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)

    if(PYBIND11_CMAKE_DIR_RESULT EQUAL 0)
        set(pybind11_DIR "${PYBIND11_CMAKE_DIR}" CACHE PATH "pybind11 CMake config dir" FORCE)
        find_package(pybind11 CONFIG REQUIRED PATHS "${pybind11_DIR}" NO_DEFAULT_PATH)
    else()
        find_package(pybind11 CONFIG REQUIRED)
    endif()

    add_library(egress MODULE
        src/python/PythonBindings.cpp
        src/graph/Graph.cpp
        src/graph/Module.cpp
        src/expr/ExprStructural.cpp
        src/expr/ExprRewrite.cpp)
    if(EGRESS_LLVM_ORC_JIT)
        find_package(LLVM CONFIG REQUIRED)
        if(LLVM_PACKAGE_VERSION VERSION_LESS 15)
            message(FATAL_ERROR "EGRESS_LLVM_ORC_JIT requires LLVM >= 15; found ${LLVM_PACKAGE_VERSION}")
        endif()
        message(STATUS "Using LLVM ${LLVM_PACKAGE_VERSION}")

        target_sources(egress PRIVATE src/jit/OrcJitEngine.cpp)
        target_include_directories(egress PRIVATE ${LLVM_INCLUDE_DIRS})
        target_compile_definitions(egress PRIVATE ${LLVM_DEFINITIONS} EGRESS_LLVM_ORC_JIT=1)

        llvm_map_components_to_libnames(LLVM_ORC_JIT_LIBS
            Core
            OrcJIT
            ExecutionEngine
            native
            nativecodegen
            Support)
        target_link_libraries(egress PRIVATE ${LLVM_ORC_JIT_LIBS})
    endif()

    target_compile_features(egress PRIVATE cxx_std_17)
    if(EGRESS_PROFILE)
        target_compile_definitions(egress PRIVATE EGRESS_PROFILE=1)
    endif()
    target_link_libraries(egress PRIVATE pybind11::module rtaudio)
    target_include_directories(egress PRIVATE src)
    set_target_properties(egress PROPERTIES PREFIX "" OUTPUT_NAME "egress")
endif()
