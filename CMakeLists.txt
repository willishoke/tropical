cmake_minimum_required(VERSION 3.20)
project(egress LANGUAGES CXX)

# Your executable
add_executable(egress Main.cpp)
target_compile_features(egress PRIVATE cxx_std_17)
target_compile_options(egress PRIVATE -Wall -Wextra)

# RtAudio as a library target
add_library(rtaudio STATIC lib/rtaudio/RtAudio.cpp)
target_include_directories(rtaudio SYSTEM PUBLIC lib/rtaudio)
target_compile_definitions(rtaudio PUBLIC __MACOSX_CORE__)
target_compile_options(rtaudio PRIVATE -Wno-vla-cxx-extension)
target_link_libraries(rtaudio PUBLIC
    "-framework CoreAudio"
    "-framework CoreFoundation"
    pthread)

# Link rtaudio into your app — PUBLIC properties propagate automatically
target_link_libraries(egress PRIVATE rtaudio ncurses)
