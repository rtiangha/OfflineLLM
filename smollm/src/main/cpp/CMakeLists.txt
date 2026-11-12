cmake_minimum_required(VERSION 3.22.1)
project("smollm")

# ggml-vulkan.cpp includes <spirv/unified1/spirv.hpp> from KhronosGroup/SPIRV-Headers,
# which the NDK sysroot doesn't carry. The include dir arrives as a cache var rather
# than through cppFlags: cppFlags lands in CMAKE_CXX_FLAGS, which is whitespace-split
# (so it breaks on any checkout path containing a space) and is applied globally to
# every C++ target rather than just the one that needs it.
# Declared before add_subdirectory so it propagates to the ggml-vulkan target.
if(DEFINED SPIRV_HEADERS_INCLUDE_DIR AND EXISTS "${SPIRV_HEADERS_INCLUDE_DIR}")
    include_directories(SYSTEM "${SPIRV_HEADERS_INCLUDE_DIR}")
endif()

# llama.cpp must be cloned at the project root as a submodule
# git submodule add https://github.com/ggerganov/llama.cpp.git
# libadrenotools (git submodule at project root, with its own linkernsbypass
# submodule): lets the Vulkan backend load a custom driver such as Mesa Turnip.
# Built PIC because it is linked into the ggml-vulkan plugin (a shared library).
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(ADRENOTOOLS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../../../../libadrenotools")
if(NOT EXISTS "${ADRENOTOOLS_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "libadrenotools submodule missing - run: git submodule update --init --recursive")
endif()
# libadrenotools' `add_library(adrenotools ...)` follows BUILD_SHARED_LIBS, which the
# Gradle config sets to ON for llama.cpp. Force a static adrenotools (linked into the
# ggml-vulkan plugin); its hook libs (libhook_impl.so, libmain_hook.so, ...) are
# explicitly SHARED targets and still land in the APK's native lib dir, which is where
# adrenotools_open_libvulkan() expects them. unset() restores the -D cache value.
set(BUILD_SHARED_LIBS OFF)
add_subdirectory(${ADRENOTOOLS_DIR} libadrenotools)
unset(BUILD_SHARED_LIBS)

add_subdirectory(../../../../llama.cpp llama.cpp)

# ggml-vulkan.cpp is patched by scripts/patch_ggml_vulkan.py to call
# adrenotools_open_libvulkan() when a custom driver is configured.
if(NOT TARGET ggml-vulkan)
    message(FATAL_ERROR "ggml-vulkan target not found (is GGML_VULKAN=ON?)")
endif()
target_link_libraries(ggml-vulkan PRIVATE adrenotools log)
target_include_directories(ggml-vulkan PRIVATE "${ADRENOTOOLS_DIR}/include")
set(LLAMA_DIR_RELATIVE "../../../../llama.cpp")
get_filename_component(LLAMA_DIR ${LLAMA_DIR_RELATIVE} ABSOLUTE)

set(GGML_DIR ${LLAMA_DIR}/ggml)

target_compile_options(
        llama
        PUBLIC
        -fvisibility=hidden -fvisibility-inlines-hidden
)
target_compile_options(
        llama
        PUBLIC
        -ffunction-sections -fdata-sections
)
target_link_options(
        llama
        PRIVATE
        -Wl,--gc-sections -flto
        -Wl,--exclude-libs,ALL
)

# Single JNI wrapper library. CPU-specific optimisation lives in the ggml-cpu
# plugin variants (GGML_CPU_ALL_VARIANTS) selected at runtime, not here — the
# wrapper is thin glue and arch flags on it never affected inference speed.
add_library(
        smollm
        SHARED
        LLMInference.cpp
        smollm.cpp
)
target_include_directories(
        smollm
        PUBLIC
        ${GGML_DIR}/include
        ${GGML_DIR}/src
        ${LLAMA_DIR}/include
)
target_compile_options(
        smollm
        PUBLIC
        -O3
        -fvisibility=hidden -fvisibility-inlines-hidden
        -ffunction-sections -fdata-sections
)
# Note: deliberately NOT linking llama-common — it contains upstream's
# HTTP/download machinery, which has no place in a zero-network app. The few
# helpers the wrapper needed are reimplemented locally on the llama.h C API.
target_link_libraries(
        smollm
        android log llama vulkan
)
target_link_options(
        smollm
        PRIVATE
        -Wl,--gc-sections -flto
        -Wl,--exclude-libs,ALL
)

# GGUFReader library
set(TARGET_NAME_GGUF_READER ggufreader)
add_library(${TARGET_NAME_GGUF_READER} SHARED GGUFReader.cpp)
target_include_directories(
        ${TARGET_NAME_GGUF_READER}
        PUBLIC
        ${GGML_DIR}/include
)
target_compile_options(
        ${TARGET_NAME_GGUF_READER}
        PUBLIC
        -fvisibility=hidden -fvisibility-inlines-hidden -ffunction-sections -fdata-sections
)
target_link_libraries(
        ${TARGET_NAME_GGUF_READER}
        ggml
)
target_link_options(
        ${TARGET_NAME_GGUF_READER}
        PRIVATE
        -Wl,--gc-sections -flto
        -Wl,--exclude-libs,ALL
)
