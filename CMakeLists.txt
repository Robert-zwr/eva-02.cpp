cmake_minimum_required(VERSION 3.12) # Don't bump this version for no reason
project("eva.cpp" C CXX)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if (NOT XCODE AND NOT MSVC AND NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    set(EVA_STANDALONE ON)

    # configure project version
    # TODO
else()
    set(EVA_STANDALONE OFF)
endif()

if (EMSCRIPTEN)
    set(BUILD_SHARED_LIBS_DEFAULT OFF)

    option(EVA_WASM_SINGLE_FILE "eva: embed WASM inside the generated eva.js" ON)
else()
    if (MINGW)
        set(BUILD_SHARED_LIBS_DEFAULT OFF)
    else()
        set(BUILD_SHARED_LIBS_DEFAULT ON)
    endif()
endif()


#
# Option list
#

# general
option(EVA_STATIC                 "eva: static link libraries"                          OFF)
option(EVA_NATIVE                 "eva: enable -march=native flag"                      OFF)
option(EVA_LTO                    "eva: enable link time optimization"                  OFF)

# debug
option(EVA_ALL_WARNINGS           "eva: enable all compiler warnings"                   ON)
option(EVA_ALL_WARNINGS_3RD_PARTY "eva: enable all compiler warnings in 3rd party libs" OFF)
option(EVA_GPROF                  "eva: enable gprof"                                   OFF)

# sanitizers
option(EVA_SANITIZE_THREAD        "eva: enable thread sanitizer"                        OFF)
option(EVA_SANITIZE_ADDRESS       "eva: enable address sanitizer"                       OFF)
option(EVA_SANITIZE_UNDEFINED     "eva: enable undefined sanitizer"                     OFF)

# instruction set specific
option(EVA_AVX                    "eva: enable AVX"                                     ON)
option(EVA_AVX2                   "eva: enable AVX2"                                    ON)
option(EVA_FMA                    "eva: enable FMA"                                     ON)

# 3rd party libs
option(EVA_ACCELERATE             "eva: enable Accelerate framework"                    ON)
option(EVA_OPENBLAS               "eva: use OpenBLAS"                                   OFF)

option(EVA_BUILD_TESTS            "eva: build tests"    ${EVA_STANDALONE})
option(EVA_BUILD_EXAMPLES         "eva: build examples" ${EVA_STANDALONE})

#
# Compile flags
#

set(CMAKE_CXX_STANDARD_REQUIRED true)
set(CMAKE_C_STANDARD_REQUIRED true)
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

if (NOT MSVC)
    if (EVA_SANITIZE_THREAD)
        add_compile_options(-fsanitize=thread)
    endif()

    if (EVA_SANITIZE_ADDRESS)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
    endif()

    if (EVA_SANITIZE_UNDEFINED)
        add_compile_options(-fsanitize=undefined)
    endif()
endif()

if (APPLE AND EVA_ACCELERATE)
    find_library(ACCELERATE_FRAMEWORK Accelerate)
    if (ACCELERATE_FRAMEWORK)
        message(STATUS "Accelerate framework found")

        add_compile_definitions(GGML_USE_ACCELERATE)
        set(EVA_EXTRA_LIBS ${EVA_EXTRA_LIBS} ${ACCELERATE_FRAMEWORK})
    else()
        message(WARNING "Accelerate framework not found")
    endif()
endif()
if (EVA_OPENBLAS)
    if (EVA_STATIC)
        set(BLA_STATIC ON)
    endif()

    set(BLA_VENDOR OpenBLAS)
    find_package(BLAS)
    if (BLAS_FOUND)
        message(STATUS "OpenBLAS found")

        add_compile_definitions(GGML_USE_OPENBLAS)
        add_link_options(${BLAS_LIBRARIES})
    else()
        message(WARNING "OpenBLAS not found")
    endif()
endif()

if (EVA_ALL_WARNINGS)
    if (NOT MSVC)
        set(c_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wshadow
            -Wcast-qual
            -Wstrict-prototypes
            -Wpointer-arith
            -Wno-unused-function
        )
        set(cxx_flags
            -Wall
            -Wextra
            -Wpedantic
            -Wcast-qual
        )
    else()
        # todo : msvc
    endif()

    add_compile_options(
            "$<$<COMPILE_LANGUAGE:C>:${c_flags}>"
            "$<$<COMPILE_LANGUAGE:CXX>:${cxx_flags}>"
    )

endif()

if (EVA_LTO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT result OUTPUT output)
    if (result)
        set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(WARNING "IPO is not supported: ${output}")
    endif()
endif()

# Architecture specific
# TODO: probably these flags need to be tweaked on some architectures
#       feel free to update the Makefile for your architecture and send a pull request or issue
message(STATUS "CMAKE_SYSTEM_PROCESSOR: ${CMAKE_SYSTEM_PROCESSOR}")
if (NOT MSVC)
    if (EVA_STATIC)
        add_link_options(-static)
        if (MINGW)
            add_link_options(-static-libgcc -static-libstdc++)
        endif()
    endif()
    if (EVA_GPROF)
        add_compile_options(-pg)
    endif()
    if (EVA_NATIVE)
        add_compile_options(-march=native)
    endif()
endif()

if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "arm" OR ${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
    message(STATUS "ARM detected")
    if (MSVC)
        # TODO: arm msvc?
    else()
        if (${CMAKE_SYSTEM_PROCESSOR} MATCHES "aarch64")
            add_compile_options(-mcpu=native)
        endif()
        # TODO: armv6,7,8 version specific flags
    endif()
elseif (${CMAKE_SYSTEM_PROCESSOR} MATCHES "^(x86_64|i686|AMD64)$")
    message(STATUS "x86 detected")
    if (MSVC)
        if (EVA_AVX2)
            add_compile_options(/arch:AVX2)
        elseif (EVA_AVX)
            add_compile_options(/arch:AVX)
        endif()
    else()
        add_compile_options(-mf16c)
        if (EVA_FMA)
            add_compile_options(-mfma)
        endif()
        if (EVA_AVX)
            add_compile_options(-mavx)
        endif()
        if (EVA_AVX2)
            add_compile_options(-mavx2)
        endif()
    endif()
else()
    # TODO: support PowerPC
    message(STATUS "Unknown architecture")
endif()

#
# Build libraries
#

add_library(utils OBJECT
            utils.cpp
            utils.h)

target_include_directories(utils PUBLIC .)
target_compile_features(utils PUBLIC cxx_std_11) # don't bump
target_link_libraries(utils PRIVATE ${EVA_EXTRA_LIBS})
if (BUILD_SHARED_LIBS)
    set_target_properties(utils PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(ggml OBJECT
            ggml.c
            ggml.h)

target_include_directories(ggml PUBLIC .)
target_compile_features(ggml PUBLIC c_std_11) # don't bump
target_link_libraries(ggml PRIVATE Threads::Threads ${EVA_EXTRA_LIBS})
if (BUILD_SHARED_LIBS)
    set_target_properties(ggml PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

add_library(eva
            eva.cpp
            eva.h)

target_include_directories(eva PUBLIC .)
target_compile_features(eva PUBLIC cxx_std_11) # don't bump
target_link_libraries(eva PRIVATE utils ggml ${EVA_EXTRA_LIBS})
if (BUILD_SHARED_LIBS)
    set_target_properties(eva PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_compile_definitions(eva PRIVATE EVA_SHARED EVA_BUILD)
endif()

#
# Executables
#

add_executable(main main.cpp)
target_link_libraries(main PRIVATE eva ggml utils)

#
# programs, examples and tests
#

if (EVA_BUILD_TESTS AND NOT CMAKE_JS_VERSION)
    enable_testing()
    # add_subdirectory(tests)
endif ()
