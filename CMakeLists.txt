# Minimum CMake version required
cmake_minimum_required(VERSION 3.20)

# Project name and version
project(Custom_KMeans VERSION 1.0)

# Specify which compiler is used when none is set
if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER g++)
endif()

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(CMAKE_CXX_EXTENSIONS OFF)


if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

option(DISABLE_ARCH_OPT "Disable architecture-specific optimizations" OFF)

# only use the following compile option if Architecture Optimzation is on
if(DISABLE_ARCH_OPT)

    message(STATUS "Architecture optimizations DISABLED")

else()

    message(STATUS "Architecture optimizations ENABLED")

    if( CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM|AppleClang")
        message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
        set(CXX_FLAGS "${CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic")
        add_compile_options(${CXX_FLAGS})

        if ( CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM" ) 
            # adds optimization for Intel CPU (auto detect)
            message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
            set(CXX_FLAGS "${CXX_FLAGS} -xHost")
            add_compile_options(-xHost)

        else()
            # optimizes for the systems cpu used for GNU and Clang
            set(CXX_FLAGS "${CXX_FLAGS} -march=native -mtune=native")
            add_compile_options(-march=native -mtune=native)
        endif()
    # We do not support MS Visual Studio compiler at the moment
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "MSVC")
        message(FATAL_ERROR "MS Visual Studion compiler is not supported at the moment")
    endif()
endif()

if(CMAKE_BUILD_TYPE MATCHES Debug)
    message(STATUS "Build Type: Debug")
    set(CXX_FLAGS "${CXX_FLAGS} -g")
    add_compile_options(-g)

elseif(CMAKE_BUILD_TYPE MACTHES Release)
    message(STATUS "Build Type: Release")
    set(CXX_FLAGS "${CXX_FLAGS} -O3")
    add_compile_options(-O3)
endif()


# print out all the compile flags that were used
message(STATUS "CMAKE Compile Flags: ${CXX_FLAGS}")


# required packages 
find_package(OpenMP REQUIRED)

find_package(ZLIB REQUIRED)

# Include directories
include_directories(include)

# Add subdirectory for source files
add_subdirectory(src)




