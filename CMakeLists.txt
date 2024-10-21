# Minimum CMake version required
cmake_minimum_required(VERSION 3.20)

# Project name and version
project(Custom_KMeans VERSION 1.0)

# Specify which compiler is used when none is set
if(NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER g++)
endif()

# Specify which build to use Debug or Release
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Specify the C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# Let Cmake create a .json file with all the compile flags
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_EXTENSIONS OFF)

# Reset the the CMAKE_CXX_FLAGS_RELEASE because i later set the flags based on debug
# or reease build anyways to avoid adding the flags double
set(CMAKE_CXX_FLAGS_RELEASE "-DNDEBUG")

if(NOT USE_CONT_MEM)
    message(STATUS "USE_CONT_MEM not defined setting it to OFF")
    option(USE_CONT_MEM OFF)
else()
    message(STATUS "Adding USE_MEM_CONT Macro to globally")
    option(USE_CONT_MEM ON)
    add_definitions(-DUSE_MEM_CONT)
endif()

if(SIMD_256)
    message(STATUS "Using AVX-256")
    add_definitions(-DSIMD_256)
elseif(SIMD_512)
    message(STATUS "Using AVX-512")
    add_definitions(-DSIMD_512)
else()
    message(STATUS "No specific SIMD directive set using default compiler optimizations")
    add_definitions(-DNO_SIMD)

endif()


message(STATUS "CXX Release flags: ${CMAKE_CXX_FLAGS_RELEASE}")

option(DISABLE_ARCH_OPT "Disable architecture-specific optimizations" OFF)

# only use the following compile option if Architecture Optimzation is on
if(DISABLE_ARCH_OPT)
    
    message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
    message(STATUS "Architecture optimizations DISABLED")
    set(CXX_FLAGS "${CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic")
    add_compile_options(-Wall -Wextra -Wshadow -pedantic)		

else()

    message(STATUS "Architecture optimizations ENABLED")

    if( CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|IntelLLVM|AppleClang")
        message(STATUS "C++ compiler: ${CMAKE_CXX_COMPILER_ID}")
        set(CXX_FLAGS "${CXX_FLAGS} -Wall -Wextra -Wshadow -pedantic")
        add_compile_options(-Wall -Wextra -Wshadow -pedantic)

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
    if(NOT COMPILER_OPTIMIZATION)
	message(STATUS "Set Compiler Optimization to -O0 for Debugging")
	set(CXX_FLAGS "${CXX_FLAGS} -O0")
	add_compile_options(-O0)
    else()
	message(STATUS "Tried adding Compiler Optimizations: ${COMPILER_OPTIMIZATION} but CMAKE_BUILD_TYPE=Debug")
	message(STATUS "Ignored all Optimizations and added -O0")
	set(CXX_FLAGS "${CXX_FLAGS} -O0")
	add_compile_options(-O0)
    endif()
    

elseif(CMAKE_BUILD_TYPE MATCHES Release)
    message(STATUS "Build Type: Release")
    
    # If no optimization flag (-O0, -O1, -O2) is found, add -O3
    if(COMPILER_OPTIMIZATION MATCHES "-O3")
    
	message(STATUS "Setting Optimization Flags: ${COMPILER_OPTIMIZATION}")
        set(CXX_FLAGS "${CXX_FLAGS} ${COMPILER_OPTIMIZATION}")
        add_compile_options(${COMPILER_OPTIMIZATION})
        
    else()
    
	if (NOT COMPILER_OPTIMIZATION)
	
	    message(STATUS "No OPTIMIZATIONS FLAG set")
	    message(STATUS "Added -O3 to compile flags")
	    set(CXX_FLAGS "{CXX_FLAGS} -O3")
	    add_compile_options(-O3)
	    
	else()
	
	    message(STATUS "Release Build optimization not set to -O3 but ${COMPILER_OPTIMIZATION}")
	    set(CXX_FLAGS "${CXX_FLAGS} ${COMPILER_OPTIMIZATION}")
    	    add_compile_options(${COMPILER_OPTIMIZATION})
    	endif()
    endif()
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




