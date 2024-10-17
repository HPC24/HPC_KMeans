# Create a static library from KMeans.cpp 
# Means the Library is compiled and can be used by multiple executables whithout the need to be 
# recompiled for every executable
if (USE_CONT_MEM)
    set(PARALLEL_KMEANS_SRC Cont_Mem_Parallel_KMeans.cpp)
else()
    set(PARALLEL_KMEANS_SRC Parallel_KMeans.cpp)
endif()

add_library(KMeansLib
            STATIC
            KMeans.cpp)

# Enable position-independent code for shared libraries (useful for shared libraries)
set_target_properties(KMeansLib 
                    PROPERTIES 
                    POSITION_INDEPENDENT_CODE ON)
                    

add_library(Parallel_KMeansLib
            STATIC
            ${PARALLEL_KMEANS_SRC})

set_target_properties(Parallel_KMeansLib 
                    PROPERTIES 
                    POSITION_INDEPENDENT_CODE ON)

target_link_libraries(Parallel_KMeansLib
                    PRIVATE 
                    OpenMP::OpenMP_CXX)


# KMeans executable
add_executable(KMeans 
            main.cpp)

# link to the KMeans Library and the lz library to read from a gzipped file
target_link_libraries(KMeans 
                    PRIVATE 
                    KMeansLib
                    Parallel_KMeansLib
                    OpenMP::OpenMP_CXX
                    ZLIB::ZLIB) 
                    

target_compile_definitions(KMeans
                        PRIVATE
                        COMPILER=\"${CMAKE_CXX_COMPILER_ID}\")
                


# Tests executable, linking to KMeansLib for access to KMeans functions
add_library(Tests 
            STATIC
            Tests.cpp)

target_link_libraries(Tests
                    PRIVATE 
                    Parallel_KMeansLib
                    KMeansLib
                    OpenMP::OpenMP_CXX
                    ZLIB::ZLIB)

# If the intel compiler is used the standard library needs to be linke to everything
if (CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
    target_link_libraries(KMeansLib PRIVATE stdc++)
    target_link_libraries(Parallel_KMeansLib PRIVATE stdc++)
    target_link_libraries(KMeans PRIVATE stdc++)
    target_link_libraries(Tests PRIVATE stdc++)
endif()