# Create a static library from KMeans.cpp 
# Means the Library is compiled and can be used by multiple executables whithout the need to be 
# recompiled for every executable
add_library(KMeansLib
            STATIC
            KMeans.cpp)

# Enable position-independent code for shared libraries (useful for shared libraries)
set_target_properties(KMeansLib 
                    PROPERTIES 
                    POSITION_INDEPENDENT_CODE ON)

add_library(Parallel_KMeansLib
            STATIC
            Parallel_KMeans.cpp)

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
