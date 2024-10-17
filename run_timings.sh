#!/bin/bash

# preprocessor macros
USE_CONT_MEM="USE_CONT_MEM"

timing_iterations=3

exectuable="src/KMeans"
data="/home/joshua/Projects/HPC_Project/data/openml/openml.org/data/v1/download/52667.gz"
output_dir="./src/out"
source_dir="./src"


cxx_compiler="g++"
cxx_standard="-std=c++20"
cxx_compile_flags="-O3"
arch_opt="ON"
#compile_definitions="-DCOMPILER=\"${cxx_compiler}\" -DOPTIMIZATION=3 -DARCH_OPT=\"no_archopt\""
link_libs="-lz -fopenmp"


include_directory="./include"
source_files=$(ls ${source_dir}/*.cpp)

if [ ${arch_opt} == "OFF" ]; then
    arch_opt="no_archopt"
    echo "arch opt OFF"
else
    arch_opt="arch_opt"
    echo "arch opt ON"
    cxx_compile_flags="${cxx_compile_flags} -march=native -mtune=native"
fi
    

output_file=${output_dir}/${cxx_compiler}_$(echo "${cxx_compile_flags}" | sed -n 's/.*-\(O[0-3]\).*/\1/p')_${arch_opt}_timings.txt
echo "${output_file}"
echo "Creating output directory: ${output_dir}"
mkdir -p "${output_dir}"

if [ ! -f ${output_file} ]; then
    touch ${output_file}
    echo "Created Output File: ${output_file}"
else
    echo "${output_file} already exists"
fi


echo "Compiling ${executable}"
echo "Compile flags: ${cxx_compile_flags}"

${cxx_compiler} ${cxx_compile_flags} -g ${cxx_standard} -I${include_directory} ${source_files} -D${USE_CONT_MEM} -o ${exectuable} ${link_libs}

echo "Compilation done"

# number of CPU cores
num_cores=$(nproc)

echo "Starting to time Parallel KMeans Algorithm"

for threads in $(seq 1 ${num_cores}); do

    export OMP_NUM_THREADS=${threads}
    echo "Running Timing with ${threads} Threads"
    ${exectuable} --data ${data} --output ${output_file} --timing_iterations ${timing_iterations}


done

echo "Finished Timing"