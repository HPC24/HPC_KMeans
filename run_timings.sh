#!/bin/bash

exectuable="src/KMeans"
data="/home/joshua/Projects/HPC_Project/data/openml/openml.org/data/v1/download/52667.gz"
output_dir="./src/out"
source_dir="./src"


cxx_compiler="g++"
cxx_standard="-std=c++20"
cxx_compile_flags="-O3"
arch_opt="OFF"
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
fi
    

output_file=${output_dir}/${cxx_compiler}_${cxx_compile_flags/-/}_${arch_opt}_timings.txt

echo "Creating output directory: ${output_dir}"
mkdir -p "${output_dir}"

if [ ! -f ${output_file} ]; then
    touch ${output_file}
    echo "Created Output File: ${output_file}"
else
    echo "${output_file} already exists"
fi


echo "Compiling ${executable}"

${cxx_compiler} ${cxx_compile_flags} ${cxx_standard} -I${include_directory} ${source_files} -o ${exectuable} ${link_libs}

echo "Compilation done"

# number of CPU cores
num_cores=$(nproc)

echo "Starting to time Parallel KMeans Algorithm"

for threads in $(seq 1 ${num_cores}); do

    export OMP_NUM_THREADS=${threads}
    echo "Running Timing with ${threads} Threads"
    ${exectuable} --data ${data} --output ${output_file}


done

echo "Finished Timing"