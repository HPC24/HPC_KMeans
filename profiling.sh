#!/usr/bin/env sh

#SBATCH --account=kurs_2024_sose_hpc
#SBATCH --reservation=hpc_course_sose2024
 
#SBATCH --job-name=profiling
#SBATCH --time=0-04:00:00
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=7900MB

#SBATCH --output=%u.log.%j.out
#SBATCH --error=%u.log.%j.err

# CMake output_Paths
BUILD_DIR="./build"

CMAKE_BUILD_TYPE="Release"

# Threads used for the Profiling
PROFILING_THREADS=24

# Number of iterations used for timing the KMeans implementation
TIMING_ITERATIONS=1

# if USE_CONT_MEM is set the Cont_Mem_Parallel_KMeans.cpp implementation will be used 
USE_CONT_MEM="USE_CONT_MEM"

if [ ${TIMING_ITERATIONS} -eq 1 ]; then
    echo "As Timing Iterations is set to ${TIMING_ITERATIONS} running VTune"
    echo "To Time the KMeans Implementation for Different Number of OMP_NUM_THREADS set TIMING_ITERATION > 1"
elif [ ${TIMING_ITERATIONS} -gt 1 ]; then
    echo "As Timing Iterations is set to ${TIMING_ITERATIONS} running Performance for different Number of OMP_NUM_THREADS"
    echo "To analyse the performance of the KMeans Implementation with VTune set TIMING_ITERATIONS == 1"
else 
    echo "TIMING_ITERATIONS set to ${TIMING_ITERATIONS} which is an invalid value needs to be >= 1"
    echo "ERROR: set wrong value for TIMING_ITERATIONS"
    exit 1
fi


# VTune Parameters
PROFILING_RESULTS_DIR="vtune_results"
ANALYSIS_TYPE="hotspots"

# Paths
EXECUTABLE="src/KMeans"
DATA="/scratch/kurs_2024_sose_hpc/kurs_2024_sose_hpc_11/data/openml/openml.org/data/v1/download/52667.gz"
#OUTPUT_DIR="./src/out"
#SOURCE_DIR="./src"

# Compiler Flags
CXX_COMPILER="g++"
CXX_STANDARD="-std=c++20"
CXX_COMPILER_FLAGS="-O3"
DISABLE_ARCH_OPT="OFF"
#compile_definitions="-DCOMPILER=\"${cxx_compiler}\" -DOPTIMIZATION=3 -DARCH_OPT=\"no_archopt\""
#LINK_LIBS="-lz -fopenmp"

#INCLUDE_DIRECTORY="./include"
#SOURCE_FILES=$(ls ${SOURCE_DIR}/*.cpp)

if [ ${DISABLE_ARCH_OPT} == "ON" ]; then
    ARCH_OPT="no_archopt"
    echo "arch opt OFF"
else
    ARCH_OPT="arch_opt"
    echo "arch opt ON"
    echo "adding compiler Flags -march=native -mtune=native"
    #CXX_COMPILER_FLAGS="$CXX_COMPILER_FLAGS -march=native -mtune=native"
fi

COMPILER_OPTIMIZATION=$(echo ${CXX_COMPILER_FLAGS} | grep -o '\-O[^-]*')
echo "Optimization Flag: ${COMPILER_OPTIMIZATION}"

# Paths of the Output files, Build directory, output directory (timings) and the VTune results directory
BUILD_DIR=${BUILD_DIR}_${CXX_COMPILER}_${COMPILER_OPTIMIZATION/-/}_${ARCH_OPT}
OUTPUT_DIR=${BUILD_DIR}/out
OUTPUT_FILE=${OUTPUT_DIR}/${CXX_COMPILER}_${COMPILER_OPTIMIZATION/-/}_${ARCH_OPT}_timings_vtune.txt
VTUNE_OUTPUT_DIRECTORY=${BUILD_DIR}/${PROFILING_RESULTS_DIR}_${CXX_COMPILER}_${COMPILER_OPTIMIZATION/-/}_OMP_${PROFILING_THREADS}_${ARCH_OPT}


echo "Creating Build directory: ${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

echo "Creating Output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [ ! -f ${OUTPUT_FILE} ]; then
    touch  ${OUTPUT_FILE}
    echo "Created Output File: ${OUTPUT_FILE}"
else
    echo "${OUTPUT_FILE} already exists"

fi

# Start with slurm specific commands
module purge
module add slurm
module add zlib/1.3.1
module add cmake/3.27.9
module add intel-oneapi-vtune/2024.1.0

if [ $CXX_COMPILER == "g++" ]; then
    echo "Loading gcc compiler"
    module add gcc/13.2.0
elif [ $CXX_COMPILER == "icx" ]; then
    echo "Loading intel compiler"
    module add intel-oneapi-compilers/2024.1.0
    echo "loading intel toolchains"
    module add toolchains/intel
fi

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Maximal threads per process: $SLURM_CPUS_PER_TASK"
echo "Current working directory is `pwd`" 

# echo "Starting Compilation"
# echo "Compiler: ${CXX_COMPILER}"
# echo "CXX_Standard: ${CXX_STANDARD}"
# echo "CXX Flags: ${CXX_COMPILER_FLAGS} ${LINK_LIBS}"

echo "Checking if executable KMeans already exists in ${BUILD_DIR}"
CHECK_EXECUTABLE=$(find ${BUILD_DIR} -wholename "*${EXECUTABLE}")

if [ ! -z ${CHECK_EXECUTABLE} ]; then
    
    echo "Executable found in ${CHECK_EXECUTABLE}"
    echo "CMake was not executed to create a new build remove the build directory: ${BUILD_DIR}"
    
else
        
    echo "No executable found in ${BUILD_DIR}"
    echo "Starting CMake Build Process"

    cmake -S . -B ${BUILD_DIR} \
	-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE} \
	-DCMAKE_CXX_COMPILER=${CXX_COMPILER} \
	-DCOMPILER_OPTIMIZATION=${COMPILER_OPTIMIZATION} \
	-DDISABLE_ARCH_OPT=${DISABLE_ARCH_OPT}

    # ${CXX_COMPILER} ${CXX_STANDARD} ${CXX_COMPILER_FLAGS} -I ${INCLUDE_DIRECTORY} ${SOURCE_FILES} -o ${EXECUTABLE} ${LINK_LIBS}
    echo "Creating executable ${EXECUTABLE} in ${BUILD_DIR}"

    cmake --build ${BUILD_DIR}

    echo "CMake finished"
fi

if [ ${TIMING_ITERATIONS} -eq 1 ]; then

    echo "Creating Vtune result directory ${VTUNE_OUTPUT_DIRECTORY}"
    mkdir -p "${VTUNE_OUTPUT_DIRECTORY}"

    echo "Starting Vtune"
    echo "Collecting: ${ANALYSIS_TYPE}"
    echo "OMP_NUM_THREADS: ${PROFILING_THREADS}"

    export OMP_NUM_THREADS=${PROFILING_THREADS}

    vtune -collect ${ANALYSIS_TYPE} \
        -result-dir ${VTUNE_OUTPUT_DIRECTORY} \
        -- ${BUILD_DIR}/${EXECUTABLE} \
        --data ${DATA} \
        --output ${OUTPUT_FILE} \
        --timing_iterations ${TIMING_ITERATIONS}

    echo "finished profiling"

else
    
    OUTPUT_FILE=${OUTPUT_FILE/_vtune/}
    
    if [ ! -f ${OUTPUT_FILE} ]; then
	touch ${OUTPUT_FILE}
	echo "Created Output File: ${OUTPUT_FILE}"
    else
	echo "Output file already exists: ${OUTPUT_FILE}"
    fi
    
    echo "Starting timing for up to ${SLURM_CPUS_PER_TASK}"
    echo "Timing iterations: ${TIMING_ITERATIONS}"

    for N in $(seq 1 ${SLURM_CPUS_PER_TASK}); 
    do
        export OMP_NUM_THREADS=${N}

        echo "Running Timing with ${N} Threads"

        ${BUILD_DIR}/${EXECUTABLE} \
        --data ${DATA} \
        --output ${OUTPUT_FILE} \
        --timing_iterations ${TIMING_ITERATIONS}
    done

    echo "Finished timing"

fi














