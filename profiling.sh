#!/usr/bin/env sh

#SBATCH --account=kurs_2024_sose_hpc
#SBATCH --reservation=hpc_course_sose2024
 
#SBATCH --job-name=
#SBATCH --time=0-00:05:00
#SBATCH --partition=single
#SBATCH --nnodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=7900MB

#SBATCH --array=1-10
#SBATCH --output=%u.log.%A_%a.out
#SBATCH --error=%u.log.%A_%a.err

# Threads used for the Profiling
PROFILING_THREADS=1

# VTune Parameters
PROFILING_RESULTS_DIR="./vtune_results"
ANALYSIS_TYPE="hotspots"

# Paths
EXECUTABLE="src/KMeans"
DATA="/home/joshua/Projects/HPC_Project/data/openml/openml.org/data/v1/download/52667.gz"
OUTPUT_DIR="./src/out"
SOURCE_DIR="./src"

# Compiler Flags
CXX_COMPILER="g++"
CXX_STANDARD="-std=c++20"
CXX_COMPILER_FLAGS="-O3"
ARCH_OPT="OFF"
#compile_definitions="-DCOMPILER=\"${cxx_compiler}\" -DOPTIMIZATION=3 -DARCH_OPT=\"no_archopt\""
LINK_LIBS="-lz -fopenmp"

INCLUDE_DIRECTORY="./include"
SOURCE_FILES=$(ls ${source_dir}/*.cpp)

if [ ${ARCH_OPT} == "OFF" ]; then
    ARCH_OPT="no_archopt"
    echo "arch opt OFF"
else
    ARCH_OPT="arch_opt"
    echo "arch opt ON"
    echo "adding compiler Flags -march=native -mtune=native"
    CXX_COMPILER_FLAGS="$CXX_COMPILER_FLAGS -march=native -mtune=native"


fi

OUTPUT_FILE=${OUTPUT_DIR}/${CXX_COMPILER}_${cxx_compile_flags/-/}_${ARCH_OPT}_timings.txt

echo "Creating output directory: ${OUTPUT_DIR}"
mkdir -p "${OUTPUT_DIR}"

if [ ! -f ${OUTPUT_FILE} ]; then
    touch ${OUTPUT_FILE}
    echo "Created Output File: ${OUTPUT_FILE}"
else
    echo "${OUTPUT_FILE} already exists"
fi



# Start with slurm specific commands
module purge
module add TOOLS
module add slurm
moduel add intel-oneapi-vtune/2024.1.0

if [ $CXX_COMPILER == "gcc" ]; then
    echo "Loading GCC"
    module add gcc/13.2.0
elif [ $CXX_COMPILER == "intel" ]; then
    echo "Loading intel"
    module add intel-oneapi-compilers/2024.1.0
fi

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Maximal threads per process: $SLURM_CPUS_PER_TASK"
echo "Current working directory is `pwd`" 

echo "Starting Compilation"
echo "Compiler: ${CXX_COMPILER}"
echo "CXX_Standard: ${CXX_STANDARD}"
echo "CXX Flags: ${CXX_COMPILER_FLAGS} ${LINK_LIBS}"

${CXX_COMPILER} ${CXX_STANDARD} ${CXX_COMPILER_FLAGS} -I ${INCLUDE_DIRECTORY} ${SOURCE_FILES} -o ${EXECUTABLE} ${LINK_LIBS}

echo "Compilation done"

echo "Creating Vtune result directory ${PROFILING_RESULTS_DIR}_${ARCH_OPT}"
mkdir -p "${PROFILING_RESULTS_DIR}_${ARCH_OPT}"

echo "Starting Vtune"
echo "Collecting: ${ANALYSIS_TYPE}"
echo "OMP_NUM_THREADS: ${PROFILING_THREADS}"

vtune -collect $ANALYSIS_TYPE -result-dir $RESULT_DIR -- $EXECUTABLE

echo "finished profiling"














