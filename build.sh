#!/usr/bin/env sh

#SBATCH --account=kurs_2024_sose_hpc
#SBATCH --reservation=hpc_course_sose2024

#SBATCH --job-name=profiling
#SBATCH --time=0-00:10:00
#SBATCH --partition=single
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH --mem-per-cpu=7900MB

#SBATCH --output=%u.log.%j.out
#SBATCH --error=%u.log.%j.err

CXX_COMPILER="g++"
BUILD_DIR="./build_pybind"

echo "Starting at `date`"
echo "Running on hosts: $SLURM_NODELIST"
echo "Running on $SLURM_NNODES nodes."
echo "Running on $SLURM_NPROCS processors."
echo "Maximal threads per process: $SLURM_CPUS_PER_TASK"
echo "Current working directory is `pwd`"


# Start with slurm specific commands
module purge
module add slurm
module add zlib/1.3.1
module add cmake/3.27.9
module add miniconda3

source deactivate
source activate data-science

if [ $CXX_COMPILER == "g++" ]; then
    echo "Loading gcc compiler"
    module add gcc/13.2.0
elif [ $CXX_COMPILER == "icx" ]; then
    echo "Loading intel compiler"
    module add intel-oneapi-compilers/2024.1.0
    echo "loading intel toolchains"
    module add toolchains/intel
fi

echo "Creating ${BUILD_DIR}"
mkdir -p ${BUILD_DIR}

echo "Starting Build Process"

cmake -S . -B ${BUILD_DIR} \


cmake --build ${BUILD_DIR}

echo "Finished Build Process"