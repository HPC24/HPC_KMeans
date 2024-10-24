import pathlib
import argparse
import numpy as np
import pandas as pd
import time
import os
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml
import sys

TIMING_ITERATIONS = 20
N_CLUSTER = 10
TOL = 1e-9
MAX_ITER = 500
SEED = 42

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir",
                        required = True,
                        help = "Path to the output directory where the results should be saved"
                        )
                        
    parser.add_argument("--output_file",
			required = True,
			help = "Name of the output file where the results will be saved"
			)
    
    parser.add_argument("--build",
                        required = True,
                        help = "Path to the build directory containing the pybind11 .so file")
    
    return parser.parse_args()


def fit_kmeans(kmeans, data) -> float:
    
    start_time = time.perf_counter()
    
    kmeans.fit(data)
    
    end_time = time.perf_counter()
    # get the fit time in ms 
    fit_time = (end_time - start_time) * 1000
    
    iterations = kmeans.n_iter_
    
    return fit_time, iterations
    
    
def main():
    
    custom_cache_dir = "/scratch/kurs_2024_sose_hpc/kurs_2024_sose_hpc_11/data"
    cmd_args = parse_args()
    output_dir = pathlib.Path(cmd_args.output_dir)
    build_dir = pathlib.Path(cmd_args.build)
    assert output_dir.is_dir(), f"{output_dir} does not exist" 
    assert build_dir.is_dir(), f"{build_dir} does not exist"
    
    sys.path.append(str(build_dir))
    import P_KMeansLib
    
    output_file = output_dir / cmd_args.output_file

    timings = []
    iterations = []
    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    
    print("Fetching MNIST dataset if it is not already loaded")
    
    mnist = fetch_openml('mnist_784', data_home = custom_cache_dir)
    data = mnist.data.to_numpy()
    
    print("Done")
    
    print("Initializing KMeans object with following Parameters:")
    print(f"N_CLUSTER: {N_CLUSTER}")
    print(f"TOLERANCE: {TOL}")
    print(f"MAX_ITER: {MAX_ITER}")
    print(f"SEED: {SEED}")

    print(f"Starting timing with {omp_num_threads} OMP_NUM_THREADS")
    
    for i in range(TIMING_ITERATIONS + 1):
        
        kmeans = P_KMeansLib.Parallel_KMeans_Float(N_CLUSTER, MAX_ITER, TOL, SEED)
        
        time, iteration = fit_kmeans(kmeans, data = data)
        timings.append(time)
        iterations.append(iteration)
         
    with open(output_file, "r") as file:
        
        content = file.read()
        
        if not content:
            file_is_empty = True
        else:
            file_is_empty = False
            
    print("Writing timings to file")
        
    with open(output_file, "a") as file:
        
        if file_is_empty:
            file.write("OMP_NUM_THREADS\tFIT_TIME\tNUM_ITERATIONS\n")
        else:
            for fit_time, num_iterations in zip(timings, iterations):
                file.write(f"{omp_num_threads}\t{fit_time}\t{num_iterations}\n")
         
    print("Finished writing to file")
    
if __name__ == "__main__":
    main()