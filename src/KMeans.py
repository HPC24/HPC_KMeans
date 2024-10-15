import pathlib
import argparse
import numpy as np
import pandas as pd
import time
import os
from sklearn.cluster import KMeans
from sklearn.datasets import fetch_openml

TIMING_ITERATIONS = 20
KMEANS_INIT="random"
KMEANS_N_INIT=1
N_CLUSTER = 10
TOL = 1e-9
MAX_ITER = 500
SEED = 42

def parse_args():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--output_dir",
                        required = True,
                        help = "Path to the output directory where the results should be saved")
    
    return parser.parse_args()


def fit_kmeans(kmeans, data) -> float:
    
    start_time = time.perf_counter()
    
    kmeans.fit(data)
    
    end_time = time.perf_counter()
    
    iterations = kmeans.n_iter
    
    return end_time, iterations
    
    
def main():
    
    cmd_args = parse_args()
    output_dir = pathlib.Path(cmd_args.output_dir)
    assert output_dir.is_dir(), f"{output_dir} does not exist"   
    output_file = output_dir / "sklearn_timings.txt" 

    timings = []
    iterations = []
    omp_num_threads = os.environ.get("OMP_NUM_THREADS")
    
    print("Fetching MNIST dataset")
    
    mnist = fetch_openml('mnist_784', version=1)
    data = mnist.data.to_numpy()
    
    print("Done")
    
    print("Initializing KMeans object with following Parameters:")
    print(f"N_CLUSTER: {N_CLUSTER}")
    print(f"INIT: {KMEANS_INIT}")
    print(f"N_INIT: {KMEANS_N_INIT}")
    print(f"TOLERANCE: {KMEANS_N_INIT}")
    print(f"MAX_ITER: {MAX_ITER}")
    print(f"SEED: {SEED}")

    print(f"Starting timing with {omp_num_threads} OMP_NUM_THREADS")
    
    for i in range(TIMING_ITERATIONS + 1):
        
         kmeans = KMeans(n_clusters = N_CLUSTER, 
                    init = KMEANS_INIT, 
                    n_init = KMEANS_N_INIT, 
                    tol = TOL,
                    max_iter = MAX_ITER,
                    random_state = SEED)
         
         time, iteration = fit_kmeans(kmeans, data = data)
         timings.append(time)
         iterations.append(iteration)
         
    with open(output_file, "r") as file:
        
        content = file.read
        
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