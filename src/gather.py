import logging
import argparse
import pathlib
import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

def get_cmdargs():
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--data_directory",
        required=True,
        nargs="*", #multiple arguments can be provided
        help="Path/Paths to the direcories with the benchmarks"
        )
    
    parser.add_argument(
        "--output_directory",
        required=False,
        help="Path to the directory where the images should be saved"
    )

    return parser.parse_args()

def combine_results(data_directories: list) -> list[pd.DataFrame]:

    dfs = []
    
    for directory in data_directories:
        
        directory = pathlib.Path(directory)
        assert directory.is_dir(), f"{directory} does not exist"
        
        for file in directory.glob("*.txt"):
            df = pd.read_csv(file, sep = "\t").assign(parameters = lambda df_: [str(file.stem).replace("_timings", "")] * df_.shape[0])
            dfs.append(df)
            
    return pd.concat(dfs, axis = "rows")
            
        

def main():
    
    cmd_args = get_cmdargs()
    
    df = combine_results(cmd_args.data_directory)
    
    
    
    # Create the pivot table for Speed up compared to single core performance
    piv_table = df.pivot_table(
        index = "OMP_NUM_THREADS", 
        values = "FIT_TIME", 
        columns = ["parameters"], 
        aggfunc="mean"
    )

    # T_n=1 / T_n
    piv_table = piv_table.div(piv_table.iloc[0], axis = "columns").pow(-1)
    
    #ideal line
    
    y_values = np.arange(1, piv_table.index[-1] + 1, 1)
    
    # create plots
    fig, (ax1, ax2) = plt.subplots(
        1, 2, 
        sharex=True, 
        layout="constrained", 
        figsize=(15, 5)
    )
    
    
    piv_table.plot(
        kind = "line", 
        marker = "o", 
        grid = True,
        ax = ax1
    )
    
    ax1.plot(
        y_values, 
        y_values, 
        color='black', 
        linestyle='--', 
        label='ideal', 
        zorder=1
    )
    
    ax1.set_xlabel("Number of OpenMP threads")
    ax2.set_ylabel("Speed up $S_n$ w.r.t using single core")
    ax1.legend()
    
    # calculate Compute performance (iterations/second)
    df_iterations = df.assign(iterations_second = lambda df_: df_["NUM_ITERATIONS"] / (df_["FIT_TIME"] / 1000))
    
    piv_table_iterations = df_iterations.pivot_table(
        index = "OMP_NUM_THREADS", 
        values = "iterations_second", 
        columns = ["parameters"], 
        aggfunc = "mean"
    )
    
    piv_table_iterations.plot(
        kind = "line", 
        marker = "o", 
        grid = True,
        ax = ax2
    )
    
    ax2.set_xlabel("Number of OpenMP threads")
    ax2.set_ylabel("Compute performance (iterations/second)")
    ax2.legend()
    
    if cmd_args.output_directory is not None:
        output_directory = pathlib.Path(cmd_args.output_directory)
        if not output_directory.exists():
        # Create the directory (and any necessary parent directories)
            output_directory.mkdir(parents=True, exist_ok=True)
            print(f"Directory {output_directory} created.")
        else:
            print(f"Directory {output_directory} already exists.")
            
        save_file = output_directory / "KMeans_Performance.png"
        fig.savefig(save_file, bbox_inches="tight")
    else:
        plt.show()
        

if __name__ == "__main__":
    main()

    
    
    
    