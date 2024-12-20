#!/share/apps/python/anaconda/bin/python

import sys
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def make_plot(runs):
    "Plot results of timing trials"


    # for arg in runs:
    #     # Read the CSV file for the current run
    #     df = pd.read_csv(f"timing-{arg}.csv")
    #     plt.plot(df['batch_size'], df['mflop'] / 1e3, marker='o', label=f"{arg} (Max Batch: {df['batch_size'].max()})")

    # plt.xlabel('batch_size (N)', fontsize=12)
    # plt.ylabel('Gflop/s', fontsize=12)
    # plt.title('Performance Timing Trials', fontsize=14)
    # plt.legend()
    # plt.grid(True, linestyle='--', alpha=0.5)


    for arg in runs:
        # Read the CSV file for the current run
        df = pd.read_csv(f"timing-{arg}.csv")
        plt.plot(df['M'], df['mflop'] / 1e3, marker='o', label=f"{arg if arg != 'basic' else 'ourBlas'}")

    prev = 0
    for _, row in df.iterrows():
        if row['M'] > (prev + 15):
            plt.text(
                row['M'], 
                row['mflop'] / 1e3, 
                f"{row['batch_size']}", 
                fontsize=8, 
                ha='left', 
                va='top',
                color='blue'
            )
            prev = row['M'] 
        else:
            plt.text(
                row['M'], 
                row['mflop'] / 1e3, 
                " ", 
                fontsize=8, 
                ha='right', 
                va='top',
                color='blue'
            )
    plt.xlabel('Dimension (M) (constant batch = 128)', fontsize=12)
    plt.ylabel('Gflop/s', fontsize=12)
    plt.title('Performance Timing Trials', fontsize=14)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)




    # for arg in runs:
    #     df = pd.read_csv("timing-{0}.csv".format(arg))
    #     plt.plot(df['M'], df['mflop'] / 1e3, label=f"{arg} {max(df['batch_size'])}")
    # plt.xlabel('Dimension')
    # plt.ylabel('Gflop/s')

def show(runs):
    "Show plot of timing runs (for interactive use)"
    make_plot(runs)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.show()

def main(runs):
    "Show plot of timing runs (non-interactive)"
    make_plot(runs)
    lgd = plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    
    # Determine the filename for saving
    base_filename = 'timing'
    file_extension = '.pdf'
    file_counter = 0
    new_filename = f"{base_filename}{file_extension}"

    # Check for existing files and generate a new filename if needed
    while os.path.exists(new_filename):
        file_counter += 1
        new_filename = f"{base_filename}{file_counter}{file_extension}"

    # Save the figure
    plt.savefig(new_filename, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print(f"Plot saved as: {new_filename}")

if __name__ == "__main__":
    main(sys.argv[1:])

