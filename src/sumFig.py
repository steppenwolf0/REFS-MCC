# -*- coding: utf-8 -*-
"""
Created on Thu Jan  7 12:15:27 2021

@author: alber
"""
# convert float to percentage string
def convert_to_percentage(f) :
    if f != "" :
        print("Converting \"%.4f\"..." % f)
        percentage = f 
        return "%.1f" % percentage
    else :
        return ""

# script to create a figure
import os
import sys

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm
import argparse

def create_summary_figure(df: pd.DataFrame, features: pd.DataFrame, totalRuns, output = ".", verbose = 1):
    k = totalRuns

    df.columns = ["features"] + [f"run{i}" for i in range(k)]

    if verbose:
        df.to_csv(os.path.join(output, "best", "sumA.csv"), index=False)

    x = df['features'].values

    print("len features:"+str(len(features.values)))

    maxValue=len(features.values)
   
    runs = [r for r in df.columns if r != 'features']

    #We declare 15 because numbers 0-5 are almost white.
    n_lines=15
    c = np.arange(1, n_lines + 1)

    norm = mpl.colors.Normalize(vmin=c.min(), vmax=c.max())
    cmap = mpl.cm.ScalarMappable(norm=norm, cmap=mpl.cm.Blues)
    cmap.set_array([])

    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111)
    ax.set_xscale('log')

    #starts in 5 because numbers 0-5 are almost white.
    i=5
    for r in runs:
        y = df[r].values
        ax.plot(x, y, label=r, c=cmap.to_rgba(i))
        i=i+1

    tick_positions = [v for v in x if v > 0]
    plt.xticks(tick_positions, [str(a) for a in list(tick_positions)], rotation=90, fontsize=6)

    y_locs = ax.get_yticks()
    print(y_locs)
    plt.yticks(y_locs, [convert_to_percentage(p) for p in y_locs])

    ax.axvline(linewidth=2, color='r', x=maxValue)
    ax.grid(linestyle='--')    
    ax.legend(loc='best')
    ax.set_xlabel("Number of features (log scale)")
    ax.set_ylabel("Ensemble MCC")
    ax.set_title("MCC vs number of features in REFS runs")

    if verbose:
        plt.savefig(os.path.join(output, "sumFig.pdf"))
        plt.savefig(os.path.join(output, "sumFig.png"), dpi=300)

    return plt
    

if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Create summary figure")
    parser.add_argument('--totalRuns', type=int, default=10, help='Total number of runs (default: 10)')
    parser.add_argument('--output', type=str, default=".", help='Path to the output folder (default: .)')
    args = parser.parse_args()

    totalRuns = args.totalRuns
    output = args.output

    df = pd.read_csv(os.path.join(output, "best", "sum.csv"), header=None)
    features=pd.read_csv(os.path.join(output, "best", "features_0.csv"), header=None)

    create_summary_figure(df, features, totalRuns, output)
