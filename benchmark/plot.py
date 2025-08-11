#!/usr/bin/env python3

import argparse
import matplotlib.pyplot as plt
import pandas as pd


def main(args):
    df = pd.read_csv(args.csv_file)

    run_cols = [col for col in df.columns if col.startswith("run")]
    x = run_cols

    for degree, group in df.groupby("degree"):
        y = group[run_cols].values.flatten()
        plt.plot(x, y, marker='o', label=f"degree {degree}")

    plt.xlabel("Run")
    plt.ylabel("Time (ms)")
    plt.title("Benchmark Results by Degree")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot benchmark results from CSV.")
    parser.add_argument("csv_file", help="Path to the csv file")
    args = parser.parse_args()

    main(args)
