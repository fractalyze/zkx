#!/usr/bin/env python3

# Copyright 2025 The ZKX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

import argparse
import matplotlib.pyplot as plt
import pandas as pd


def main(args):
    df = pd.read_csv(args.csv_file)

    run_cols = [col for col in df.columns if col.startswith("run")]
    x = run_cols

    for degree, group in df.groupby("degree"):
        y = group[run_cols].values.flatten()
        plt.plot(x, y, marker="o", label=f"degree {degree}")

    plt.xlabel("Run")
    plt.ylabel("Time (ms)")
    plt.title("Benchmark Results by Degree")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot benchmark results from CSV.")
    parser.add_argument("csv_file", help="Path to the csv file")
    args = parser.parse_args()

    main(args)
