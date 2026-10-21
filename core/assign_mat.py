import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# visualize_assignment.py
# Visualize binary .h5 assignment matrices using a dot plot
# Author: [Your Name]

import os
import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt

def visualize_assignment_matrix(h5_path, output_root):
    # --- Check input file ---
    if not os.path.isfile(h5_path):
        raise FileNotFoundError(f"{h5_path} does not exist.")

    # --- Load matrix ---
    with h5py.File(h5_path, "r") as f:
        matrix = f["assignment_matrix"][:]

    # --- Prepare output directory ---
    assign_mat_dir = os.path.join(output_root, "assign_mat")
    os.makedirs(assign_mat_dir, exist_ok=True)

    # --- Extract (row, col) positions where A[i,j] == 1 ---
    rows, cols = np.where(matrix == 1)

    # --- Dot plot ---
    plt.figure(figsize=(8, 6))
    plt.scatter(cols, rows, s=4, c='blue', alpha=0.8)
    plt.gca().invert_yaxis()
    plt.xlabel("Tissue Region Index (Columns)")
    plt.ylabel("Nucleus Index (Rows)")
    plt.title("Assignment Matrix (Nuclei → Tissue) - Dot Visualization")
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()

    # --- Save plot ---
    base_name = os.path.splitext(os.path.basename(h5_path))[0]
    out_path = os.path.join(assign_mat_dir, base_name + "_assign_dots.png")
    plt.savefig(out_path, dpi=300)
    plt.close()

    print(f"[✓] Saved assignment matrix dot plot to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Path to input .h5 file")
    parser.add_argument("--output", type=str, required=True, help="Directory to save outputs")
    args = parser.parse_args()

    visualize_assignment_matrix(args.input, args.output)
