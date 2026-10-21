import argparse
import os
from dgl.data.utils import load_graphs

# Argument Parser
parser = argparse.ArgumentParser(description="Find corrupt .bin files in a given directory")
parser.add_argument("--path", type=str, required=True, help="Path to the directory containing .bin files")
args = parser.parse_args()

path = args.path  # Get path from argparse

# List all `.bin` files
bin_files = [f for f in os.listdir(path) if f.endswith(".bin")]

corrupt_files = []

for file in bin_files:
    file_path = os.path.join(path, file)
    try:
        graphs, _ = load_graphs(file_path)
        for g in graphs:
            if 'feat' not in g.ndata:
                print(f"⚠️ Corrupt file (missing features): {file_path}")
                corrupt_files.append(file_path)
            elif g.ndata['feat'].isnan().any():
                print(f"⚠️ Corrupt file (NaN in features): {file_path}")
                corrupt_files.append(file_path)
    except Exception as e:
        print(f"❌ Could not load {file_path}: {e}")
        corrupt_files.append(file_path)

# Print total corrupt files
print(f"\n🛑 Found {len(corrupt_files)} corrupt files!")

