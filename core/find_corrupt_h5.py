
import argparse
import os
import h5py

# Argument Parser
parser = argparse.ArgumentParser(description="Find corrupt .h5 files in a given directory")
parser.add_argument("--path", type=str, required=True, help="Path to the directory containing .h5 files")
args = parser.parse_args()

path = args.path  # Get path from argparse

# List all `.h5` files
h5_files = [f for f in os.listdir(path) if f.endswith(".h5")]

corrupt_files = []

for file in h5_files:
    file_path = os.path.join(path, file)
    try:
        with h5py.File(file_path, 'r') as f:
            if not list(f.keys()):  # Check if the file is empty
                print(f"⚠️ Corrupt file (empty file): {file_path}")
                corrupt_files.append(file_path)
            else:
                # Check for missing datasets
                for key in f.keys():
                    try:
                        data = f[key][()]
                        if data is None or (isinstance(data, (list, tuple)) and len(data) == 0):
                            print(f"⚠️ Corrupt file (missing or empty dataset '{key}'): {file_path}")
                            corrupt_files.append(file_path)
                    except Exception as e:
                        print(f"⚠️ Corrupt file (error reading dataset '{key}'): {file_path} - {e}")
                        corrupt_files.append(file_path)
    except Exception as e:
        print(f"❌ Could not open {file_path}: {e}")
        corrupt_files.append(file_path)

# Print total corrupt files
print(f"\n🛑 Found {len(corrupt_files)} corrupt files!")