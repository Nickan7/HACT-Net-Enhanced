import os
import argparse
import random
import shutil


def create_folders(base_path, percent):
    for folder in ['train', 'val', 'test']:
        path = os.path.join(base_path, f"{folder}_{percent}")
        os.makedirs(path, exist_ok=True)

def split_data(file_list, train_ratio=0.7, val_ratio=0.15):
    random.shuffle(file_list)
    total = len(file_list)
    train_end = int(train_ratio * total)
    val_end = train_end + int(val_ratio * total)
    train_files = file_list[:train_end]
    val_files = file_list[train_end:val_end]
    test_files = file_list[val_end:]
    return train_files, val_files, test_files

def move_files(file_list, dest_folder, src_folder):
    for file in file_list:
        src_path = os.path.join(src_folder, file)
        dst_path = os.path.join(dest_folder, file)
        shutil.move(src_path, dst_path)

def main(data_folder, percentance):
    # Get only image files
    all_files = [f for f in os.listdir(data_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Apply percentage
    if percentance < 100:
        n = int(len(all_files) * percentance / 100)
        all_files = random.sample(all_files, n)

    # Split
    train_files, val_files, test_files = split_data(all_files)

    # Create folders with suffix
    create_folders(data_folder, percentance)

    # Move files
    move_files(train_files, os.path.join(data_folder, f'train_{percentance}'), data_folder)
    move_files(val_files, os.path.join(data_folder, f'val_{percentance}'), data_folder)
    move_files(test_files, os.path.join(data_folder, f'test_{percentance}'), data_folder)

    print(f"Done! ({percentance}%) - Train: {len(train_files)}, Val: {len(val_files)}, Test: {len(test_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type=str, required=True, help='Folder with files')
    parser.add_argument('--percentance', type=int, default=100, help='Percentage of total files to use (default: 100)')
    args = parser.parse_args()
    main(args.data_folder, args.percentance)
