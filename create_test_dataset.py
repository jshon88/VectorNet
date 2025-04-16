import os
import random
import shutil
from utils.config import DATA_DIR


def create_test_dataset(data_dir, val_folder='val', test_folder='test', test_ratio=0.5, seed=42):
    '''
    Splits the CSV files from the validation folder into a test set
    '''
    test_dir = os.path.join(data_dir, test_folder)
    
    if not os.path.exists(test_dir):
        os.makedirs(test_dir)
        print(f'Created folder: {test_dir}')

    val_dir = os.path.join(data_dir, val_folder)
    csv_files = [f for f in os.listdir(val_dir) if f.endswith('.csv')]

    if not csv_files:
        print('No CSV files found in validation folder')
        return

    random.seed(seed)
    random.shuffle(csv_files)

    num_test_files = int(len(csv_files) * test_ratio)
    test_files = csv_files[:num_test_files]

    cnt = 0
    for file_name in test_files:
        src = os.path.join(val_dir, file_name)
        dst = os.path.join(test_dir, file_name)
        shutil.move(src, dst)
        cnt += 1
    print(f'{cnt} number of files are moved to test folder')

if __name__ == '__main__':
    create_test_dataset(data_dir=DATA_DIR, val_folder='val', test_folder='test', test_ratio=0.5, seed=42)
    