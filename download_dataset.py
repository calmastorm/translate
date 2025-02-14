from datasets import load_dataset
import os
from config import *

def download(dataset_name=DATASET_NAME, dataset_language_pair=DATASET_LANGUAGE_PAIR, save_directory=DATASET_SAVE_DIRECTORY):
    print(f'Loading {dataset_name} to {save_directory}')
    dataset = load_dataset(dataset_name, dataset_language_pair)

    os.makedirs(save_directory, exist_ok=True)

    for split in ['train', 'validation']:
        dataset[split].save_to_disk(f'{save_directory}/{split}')
        print(f'The {split} set saved')

    print(f'Dataset {dataset_name} has been downloaded and saved at {save_directory}')

if __name__ == '__main__':
    download()