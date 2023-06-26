from datasets import load_dataset, concatenate_datasets
import pandas as pd
import numpy as np
import os
import datasets
# define global variables
SOURCE_LANG = 'en'
TARGET_LANG = 'vi'

SUFFIX_SOURCE = '.en'
SUFFIX_TARGET = '.vn'

def load_data(source_path, target_path):
    """
    Load data from the given source and target paths, and create a dataset.
    
    Args:
        source_path (str): The path to the source file.
        target_path (str): The path to the target file.
    
    Returns:
        dataset (datasets.Dataset): The created dataset.
    """
    data = []
    with open(source_path, 'r') as f1, open(target_path, 'r') as f2:
        for src, tgt in zip(f1, f2):
            data.append(
                {
                    'translation': {
                        SOURCE_LANG: src.strip(),
                        TARGET_LANG: tgt.strip()
                    }
                }
            )
    print(f"total data: {len(data)}")
    tdata = pd.DataFrame(data)
    tdata = tdata.reset_index()
    tdata = tdata.rename(columns={'index': 'id'})

    dataset = datasets.Dataset.from_pandas(tdata)
    return dataset

def load_multi_files(path_dir):
    """
    Load multiple files from a given directory path and return a list of datasets.
    
    :param path_dir: A string representing the path to the directory containing the files.
    :return: A list of datasets loaded from the files in the directory.
    """
    files_list = sorted(os.listdir(path_dir))
    total_pair_files = len(files_list) // 2
    return_result = []

    for i in range(0,len(files_list), 2):
        print("Processing file ", files_list[i])

        name_file = files_list[i].split('.')[0]
        source_path = path_dir + '/' + name_file + SUFFIX_SOURCE
        target_path = path_dir + '/' + name_file + SUFFIX_TARGET
        dataset = load_data(source_path, target_path)
        return_result.append(dataset)
    
    assert len(return_result) == total_pair_files
    return return_result

def concat_dataset(dataset_list):
    """
    Concatenates a list of datasets into a single dataset.
    
    :param dataset_list: A list of datasets to concatenate.
    :type dataset_list: list of datasets
    :return: A concatenated dataset.
    :rtype: dataset
    """
    return concatenate_datasets(dataset_list)

def split_shuffle(dataset):
    """
    Shuffles the input dataset and splits it into train, validation, and test sets with the specified ratios.
    
    Args:
        dataset (datasets.Dataset): The dataset to be split and shuffled.
    
    Returns:
        datasets.DatasetDict: A dictionary containing the shuffled train, validation, and test sets.
    """
    shuffled_dataset = dataset.shuffle(seed=42)
    data_splitted = shuffled_dataset.train_test_split(test_size=0.1)
    test_val = data_splitted['test'].train_test_split(test_size=0.5)
    tmp = datasets.DatasetDict({'test': test_val['test'], 'val': test_val['train'], 'train': data_splitted['train']})
    return tmp

def prepare_dataset(path):
    dataset_list = load_multi_files(path)
    dataset = concat_dataset(dataset_list)
    return split_shuffle(dataset)
def main():
    dataset_list = load_multi_files('/home/maiphong/Documents/20222/code/data')
    print(dataset_list[0]['translation'][0])

    dataset = concat_dataset(dataset_list)
    print(len(dataset))
    x = split_shuffle(dataset)
    print(x)

if __name__ == '__main__':    
    # main()
    pass
