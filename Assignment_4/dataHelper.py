import json
import os.path
import random

import jsonlines
import pandas as pd
from datasets import Dataset, DatasetDict
from datasets import load_dataset, concatenate_datasets

def prepare_dataset(dataset_name, sep_token, label_offset=0):
    if 'restaurant' or 'laptop' in dataset_name:
        root_dir = './data/SemEval14/SemEval14-' 
        root_dir = root_dir + 'res' if 'res' in dataset_name else root_dir + 'laptop'
        def label2idx(label):
            if label == 'positive': return 0
            elif label == 'neutral': return 1
            else: return 2

        new_data = {}
        for ds in ['train', 'test']:
            new_data[ds] = {}
            new_data[ds]['text'] = []
            new_data[ds]['labels'] = []
            with open(os.path.join(root_dir, ds + '.json')) as f:
                data = json.load(f)
            for _data in data:
                new_data[ds]['text'].append(
                    data[_data]['term'] + ' ' + sep_token + ' ' + data[_data]['sentence'])
                new_data[ds]['labels'].append(label2idx(data[_data]['polarity']) + label_offset)
        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )
        
    elif dataset_name == 'agnews_sup':
        df = pd.read_csv('./data/agnews_sup/agnews_sup.csv',
                         names=['label', 'title', 'description'])

        datasets = Dataset.from_pandas(df)

        def combine_function(example):
            example['text'] = example['description']
            example['labels'] = example['label'] - 1
            return example

        datasets = datasets.map(combine_function,
                                batched=False,
                                num_proc=16,
                                remove_columns=['label', 'title', 'description'])

        datasets = datasets.train_test_split(
            test_size=0.1, seed=2021, shuffle=True)

    elif 'acl' in dataset_name:
        label2idx = {'Uses': 0, 'Future': 1, 'CompareOrContrast': 2, 'Motivation': 3, 'Extends': 4, 'Background': 5}
        new_data = {}
        for ds in ['train', 'test']:
            new_data[ds] = {}
            new_data[ds]['text'] = []
            new_data[ds]['labels'] = []
            with open('./data/acl_sup/{}.jsonl'.format(ds), 'r+') as f:
                for item in jsonlines.Reader(f):
                    new_data[ds]['text'].append(item['text'])
                    new_data[ds]['labels'].append(label2idx[item['intent']] + label_offset)

        datasets = DatasetDict(
            {
                'train': Dataset.from_dict(new_data['train']),
                'test': Dataset.from_dict(new_data['test'])
            }
        )

    if 'fs' in dataset_name:
        seed = 2022
        num_labels = max(datasets['train']['labels']) - label_offset
        ## 32 samples for tasks less than 5 labels, 
        ## 8 samples per class for task with 5 or more than 5 labels
        if num_labels < 4:
            datasets['train'] = datasets['train'].shuffle(seed=seed)
            datasets['train'] = datasets['train'].select(range(32))
        else:
            datasets['train'] = datasets['train'].shuffle(seed=seed)
            _idx = [[] for i in range(num_labels+1)]
            for idx, label in enumerate(datasets['train']['labels']):
                if len(_idx[label]) < 8:
                    _idx[label].append(idx)
            idx_lst = [i for item in _idx for i in item]
            datasets['train'] = datasets['train'].select(idx_lst).shuffle(seed=seed)
    
    return datasets

def get_dataset(dataset_name, sep_token):
    # --- Unsupervised Learning datasets ---
    # attributes: 'text'
    if isinstance(dataset_name, str):
        return prepare_dataset(dataset_name, sep_token)
    elif isinstance(dataset_name, list):
        label_offset = 0
        dataset_list = []
        for dataset in dataset_name:
            datasets = prepare_dataset(dataset, sep_token, label_offset)
            label_offset = max(datasets['train']['labels']) + 1
            dataset_list.append(datasets)
        return concatenate_datasets(dataset_list)
    
