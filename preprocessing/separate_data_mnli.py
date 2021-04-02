import os
import numpy as np

def process(line, labels):
    s1_ind = labels.index('sentence1')
    s2_ind = labels.index('sentence2')
    lab_ind = labels.index('gold_label')
    genre_ind = labels.index('genre')

    splits = line.split('\t')
    s1 = splits[s1_ind]
    s2 = splits[s2_ind]
    lab = splits[lab_ind]
    genre = splits[genre_ind]
    return [s1, s2, lab, genre]

def separate(base_dir):
    for split in ['train', 'dev_matched', 'dev_mismatched']:
        base_file = open(os.path.join(base_dir, split + '.tsv'))
        res = []
        labels = next(base_file).rstrip().split('\t')
        for line in base_file:
            res.append(process(line.rstrip(), labels))
        print(len(res))
        if 'dev' in split:
            split = 'valid'

        for entry in res:
            if not os.path.exists(os.path.join(base_dir, entry[3], 'orig')):
                os.makedirs(os.path.join(base_dir, entry[3], 'orig'))
            with open(os.path.join(base_dir, entry[3], 'orig', split + '.raw.input0'), 'a+') as file0:
                file0.write(entry[0] + '\n')
            with open(os.path.join(base_dir, entry[3], 'orig', split + '.raw.input1'), 'a+') as file1:
                file1.write(entry[1] + '\n')
            with open(os.path.join(base_dir, entry[3], 'orig', split + '.raw.label'), 'a+') as filelab:
                filelab.write(entry[2] + '\n')

if __name__ == "__main__":
    separate('/scratch/ssd001/datasets/nli/mnli')

