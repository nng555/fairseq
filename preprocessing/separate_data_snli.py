import os
import numpy as np

def process(line, labels):
    s1_ind = labels.index('sentence1')
    s2_ind = labels.index('sentence2')
    lab_ind = labels.index('gold_label')

    splits = line.split('\t')
    s1 = splits[s1_ind]
    s2 = splits[s2_ind]
    lab = splits[lab_ind]
    return [s1, s2, lab]

def separate(base_dir):
    if not os.path.exists(os.path.join(base_dir, 'orig')):
        os.makedirs(os.path.join(base_dir, 'orig'))
    for split in ['train', 'dev', 'test']:
        base_file = open(os.path.join(base_dir, 'snli_1.0_' + split + '.txt'))
        res = []
        labels = next(base_file).rstrip().split('\t')
        for line in base_file:
            res.append(process(line.rstrip(), labels))
        print(len(res))
        if 'dev' in split:
            split = 'valid'

        with open(os.path.join(base_dir, 'orig', split + '.raw.input0'), 'w') as file0, \
             open(os.path.join(base_dir, 'orig', split + '.raw.input1'), 'w') as file1, \
             open(os.path.join(base_dir, 'orig', split + '.raw.label'), 'w') as filelab:

            for entry in res:
                file0.write(entry[0] + '\n')
                file1.write(entry[1] + '\n')
                filelab.write(entry[2] + '\n')

if __name__ == "__main__":
    separate('/scratch/ssd001/datasets/nng_dataset/nli/snli_1.0/')

