import os
import numpy as np

def encode_soft_labels(data_path, genre, aug, split, num_classes):
    with open(os.path.join(data_path, genre, aug, split + '.raw.label'), 'w') as labelf, \
         open(os.path.join(data_path, genre, aug, split + '.soft.label'), 'r') as softf:
        orddict = [[float(val) for val in l.strip().split()] for l in softf.readlines()]
        for row in orddict:
            row = np.asarray(row)
            row = row - np.pad(row[1:], (0, 1), 'constant')
            label = row.argmax() + 1
            labelf.write(str(label) + '\n')

if __name__ == '__main__':
    #for genre in ['slate', 'fiction', 'telephone', 'travel', 'government']:
    #    for aug in ['orig']:
    #        for split in ['train', 'dev_matched']:
    #            encode_soft_labels('/scratch/ssd001/datasets/nli/mnli', genre, aug, split, 3)
    #for genre in ['R1', 'R2', 'R3']:
    #    for aug in ['orig']:
    #        for split in ['train', 'dev', 'test']:
    #            encode_soft_labels('/scratch/ssd001/datasets/nli/anli', genre, aug, split, 3)
    #for genre in ['books', 'clothing', 'home', 'kindle', 'movies', 'pets', 'sports', 'tech', 'tools', 'toys']:
    #    for aug in ['orig']:
    #        for split in ['train', 'valid', 'test']:
    #            encode_soft_labels('/h/nng/data/sentiment/aws', genre, aug, split, 5)
    encode_soft_labels('/h/nng/data/sentiment/aws', 'movies', 'sampling_ord', 'train', 5)
