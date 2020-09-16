import os
import numpy

def encode_soft_labels(data_path, genre, aug, split, num_classes):
    with open(os.path.join(data_path, 'label.dict.txt'), 'r') as dictf, \
         open(os.path.join(data_path, genre, aug, split + '.raw.label'), 'r') as labelf, \
         open(os.path.join(data_path, genre, aug, split + '.ord.label'), 'w') as softf:
        dictf = dictf.readlines()
        labeldict = [l.strip().split()[0] for l in dictf[:num_classes]]
        for row in labelf:
            probs = [0.0] * num_classes
            for i in range(labeldict.index(row.strip()) + 1):
                probs[i] = 1.0
            probs = [str(prob) for prob in probs]
            softf.write(' '.join(probs) + '\n')

if __name__ == '__main__':
    #for genre in ['slate', 'fiction', 'telephone', 'travel', 'government']:
    #    for aug in ['orig']:
    #        for split in ['train', 'dev_matched']:
    #            encode_soft_labels('/scratch/ssd001/datasets/nli/mnli', genre, aug, split, 3)
    #for genre in ['R1', 'R2', 'R3']:
    #    for aug in ['orig']:
    #        for split in ['train', 'dev', 'test']:
    #            encode_soft_labels('/scratch/ssd001/datasets/nli/anli', genre, aug, split, 3)
    for genre in ['books', 'clothing', 'home', 'kindle', 'movies', 'pets', 'sports', 'tech', 'tools', 'toys']:
        for aug in ['orig']:
            for split in ['train', 'valid', 'test']:
                encode_soft_labels('/h/nng/data/sentiment/aws', genre, aug, split, 5)
