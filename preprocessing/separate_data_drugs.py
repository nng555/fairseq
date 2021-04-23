import os
import html
import jsonlines as jsonl
import numpy as np
import csv

def separate(base_dir):
    if not os.path.exists(os.path.join(base_dir, 'orig')):
        os.makedirs(os.path.join(base_dir, 'orig'))
    for split in ['train', 'valid', 'test']:
        base_file = os.path.join(base_dir, 'drugsCom' + split.capitalize() + '_raw.tsv')
        res = []
        with open(base_file) as ifile:
            for ex in csv.reader(ifile, delimiter='\t'):
                if float(ex[4].strip()) > 6.0:
                    label = '1'
                    res.append([html.unescape(ex[3].strip()), label])
                elif float(ex[4].strip()) < 4.0:
                    label = '0'
                    res.append([html.unescape(ex[3].strip()), label])
        print(len(res))

        with open(os.path.join(base_dir, 'orig', split + '.raw.input0'), 'w') as file0, \
             open(os.path.join(base_dir, 'orig', split + '.raw.label'), 'w') as filelab:

            for entry in res:
                file0.write(repr(entry[0][1:-1])[1:-1] + '\n')
                filelab.write(entry[1] + '\n')

if __name__ == "__main__":
    separate('/scratch/ssd001/datasets/nng_dataset/sentiment/drugs')

