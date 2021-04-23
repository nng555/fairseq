import os
import numpy as np
import random

def filter_aws(split, num):
    base_dir = '/h/nng/data/sentiment/aws/full'
    with open(os.path.join(base_dir + '5', split + '.raw.input0'), 'r') as ifile5, \
         open(os.path.join(base_dir + '5', split + '.raw.label') , 'r') as lfile5, \
         open(os.path.join(base_dir + '2', split + '.raw.input0'), 'w') as ifile2, \
         open(os.path.join(base_dir + '2', split + '.raw.label') , 'w') as lfile2:

        ex0 = []
        ex1 = []

        for line, label in zip(ifile5, lfile5):
            if label.strip() in ['5', '4']:
                ex1.append((line, '1'))
            elif label.strip() in ['1', '2']:
                ex0.append((line, '0'))

        random.shuffle(ex0)
        random.shuffle(ex1)

        for (line, label) in ex0[:num] + ex1[:num]:
            ifile2.write(line)
            lfile2.write(label + '\n')

if __name__ == "__main__":
    filter_aws('train', 20000)
    filter_aws('test', 2000)
    filter_aws('valid', 2000)
