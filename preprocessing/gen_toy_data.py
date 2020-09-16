import os
import json
import numpy as np
import random

def toy_data(genre):
    d_path = '/h/nng/data/sentiment/aws'
    with open(os.path.join(d_path, genre, 'orig', 'train.raw.input0'), 'w') as ifile:
        for i in range(500):
            nrand1 = random.randint(0, 10)
            nrand2 = random.randint(0, 10)
            text = ['none'] * nrand1 + ['one'] + ['none'] * nrand2
            ifile.write(' '.join(text) + '\n')
        for i in range(500):
            nrand1 = random.randint(0, 10)
            nrand2 = random.randint(0, 10)
            text = ['none'] * nrand1 + ['two'] + ['none'] * nrand2
            ifile.write(' '.join(text) + '\n')
    with open(os.path.join(d_path, genre, 'orig', 'train.raw.label'), 'w') as ifile:
        for i in range(500):
            ifile.write('positive\n')
        for i in range(500):
            ifile.write('negative\n')

if __name__ == "__main__":
    for genre in ['cars']:
        toy_data(genre)
