import os
import random
import json
import numpy as np

def separate_data(genre, aug):
    d_path = '/h/nng/data/sentiment/aws'
    with open(os.path.join(d_path, genre, aug, 'raw.json'), 'r') as ifile, \
         open(os.path.join(d_path, genre, aug, 'train.raw.input0'), 'w') as t0file, \
         open(os.path.join(d_path, genre, aug, 'train.raw.label'), 'w') as tlfile, \
         open(os.path.join(d_path, genre, aug, 'valid.raw.input0'), 'w') as v0file, \
         open(os.path.join(d_path, genre, aug, 'valid.raw.label'), 'w') as vlfile, \
         open(os.path.join(d_path, genre, aug, 'test.raw.input0'), 'w') as te0file, \
         open(os.path.join(d_path, genre, aug, 'test.raw.label'), 'w') as telfile:
        tline = [[] for _ in range(5)]
        vline = [[] for _ in range(5)]
        teline = [[] for _ in range(5)]
        for line in ifile:
            row = json.loads(line)
            if 'reviewText' not in row:
                continue
            text = row['reviewText'].encode('unicode_escape').decode('utf-8')
            if len(text.split()) > 300:
                text = ' '.join(text.split()[:300])
            score = int(row['overall'])
            lind = score - 1
            label = str((score-1)/4.0)
            rand = np.random.rand()
            if rand < 0.975:
                tline[lind].append([text, label])
            elif rand < 0.9875:
                vline[lind].append([text, label])
            else:
                teline[lind].append([text, label])

        for i in range(5):
            random.shuffle(tline[i])
            random.shuffle(vline[i])
            random.shuffle(teline[i])

        tmin = min([len(l) for l in tline])
        vmin = min([len(l) for l in vline])
        temin = min([len(l) for l in teline])
        temin = 500
        tmin = 20000
        vmin = 500
        for i in range(5):
            for line in tline[i][:tmin]:
                t0file.write(line[0] + '\n')
                tlfile.write(line[1] + '\n')
            for line in vline[i][:vmin]:
                v0file.write(line[0] + '\n')
                vlfile.write(line[1] + '\n')
            for line in teline[i][:temin]:
                te0file.write(line[0] + '\n')
                telfile.write(line[1] + '\n')

if __name__ == "__main__":
    for genre in ['cars', 'food', 'kindle', 'pets', 'tools']:
        separate_data(genre, 'orig.reg')
