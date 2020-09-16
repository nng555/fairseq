import os
import random
import json
import numpy as np

def separate_data(genre):
    d_path = '/h/nng/data/sentiment/aws'
    with open(os.path.join(d_path, genre, 'orig', 'raw.json'), 'r') as ifile, \
         open(os.path.join(d_path, genre, 'orig', 'train.raw.input0'), 'w') as t0file, \
         open(os.path.join(d_path, genre, 'orig', 'train.raw.label'), 'w') as tlfile, \
         open(os.path.join(d_path, genre, 'orig', 'valid.raw.input0'), 'w') as v0file, \
         open(os.path.join(d_path, genre, 'orig', 'valid.raw.label'), 'w') as vlfile, \
         open(os.path.join(d_path, genre, 'orig', 'test.raw.input0'), 'w') as te0file, \
         open(os.path.join(d_path, genre, 'orig', 'test.raw.label'), 'w') as telfile:
        vline = []
        #tline = [[] for _ in range(5)]
        tline = []
        teline = []
        for line in ifile:
            row = json.loads(line)
            if 'reviewText' not in row:
                continue
            text = row['reviewText'].encode('unicode_escape').decode('utf-8')
            if len(text.split()) > 300:
                text = ' '.join(text.split()[:300])
            score = int(row['overall'])
            if score == 1:
                label = 'negative'
            elif score in [2, 3, 4]:
                label = 'neutral'
            else:
                label = 'positive'

            '''
            if score < 3:
                label = 'negative'
                lind = 0
            elif score == 3:
                label = 'neutral'
                lind = 1
            else:
                label = 'positive'
                lind = 2
            '''
            rand = np.random.rand()
            if rand < 0.90:
                #tline[lind].append([text, label])
                tline.append([text, label])
            elif rand < 0.95:
                vline.append([text, label])
            else:
                teline.append([text, label])

        #for i in range(5):
        random.shuffle(tline)
        random.shuffle(vline)
        random.shuffle(teline)

        #tmin = min([len(l) for l in tline])
        #vmin = min([len(l) for l in vline])
        #temin = min([len(l) for l in teline])
        temin = 2000
        tmin = 25000
        vmin = 2000
        for line in tline[:tmin]:
            t0file.write(line[0] + '\n')
            tlfile.write(line[1] + '\n')
        for line in vline[:vmin]:
            v0file.write(line[0] + '\n')
            vlfile.write(line[1] + '\n')
        for line in teline[:temin]:
            te0file.write(line[0] + '\n')
            telfile.write(line[1] + '\n')

if __name__ == "__main__":
    for genre in ['kindle', 'pets', 'tools', 'books', 'clothing', 'home', 'movies', 'sports', 'tech', 'toys']:
    #for genre in ['kindle', 'toys']:
        separate_data(genre)
