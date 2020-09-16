import os
import random
import json
import numpy as np

def separate_data():
    d_path = '/h/nng/data/sentiment/aws'
    genres = ['men', 'women', 'baby', 'shoes']
    with open(os.path.join(d_path, 'clothing', 'orig', 'raw.json'), 'r') as ifile, \
            open(os.path.join(d_path, 'clothing', 'orig', 'meta.json'), 'r') as mfile:
        mdict = {g: {} for g in genres}
        for line in mfile:
            prod = json.loads(line)
            if "Shoes" in prod['category']:
                mdict['shoes'][prod['asin']] = True
            elif "Clothing" in prod['category']:
                if 'Men' in prod['category']:
                    mdict['men'][prod['asin']] = True
                elif 'Women' in prod['category']:
                    mdict['women'][prod['asin']] = True
                elif 'Baby' in prod['category']:
                    mdict['baby'][prod['asin']] = True

        filematrix = {}
        sentmatrix = {}
        for g in genres:
            t0file = open(os.path.join(d_path, g, 'orig', 'train.raw.input0'), 'w')
            tlfile = open(os.path.join(d_path, g, 'orig', 'train.raw.label'), 'w')
            v0file = open(os.path.join(d_path, g, 'orig', 'valid.raw.input0'), 'w')
            vlfile = open(os.path.join(d_path, g, 'orig', 'valid.raw.label'), 'w')
            te0file = open(os.path.join(d_path, g, 'orig', 'test.raw.input0'), 'w')
            telfile = open(os.path.join(d_path, g, 'orig', 'test.raw.label'), 'w')
            filematrix[g] = [t0file, tlfile, v0file, vlfile, te0file, telfile]
            sentmatrix[g] = [[], [], []]

        for line in ifile:
            row = json.loads(line)
            if 'reviewText' not in row:
                continue
            category = None
            for g in genres:
                if row['asin'] in mdict[g]:
                    category = g
            if category is None:
                continue

            text = row['reviewText'].encode('unicode_escape').decode('utf-8')
            if len(text.split()) > 300:
                text = ' '.join(text.split()[:300])
            score = str(int(row['overall']))
            rand = np.random.rand()
            if rand < 0.90:
                sentmatrix[category][0].append([text, score])
            elif rand < 0.95:
                sentmatrix[category][1].append([text, score])
            else:
                sentmatrix[category][2].append([text, score])

        dsizes = [25000, 2000, 2000]

        for g in genres:
            for f in range(3):
                random.shuffle(sentmatrix[g][f])
                for line in sentmatrix[g][f][:dsizes[f]]:
                    filematrix[g][f*2].write(line[0] + '\n')
                    filematrix[g][f*2 + 1].write(line[1] + '\n')

if __name__ == "__main__":
    #for genre in ['kindle', 'pets', 'tools', 'books', 'clothing', 'home', 'movies', 'sports', 'tech', 'toys']:
    #for genre in ['kindle', 'toys']:
    separate_data()
