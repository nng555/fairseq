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
         open(os.path.join(d_path, genre, 'orig', 'test.raw.label'), 'w') as telfile, \
         open(os.path.join(d_path, genre, 'unlabelled', 'train.raw.input0'), 'w') as ut0file, \
         open(os.path.join(d_path, genre, 'unlabelled', 'train.raw.label'), 'w') as utlfile, \
         open(os.path.join(d_path, genre, 'unlabelled', 'valid.raw.input0'), 'w') as uv0file, \
         open(os.path.join(d_path, genre, 'unlabelled', 'valid.raw.label'), 'w') as uvlfile, \
         open(os.path.join(d_path, genre, 'unlabelled', 'test.raw.input0'), 'w') as ute0file, \
         open(os.path.join(d_path, genre, 'unlabelled', 'test.raw.label'), 'w') as utelfile:
        orig_in_files = [t0file, v0file, te0file]
        orig_l_files = [tlfile, vlfile, telfile]
        unl_in_files = [ut0file, uv0file, ute0file]
        unl_l_files = [utlfile, uvlfile, utelfile]

        print("Processing data for {}".format(genre))
        lines = []
        for line in ifile:
            if len(lines) > 2000000:
                break
            row = json.loads(line)
            if 'reviewText' not in row:
                continue
            text = row['reviewText'].encode('unicode_escape').decode('utf-8')
            if len(text.split()) > 300:
                text = ' '.join(text.split()[:300])
            score = int(row['overall'])
            lines.append([text, score])

        print("Shuffling data for {}".format(genre))
        random.shuffle(lines)

        # 25k train, 2k valid, 2k test
        train = lines[:25000]
        valid = lines[25000:25000+2000]
        test = lines[27000:27000+2000]
        oarrays = [train, valid, test]

        # 1.5m train, 10k valid, 10k test
        utrain = lines[29000:1529000]
        uvalid = lines[1529000:1539000]
        utest = lines[1539000:1549000]
        uarrays = [utrain, uvalid, utest]

        print("Writing data for {}".format(genre))
        for i in range(3):
            for line in oarrays[i]:
                orig_in_files[i].write(line[0] + '\n')
                orig_l_files[i].write(str(line[1]) + '\n')
            for line in uarrays[i]:
                unl_in_files[i].write(line[0] + '\n')
                unl_l_files[i].write(str(line[1]) + '\n')

if __name__ == "__main__":
    random.seed(1)
    for genre in ['kindle', 'pets', 'tools', 'books', 'clothing', 'home', 'movies', 'sports', 'tech', 'toys']:
        separate_data(genre)
