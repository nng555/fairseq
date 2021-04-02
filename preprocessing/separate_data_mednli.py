import os
import jsonlines as jsonl
import numpy as np

def separate(base_dir):
    if not os.path.exists(os.path.join(base_dir, 'orig')):
        os.makedirs(os.path.join(base_dir, 'orig'))
    for split in ['train', 'dev', 'test']:
        base_file = jsonl.open(os.path.join(base_dir, 'mli_' + split + '_v1.jsonl'))
        res = []
        for ex in base_file:
            res.append([ex['sentence1'], ex['sentence2'], ex['gold_label']])
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
    separate('/scratch/ssd001/datasets/nng_dataset/nli/mednli')

