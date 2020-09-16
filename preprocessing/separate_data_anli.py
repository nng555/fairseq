import os
import jsonlines

label_dict = {'c': 'contradiction', 'n': 'neutral', 'e': 'entailment'}

def process(line):
    s1 = line['context'].strip().encode('unicode_escape').decode('utf-8')
    s2 = line['hypothesis'].strip().encode('unicode_escape').decode('utf-8')
    res = label_dict[line['label']]
    return [s1, s2, res]

def separate(base_dir, split):
    base_file = jsonlines.open(os.path.join(base_dir, 'raw', split + '.jsonl'))
    res = []
    for line in base_file:
        res.append(process(line))
    with open(os.path.join(base_dir, 'orig', split + '.tmp.input0'), 'w') as file0, \
         open(os.path.join(base_dir, 'orig', split + '.tmp.input1'), 'w') as file1, \
         open(os.path.join(base_dir, 'orig', split + '.tmp.label'), 'w') as filel:
        for entry in res:
            if entry[0] == "" or entry[1] == "":
                print(entry)
            file0.write(entry[0] + '\n')
            file1.write(entry[1] + '\n')
            filel.write(entry[2] + '\n')

if __name__ == "__main__":
    for split in ['train', 'dev', 'test']:
        separate('/scratch/ssd001/datasets/nli/anli/R3', split)

