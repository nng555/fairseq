import os
import csv

def eec_preprocess():
    bpath = "/h/nng/data/sentiment/eec"
    with open(os.path.join(bpath, 'eec.csv'), 'r') as csv_file, \
         open(os.path.join(bpath, 'full', 'test.raw.input0'), 'w') as full_input, \
         open(os.path.join(bpath, 'full', 'test.raw.label'), 'w') as full_label:
        reader = csv.reader(csv_file)
        header = next(reader)
        s_idx = header.index("Sentence")
        g_idx = header.index("Gender")
        r_idx = header.index("Race")
        for row in reader:
            sent = row[s_idx]
            race = row[r_idx]
            gender = row[g_idx]
            full_input.write(sent + '\n')
            full_label.write('0\n')

if __name__ == "__main__":
    eec_preprocess()
