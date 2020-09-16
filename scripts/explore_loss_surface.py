from fairseq.models.roberta import RobertaModel
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from matplotlib import cm

def explore_loss_surface(label, args):
    r_mnli = RobertaModel.from_pretrained(
        '/scratch/ssd001/home/nng/slurm/' + args.checkpoint,
        checkpoint_file = 'checkpoint_best.pt',
        data_name_or_path = '/scratch/hdd001/datasets/wmt/MNLI/' + args.train + '/bin'
    )
    r_mnli.eval()

    n_tok = torch.load(args.data + '/tok.pt')
    embed = np.load(args.data + '/embed.npy')

    label_fn = lambda label: roberta.task.label_dictionary.string(
        [label + roberta.task.label_dictionary.nspecial]
    )
    targets = {'neutral': torch.tensor([0]),
               'contradiction': torch.tensor([1]),
               'entailment': torch.tensor([2]) }

    batch_num = int(len(n_tok)/args.batch) + 1
    n_loss = []
    for b in range(batch_num):
        b_start = b * args.batch
        b_end = (b + 1) * args.batch
        batch = n_tok[b_start:b_end]
        target = targets[label].repeat(min(len(n_tok) - b_start, args.batch))
        pred_prob = r_mnli.predict('sentence_classification_head', batch)
        loss = F.cross_entropy(pred_prob, target, reduction='none').flatten().tolist()
        n_loss.extend(loss)

    np.save(args.save_dir + '/data.npz', np.hstack((embed, np.expand_dims(n_loss, axis=1))))

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(embed[0][0], embed[0][1], n_loss[0], color='b')
    ax.plot_trisurf(embed[:,0], embed[:,1], n_loss[:], cmap=cm.coolwarm, linewidth=0, antialiased=False)
    plt.savefig(args.save_dir + "/out.png")

if __name__ == "__main__":

    label = "entailment"
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--checkpoint', help='tokens and embeddings directory to use')
    parser.add_argument('-d', '--data', help='data storage directory')
    parser.add_argument('-t', '--train', help='the training domain')
    parser.add_argument('-s', '--save-dir', help='where to save generated images')
    parser.add_argument('-b', '--batch', help='batch size', default=8, type=int)
    args = parser.parse_args()
    explore_loss_surface(label, args)
