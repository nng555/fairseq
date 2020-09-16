import os
import sys
import subprocess

import hydra
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path='/h/nng/conf/robust/config.yaml')
def gen_bleu(cfg: DictConfig):
    base_path = '/h/nng/data'
    model_data_path = os.path.join(base_path, cfg.data.task, cfg.eval.model.data)
    eval_data_path = os.path.join(base_path, cfg.data.task, cfg.eval.data)
    model_path = os.path.join('/h/nng/slurm', cfg.eval.model.date, slurm_utils.resolve_name(cfg.eval.model.name))
    if not os.path.exists(os.path.join(model_path, 'checkpoint_best.pt')):
        for f in sorted(os.listdir(model_path))[::-1]:
            if os.path.exists(os.path.join(model_path, f, 'checkpoint_best.pt')):
                model_path = os.path.join(model_path, f)
                break


    model_path = os.path.join(model_path, 'checkpoint_best.pt')

    bin_path = os.path.join(model_data_path, cfg.data.fdset, cfg.data.bin, 'bin')

    t_path = os.path.join(eval_data_path, cfg.data.tdset, 'orig', cfg.eval.split + '.bpe.' + cfg.data.src)
    ref_path = os.path.join(eval_data_path, cfg.data.tdset, 'orig', cfg.eval.split + '.raw.' + cfg.data.tgt)
    bpe_path = '/h/nng/programs/subword-nmt/subword_nmt'

    cat_sh = ['cat', t_path]
    fair_sh = ['fairseq-interactive', bin_path, \
               '--path', model_path, \
               '-s', cfg.data.src, \
               '-t', cfg.data.tgt, \
               '--beam', '10', \
               '--remove-bpe', \
               '--buffer-size', '1024', \
               '--max-tokens', '8000']
    grep_sh = ['grep', '^H-']
    cut_sh = ['cut', '-f', '3-']
    detoken_sh = ['sacremoses', 'detokenize', '-l', cfg.data.tgt, '-q']

    #out_f = open(os.path.join(slurm_utils.get_j_dir(cfg), 'log', 'gen'), 'w')
    cat_p = subprocess.Popen(cat_sh, stdout=subprocess.PIPE)
    fair_p = subprocess.Popen(fair_sh, stdin=cat_p.stdout, stdout=subprocess.PIPE)
    cat_p.stdout.close()
    grep_p = subprocess.Popen(grep_sh, stdin=fair_p.stdout, stdout=subprocess.PIPE)
    fair_p.stdout.close()
    cut_p = subprocess.Popen(cut_sh, stdin=grep_p.stdout, stdout=subprocess.PIPE)
    grep_p.stdout.close()
    detoken_p = subprocess.check_output(detoken_sh, stdin=cut_p.stdout)
    cut_p.stdout.close()
    print(detoken_p.decode('utf-8').strip())

if __name__ == "__main__":
    gen_bleu()

