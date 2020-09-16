import os
import hydra
import subprocess
import logging
from omegaconf import DictConfig
from hydra import slurm_utils

log = logging.getLogger(__name__)

@hydra.main(config_path='/h/nng/conf/robust/config.yaml', strict=False)
def launch(cfg: DictConfig):
    slurm_utils.symlink_hydra(cfg, os.getcwd())

    os.environ['NCCL_DEBUG'] = 'INFO'

    if cfg.data.task in ['nli']:
        base_path = '/scratch/ssd001/datasets/'
    elif cfg.data.task in ['sentiment', 'translation']:
        base_path = '/h/nng/data'
    else:
        raise Exception('task %s data path not found'.format(cfg.data.task))

    bin_path = os.path.join(base_path, cfg.data.task, cfg.data.name, cfg.data.fdset, cfg.data.bin, 'bin')
    j_dir = slurm_utils.get_j_dir(cfg)

    if os.path.exists(os.path.join(j_dir, os.environ['SLURM_JOB_ID'], 'checkpoint_last.pt')):
        cfg.train.restore_file = os.path.join(j_dir, os.environ['SLURM_JOB_ID'], 'checkpoint_last.pt')
    elif os.path.exists(os.path.join(j_dir, 'checkpoint_last.pt')):
        cfg.train.restore_file = os.path.join(j_dir, 'checkpoint_last.pt')
    else:
        found = False
        for f in sorted(os.listdir(j_dir))[::-1]:
            if os.path.exists(os.path.join(j_dir, f, 'checkpoint_last.pt')):
                cfg.train.restore_file = os.path.join(j_dir, f, 'checkpoint_last.pt')
                found = True
                break

        if cfg.data.task in ['nli'] and not found:
            cfg.train.restore_file = '/scratch/hdd001/home/nng/roberta/roberta.base/model.pt'
            cfg.train.reset_optimizer = True
            cfg.train.reset_dataloader = True
            cfg.train.reset_meters = True

    flags = [['--' + k.replace('_', '-'), slurm_utils.eval_val(str(v))] for k, v in cfg.train.items() if v != None]
    flags = [val for sublist in flags if sublist[1] != "False" for val in sublist if val != "True"]
    flags = flags + ['--save-dir', os.path.join(j_dir, os.environ['SLURM_JOB_ID'])]

    command = ' '.join(['fairseq-train', bin_path] + flags)
    log.info(command)
    subprocess.call(['fairseq-train', bin_path] + flags)

if __name__ == "__main__":
    launch()
