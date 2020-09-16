import os
import hydra
import subprocess
import logging
from omegaconf import DictConfig
from hydra import slurm_utils

log = logging.getLogger(__name__)

@hydra.main(config_path='/h/nng/conf/robust/config.yaml', strict=False)
def launch(cfg: DictConfig):
    os.environ['NCCL_DEBUG'] = 'INFO'

    if cfg.gen.task in ['nli']:
        base_path = '/scratch/ssd001/datasets/'
    elif cfg.gen.task in ['sent.fine', 'sent.bin']:
        base_path = '/h/nng/data'
    else:
        raise Exception('task %s data path not found'.format(cfg.data.task))

    data_dir = os.path.join(base_path, cfg.data.task, cfg.data.name, cfg.data.fdset, 'cbert')

    flags = ['--data_dir', data_dir, \
             '--output_dir', data_dir, \
             '--save_model_dir', data_dir, \
             '--task_name', cfg.gen.task, \
             '--num_train_epochs', str(cfg.train.max_epochs), \
             '--seed', str(cfg.train.seed)]
    command = ['python3', '/h/nng/programs/cbert_aug/cbert_finetune.py'] + flags
    print(command)

    log.info(' '.join(command))
    subprocess.call(command)

if __name__ == "__main__":
    launch()

