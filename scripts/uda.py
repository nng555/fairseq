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

    if cfg.data.task in ['nli']:
        base_path = '/scratch/ssd001/datasets/'
    elif cfg.data.task in ['sentiment']:
        base_path = '/h/nng/data'
    else:
        raise Exception('task %s data path not found'.format(cfg.data.task))

    data_dir = os.path.join(base_path, cfg.data.task, cfg.data.name, cfg.data.fdset)

    flags = [data_dir, str(cfg.gen.num_shards), str(cfg.gen.shard), str(cfg.gen.sampling_temp), cfg.gen.fname]
    command = ['bash', 'run.sh'] + flags

    os.chdir('/h/nng/programs/uda/back_translate')
    log.info(' '.join(command))
    subprocess.call(command)

if __name__ == "__main__":
    launch()
