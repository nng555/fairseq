import os
import hydra
import subprocess
import logging
from omegaconf import DictConfig
from hydra import slurm_utils

log = logging.getLogger(__name__)

@hydra.main(config_path='/h/nng/conf/robust', config_name='config', strict=False)
def launch(cfg: DictConfig):
    slurm_utils.symlink_hydra(cfg, os.getcwd())

    os.environ['NCCL_DEBUG'] = 'INFO'

    log.info(cfg.command)
    subprocess.run(cfg.command, shell=True)

if __name__ == "__main__":
    launch()

