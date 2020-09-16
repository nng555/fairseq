import hydra
import os
from omegaconf import DictConfig
from hydra import slurm_utils

@hydra.main(config_path='/h/nng/conf/robust/config.yaml')
def test(cfg: DictConfig):
    slurm_utils.symlink_hydra(cfg, os.getcwd())
    print(cfg.pretty())

if __name__ == "__main__":
    test()
