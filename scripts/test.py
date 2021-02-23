import hydra
import os
from omegaconf import DictConfig, OmegaConf
from hydra import slurm_utils

@hydra.main(config_name='/h/nng/conf/test/config.yaml')
def test(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))

if __name__ == "__main__":
    test()
