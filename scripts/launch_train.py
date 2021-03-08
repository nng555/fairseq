import os
import hydra
import subprocess
import logging
from omegaconf import DictConfig, OmegaConf
from hydra import slurm_utils

log = logging.getLogger(__name__)

@hydra.main(config_path='/h/nng/conf/selftrain/config.yaml')
def launch(cfg: DictConfig):
    #print(OmegaConf.to_yaml(cfg))
    slurm_utils.symlink_hydra(cfg, os.getcwd())

    os.environ['NCCL_DEBUG'] = 'INFO'

    if cfg.data.task in ['nli']:
        base_path = '/scratch/ssd001/datasets/'
    elif cfg.data.task in ['sentiment', 'translation', 'robust']:
        base_path = '/h/nng/data'
    else:
        raise Exception('task %s data path not found'.format(cfg.data.task))

    d_path = os.path.join(base_path, cfg.data.task, cfg.data.name)
    bin_path = os.path.join(base_path, cfg.data.task, cfg.data.name, cfg.data.fdset, cfg.data.bin.name, 'bin')
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

        if not found and 'roberta' in cfg.train.arch:
            cfg.train.restore_file = '/scratch/hdd001/home/nng/roberta/roberta.base/model.pt'
            cfg.train.reset_optimizer = True
            cfg.train.reset_dataloader = True
            cfg.train.reset_meters = True

    if cfg.train.augment and cfg.gen.recon == 'local' and 'self_train' not in cfg.data.bin:
        print("FINDING LOCAL PATH", flush=True)
        r_path = os.path.join('/h/nng/slurm', cfg.gen.recon_file.date, slurm_utils.resolve_name(cfg.gen.recon_file.name))
        if os.path.exists(os.path.join(r_path, 'checkpoint_best.pt')):
            r_file = r_path
        else:
            found = False
            for f in sorted(os.listdir(r_path))[::-1]:
                if os.path.exists(os.path.join(r_path, f, 'checkpoint_best.pt')):
                    r_file = os.path.join(r_path, f)
                    found = True
                    break
            if not found:
                raise Exception("Model in path {} not found".format(r_path))

        cfg.train.recon_model_path = r_file

    flags = [['--' + k.replace('_', '-'), slurm_utils.eval_val(str(v))] for k, v in cfg.train.items() if v != None]
    flags = [val for sublist in flags if sublist[1] != "False" for val in sublist if val != "True"]
    flags = flags + ['--save-dir', os.path.join(j_dir, os.environ['SLURM_JOB_ID'])]

    command = ' '.join(['fairseq-train', bin_path] + flags)
    log.info(command)
    subprocess.call(['fairseq-train', bin_path] + flags)

if __name__ == "__main__":
    launch()
