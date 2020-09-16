movies=sst2,imdb
yelp=american,chinese,italian,japanese
aws=books,clothing,home,kindle,movies,pets,sports,tech,tools,toys
caws=men,women,baby,shoes

# aws
#python3 launch_train.py model=rnn data=aws aws=uda train.seed=$(seq -s, 0 1 9) data.fdset=$aws,$caws data.tdset=null -m
#python3 launch_train.py model=cnn data=aws aws=eda,uda train.seed=$(seq -s, 0 1 9) data.fdset=$aws,$caws data.tdset=null -m
python3 launch_train.py model=rnn,cnn data=aws aws=cbert train.seed=$(seq -s, 0 1 9) data.fdset=$aws,$caws data.tdset=null -m

# yelp
#python3 launch_train.py model=rnn data=yelp aws=sampling_noise_imp gen.mask_prob=0.3,0.4,0.5,0.6 train.seed=$(seq -s, 0 1 9) data.fdset=$yelp data.tdset=null -m
#python3 launch_train.py model=cnn data=yelp aws=sampling_noise_soft_conv gen.mask_prob=0.3,0.4,0.5,0.6 train.seed=$(seq -s, 0 1 9) data.fdset=$yelp data.tdset=null -m
#python3 launch_train.py model=rnn,cnn data=yelp aws=eda,uda train.seed=$(seq -s, 0 1 9) data.fdset=$yelp data.tdset=null -m
python3 launch_train.py model=rnn,cnn data=yelp aws=cbert train.seed=$(seq -s, 0 1 9) data.fdset=$yelp data.tdset=null -m

# movies
#python3 launch_train.py model=rnn data=movies movies=sampling_noise_soft,sampling_noise_hard gen.mask_prob=0.1,0.2,0.3,0.4,0.5,0.6 train.seed=$(seq -s, 0 1 9) data.fdset=sst2,imdb data.tdset=null -m
#python3 launch_train.py model=cnn data=movies movies=sampling_noise_soft_conv,sampling_noise_hard_conv gen.mask_prob=0.1,0.2,0.3,0.4,0.5,0.6 train.seed=$(seq -s, 0 1 9) data.fdset=sst2,imdb data.tdset=null -m
#python3 launch_train.py model=rnn,cnn data=movies movies=sampling_noise_imp gen.mask_prob=0.2,0.3 train.seed=$(seq -s, 0 1 9) data.fdset=sst2,imdb data.tdset=null -m
#python3 launch_train.py model=rnn,cnn data=movies movies=eda,uda train.seed=$(seq -s, 0 1 9) data.fdset=$movies data.tdset=null -m
python3 launch_train.py model=rnn,cnn data=movies movies=cbert train.seed=$(seq -s, 0 1 9) data.fdset=$movies data.tdset=null -m

#python3 launch_train.py model=roberta data=anli anli=eda train.seed=$(seq -s, 0 1 4) data.fdset=R1 data.tdset=null data.num_sents=110000 slurm.gres=gpu:4 train.update_freq=2 -m
#python3 launch_train.py model=roberta data=anli anli=uda train.seed=$(seq -s, 0 1 4) data.fdset=R1 data.tdset=null data.num_sents=34000 slurm.gres=gpu:4 train.update_freq=2 -m
#python3 launch_train.py model=roberta data=anli anli=eda train.seed=$(seq -s, 0 1 4) data.fdset=R2 data.tdset=null data.num_sents=280000 slurm.gres=gpu:4 train.update_freq=2 -m
#python3 launch_train.py model=roberta data=anli anli=uda train.seed=$(seq -s, 0 1 4) data.fdset=R2 data.tdset=null data.num_sents=91000 slurm.gres=gpu:4 train.update_freq=2 -m
#python3 launch_train.py model=roberta data=anli anli=eda train.seed=$(seq -s, 0 1 4) data.fdset=R3 data.tdset=null data.num_sents=610000 slurm.gres=gpu:4 train.update_freq=2 -m
#python3 launch_train.py model=roberta data=anli anli=uda train.seed=$(seq -s, 0 1 4) data.fdset=R3 data.tdset=null data.num_sents=210000 slurm.gres=gpu:4 train.update_freq=2 -m
#python3 launch_train.py model=roberta data=mnli mnli=eda train.seed=$(seq -s, 0 1 4) data.fdset=fiction,government,travel,telephone,slate data.tdset=null slurm.gres=gpu:4 train.update_freq=2 -m
python3 launch_train.py model=roberta data=mnli mnli=cbert train.seed=$(seq -s, 0 1 4) data.fdset=fiction,government,travel,telephone,slate data.tdset=null slurm.gres=gpu:4 train.update_freq=2 -m
python3 launch_train.py model=roberta data=anli anli=cbert train.seed=$(seq -s, 0 1 4) data.fdset=R1 data.tdset=null data.num_sents=34000 slurm.gres=gpu:4 train.update_freq=2 -m
python3 launch_train.py model=roberta data=anli anli=cbert train.seed=$(seq -s, 0 1 4) data.fdset=R2 data.tdset=null data.num_sents=92000 slurm.gres=gpu:4 train.update_freq=2 -m
python3 launch_train.py model=roberta data=anli anli=cbert train.seed=$(seq -s, 0 1 4) data.fdset=R3 data.tdset=null data.num_sents=200000 slurm.gres=gpu:4 train.update_freq=2 -m

#python3 launch_train.py model=rnn,cnn data=movies movies=orig train.seed=$(seq -s, 0 1 9) data.fdset=sst2,imdb data.tdset=null -m
#python3 launch_train.py model=rnn,cnn data=yelp aws=orig train.seed=$(seq -s, 0 1 9) data.fdset=american,chinese,italian,japanese data.tdset=null -m
