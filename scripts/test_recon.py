from fairseq.models.roberta import RobertaModel
import torch
r_model = RobertaModel.from_pretrained(
        '/scratch/hdd001/home/nng/roberta/roberta.base',
        checkpoint_file='model.pt',
        data_name_or_path='/scratch/hdd001/home/nng/roberta/roberta.base')
ex0 = r_model.masked_encode('This is a test of the masking and reconstruction', mask_prob=0.5, random_token_prob=0.4, leave_unmasked_prob=0.0)
ex1 = r_model.masked_encode('This is a test of the masking and reconstruction', mask_prob=0.5, random_token_prob=0.4, leave_unmasked_prob=0.0)
toks = torch.stack([ex0[0], ex1[0]])
masks = torch.stack([ex0[1], ex1[1]])
recon = r_model.reconstruction_prob_tok(toks, masks, reconstruct=True, topk=-1)
