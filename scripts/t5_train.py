import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

import logging
logging.basicConfig(level=logging.INFO)

tokenizer = T5Tokenizer.from_pretrained('t5-base')

model = T5ForConditionalGeneration.from_pretrained('t5-base')
