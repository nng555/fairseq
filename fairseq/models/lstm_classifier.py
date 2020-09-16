# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    BaseFairseqModel,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax
from fairseq.models.lstm import LSTMEncoder
from fairseq.hub_utils import GeneratorHubInterface

@register_model('lstm_classifier')
class LSTMClassifier(BaseFairseqModel):

    def __init__(self, args, encoder):
        super().__init__()
        self.args = args
        self.encoder = encoder
        self.classification_heads = nn.ModuleDict()

    def add_args(parser):
        """Add model-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('--dropout', type=float, metavar='D',
                            help='dropout probability')
        parser.add_argument('--encoder-embed-dim', type=int, metavar='N',
                            help='encoder embedding dimension')
        parser.add_argument('--encoder-embed-path', type=str, metavar='STR',
                            help='path to pre-trained encoder embedding')
        parser.add_argument('--encoder-freeze-embed', action='store_true',
                            help='freeze encoder embeddings')
        parser.add_argument('--encoder-hidden-size', type=int, metavar='N',
                            help='encoder hidden size')
        parser.add_argument('--encoder-layers', type=int, metavar='N',
                            help='number of encoder layers')
        parser.add_argument('--encoder-bidirectional', action='store_true',
                            help='make all layers of encoder bidirectional')

        # Granular dropout settings (if not specified these default to --dropout)
        parser.add_argument('--encoder-dropout-in', type=float, metavar='D',
                            help='dropout probability for encoder input embedding')
        parser.add_argument('--encoder-dropout-out', type=float, metavar='D',
                            help='dropout probability for encoder output')

        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')

    @classmethod
    def build_model(cls, args, task):

        base_architecture(args)

        def load_pretrained_embedding_from_file(embed_path, dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            embed_tokens = Embedding(num_embeddings, embed_dim, padding_idx)
            embed_dict = utils.parse_embedding(embed_path)
            utils.print_embed_overlap(embed_dict, dictionary)
            return utils.load_embedding(embed_dict, dictionary, embed_tokens)

        if args.encoder_embed_path:
            pretrained_encoder_embed = load_pretrained_embedding_from_file(
                args.encoder_embed_path, task.source_dictionary, args.encoder_embed_dim)
        else:
            num_embeddings = len(task.source_dictionary)
            pretrained_encoder_embed = Embedding(
                num_embeddings, args.encoder_embed_dim, task.source_dictionary.pad()
            )


        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample


        encoder = LSTMEncoder(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            hidden_size=args.encoder_hidden_size,
            num_layers=args.encoder_layers,
            dropout_in=args.encoder_dropout_in,
            dropout_out=args.encoder_dropout_out,
            bidirectional=args.encoder_bidirectional,
            pretrained_embed=pretrained_encoder_embed,
        )
        return cls(args, encoder)

    def forward(self, src_tokens, src_lengths, features_only=False, classification_head_name=None, return_all_hiddens=False, *kwargs):
        x = self.encoder(src_tokens, src_lengths)
        encoder_outs, encoder_hiddens, encoder_final = x[:3]
        x = encoder_final[-1] # b x d

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)

        return x, {'inner_states': encoder_hiddens if return_all_hiddens else None}

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions[0]

    def register_classification_head(self, name, num_classes=None, inner_dim=None, **kwargs):
        """Register a classification head."""
        if name in self.classification_heads:
            prev_num_classes = self.classification_heads[name].out_proj.out_features
            prev_inner_dim = self.classification_heads[name].dense.out_features
            if num_classes != prev_num_classes or inner_dim != prev_inner_dim:
                print(
                    'WARNING: re-registering head "{}" with num_classes {} (prev: {}) '
                    'and inner_dim {} (prev: {})'.format(
                        name, num_classes, prev_num_classes, inner_dim, prev_inner_dim
                    )
                )

        if self.args.encoder_bidirectional:
            num_in = 2 * self.args.encoder_hidden_size
        else:
            num_in = self.args.encoder_hidden_size

        self.classification_heads[name] = LSTMClassificationHead(
            num_in,
            inner_dim or self.args.encoder_embed_dim,
            num_classes,
            self.args.pooler_activation_fn,
            self.args.pooler_dropout,
        )

    @classmethod
    def from_pretrained(cls, model_name_or_path, checkpoint_file='model.pt', data_name_or_path='.', bpe='gpt2', **kwargs):
        from fairseq import hub_utils
        x = hub_utils.from_pretrained(
            model_name_or_path,
            checkpoint_file,
            data_name_or_path,
            archive_map=cls.hub_models(),
            bpe=bpe,
            load_checkpoint_heads=True,
            **kwargs,
        )
        return GeneratorHubInterface(x['args'], x['task'], [x['models'][0]])

class LSTMClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, input_dim, inner_dim, num_classes, activation_fn, pooler_dropout):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.activation_fn = utils.get_activation_fn(activation_fn)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, features, **kwargs):
        x = self.dropout(features)
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True, dropout=0):
    """Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

@register_model_architecture('lstm_classifier', 'lstm_classifier')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_freeze_embed = getattr(args, 'encoder_freeze_embed', False)
    args.encoder_hidden_size = getattr(args, 'encoder_hidden_size', args.encoder_embed_dim)
    args.encoder_layers = getattr(args, 'encoder_layers', 2)
    args.encoder_bidirectional = getattr(args, 'encoder_bidirectional', False)
    args.encoder_dropout_in = getattr(args, 'encoder_dropout_in', args.dropout)
    args.encoder_dropout_out = getattr(args, 'encoder_dropout_out', args.dropout)
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
