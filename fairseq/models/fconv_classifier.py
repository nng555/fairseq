# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from fairseq import options, utils
from fairseq.models import (
    BaseFairseqModel,
    FairseqEncoder,
    register_model,
    register_model_architecture,
)
from fairseq.modules import AdaptiveSoftmax
from fairseq.models.fconv import FConvEncoder

@register_model('fconv_classifier')
class FConvClassifier(BaseFairseqModel):

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
        parser.add_argument('--encoder-layers', type=str, metavar='EXPR',
                            help='encoder layers [(dim, kernel_size), ...]')

        parser.add_argument('--pooler-dropout', type=float, metavar='D',
                            help='dropout probability in the masked_lm pooler layers')
        parser.add_argument('--pooler-activation-fn',
                            choices=utils.get_available_activation_fns(),
                            help='activation function to use for pooler layer')

    @classmethod
    def build_model(cls, args, task):

        base_architecture(args)

        encoder_embed_dict = None
        if args.encoder_embed_path:
            encoder_embed_dict = utils.parse_embedding(args.encoder_embed_path)
            utils.print_embed_overlap(encoder_embed_dict, task.source_dictionary)

        encoder = ConvClassifier(
            dictionary=task.source_dictionary,
            embed_dim=args.encoder_embed_dim,
            embed_dict=encoder_embed_dict,
            convolutions=eval(args.encoder_layers),
            dropout=args.dropout,
            max_positions=args.max_source_positions,
        )

        if not hasattr(args, 'max_positions'):
            args.max_positions = args.tokens_per_sample

        return cls(args, encoder)

    def max_positions(self):
        """Maximum output length supported by the encoder."""
        return self.args.max_positions[0]

    def forward(self, src_tokens, src_lengths, features_only=False, classification_head_name=None, **kwargs):
        x = self.encoder(src_tokens, src_lengths)
        x = x['encoder_out']

        # max pool over time, transpose to B x C x T -> B x C x 1
        #x = F.max_pool1d(encoder_out.transpose(1, 2), kernel_size=encoder_out.size()[1])
        #x = x.squeeze(-1)

        if classification_head_name is not None:
            x = self.classification_heads[classification_head_name](x)
        return x, None

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

        out_dim = sum([val for (val, _) in eval(self.args.encoder_layers)])
        self.classification_heads[name] = FConvClassificationHead(
            out_dim,
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
        return hub_utils.GeneratorHubInterface(x['args'], x['task'], [x['models'][0]])

class ConvClassifier(FairseqEncoder):

    def __init__(
        self, dictionary, embed_dim=512, embed_dict=None, max_positions=1024,
        convolutions=((256, 3), (256, 4), (256, 5)), dropout=0.5,
    ):
        print(convolutions, flush=True)
        super().__init__(dictionary)
        self.dropout = dropout

        num_embeddings = len(dictionary)
        self.padding_idx = dictionary.pad()
        self.embed_tokens = Embedding(num_embeddings, embed_dim, self.padding_idx)
        if embed_dict:
            self.embed_tokens = utils.load_embedding(embed_dict, self.dictionary, self.embed_tokens)

        self.max_positions = max_positions
        self.convolutions = nn.ModuleList()
        for conv in convolutions:
            if conv[1] % 2 == 1:
                padding = conv[1] // 2
            else:
                padding = 0
            self.convolutions.append(
                ConvTBC(embed_dim, conv[0], conv[1],
                        dropout=dropout, padding=padding),
            )

    def forward(self, src_tokens, src_lengths):
        x = self.embed_tokens(src_tokens)

        # used to mask padding in input
        encoder_padding_mask = src_tokens.eq(self.padding_idx).t()  # -> T x B
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        outs = []
        for conv in self.convolutions:
            x_in = F.dropout(x , p=self.dropout, training=self.training)
            if encoder_padding_mask is not None:
                x_in = x_in.masked_fill(encoder_padding_mask.unsqueeze(-1), 0)

            if conv.kernel_size[0] % 2 == 1:
                # padding is implicit in the conv
                x_in = conv(x_in)
            else:
                padding_l = (conv.kernel_size[0] - 1) // 2
                padding_r = conv.kernel_size[0] // 2
                x_in = F.pad(x_in, (0, 0, 0, 0, padding_l, padding_r))
                x_in = conv(x_in)

            x_in = F.relu(x_in)
            T, B, C = x_in.size()
            #x_in = x_in.reshape(B, C, T) # B x C x T
            x_in = x_in.permute(1, 2, 0) # T x B x C -> B x C x T
            x_in = F.max_pool1d(x_in, T).squeeze(-1)

            outs.append(x_in)

        out = torch.cat(outs, dim=-1) # B x C

        return {
            'encoder_out': out
        }


class FConvClassificationHead(nn.Module):
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
    nn.init.normal_(m.weight, 0, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight, mean=0, std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)

def ConvTBC(in_channels, out_channels, kernel_size, dropout=0, **kwargs):
    """Weight-normalized Conv1d layer"""
    from fairseq.modules import ConvTBC
    m = ConvTBC(in_channels, out_channels, kernel_size, **kwargs)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)

@register_model_architecture('fconv_classifier', 'fconv_classifier')
def base_architecture(args):
    args.dropout = getattr(args, 'dropout', 0.1)
    args.encoder_embed_dim = getattr(args, 'encoder_embed_dim', 512)
    args.encoder_embed_path = getattr(args, 'encoder_embed_path', None)
    args.encoder_layers = getattr(args, 'encoder_layers', '[(256, 3), (256, 4), (256, 5)]')
    args.pooler_activation_fn = getattr(args, 'pooler_activation_fn', 'tanh')
    args.pooler_dropout = getattr(args, 'pooler_dropout', 0.0)
