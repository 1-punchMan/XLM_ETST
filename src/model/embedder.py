# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import math
import torch
import torch.nn.functional as F
from torch import nn



from .transformer import TransformerModel
from ..data.dictionary import Dictionary, BOS_WORD, EOS_WORD, PAD_WORD, UNK_WORD, MASK_WORD
from ..utils import AttrDict


logger = getLogger()


class SentenceEmbedder(object):

    @staticmethod
    def reload(path, params):
        """
        Create a sentence embedder from a pretrained model.
        """
        # reload model
        reloaded = torch.load(path)
        state_dict = reloaded['model']

        # handle models from multi-GPU checkpoints
        if 'checkpoint' in path:
            state_dict = {(k[7:] if k.startswith('module.') else k): v for k, v in state_dict.items()}

        # reload dictionary and model parameters
        dico = Dictionary(reloaded['dico_id2word'], reloaded['dico_word2id'], reloaded['dico_counts'])
        pretrain_params = AttrDict(reloaded['params'])
        pretrain_params.n_words = len(dico)
        pretrain_params.bos_index = dico.index(BOS_WORD)
        pretrain_params.eos_index = dico.index(EOS_WORD)
        pretrain_params.pad_index = dico.index(PAD_WORD)
        pretrain_params.unk_index = dico.index(UNK_WORD)
        pretrain_params.mask_index = dico.index(MASK_WORD)

        # build model and reload weights
        model = TransformerModel(pretrain_params, dico, True, True)
        model.load_state_dict(state_dict)
        model.eval()

        # adding missing parameters
        params.max_batch_size = 0

        return SentenceEmbedder(model, dico, pretrain_params)

    def __init__(self, model, dico, pretrain_params):
        """
        Wrapper on top of the different sentence embedders.
        Returns sequence-wise or single-vector sentence representations.
        """
        self.pretrain_params = {k: v for k, v in pretrain_params.__dict__.items()}
        self.model = model
        self.dico = dico
        self.n_layers = model.n_layers
        self.out_dim = model.dim
        self.n_words = model.n_words

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def cuda(self):
        self.model.cuda()

    def get_parameters(self, layer_range):

        s = layer_range.split(':')
        assert len(s) == 2
        i, j = int(s[0].replace('_', '-')), int(s[1].replace('_', '-'))

        # negative indexing
        i = self.n_layers + i + 1 if i < 0 else i
        j = self.n_layers + j + 1 if j < 0 else j

        # sanity check
        assert 0 <= i <= self.n_layers
        assert 0 <= j <= self.n_layers

        if i > j:
            return []

        parameters = []

        # embeddings
        if i == 0:
            # embeddings
            parameters += self.model.embeddings.parameters()
            logger.info("Adding embedding parameters to optimizer")
            # positional embeddings
            if self.pretrain_params['sinusoidal_embeddings'] is False:
                parameters += self.model.position_embeddings.parameters()
                logger.info("Adding positional embedding parameters to optimizer")
            # language embeddings
            if hasattr(self.model, 'lang_embeddings'):
                parameters += self.model.lang_embeddings.parameters()
                logger.info("Adding language embedding parameters to optimizer")
            parameters += self.model.layer_norm_emb.parameters()
        # layers
        for l in range(max(i - 1, 0), j):
            parameters += self.model.attentions[l].parameters()
            parameters += self.model.layer_norm1[l].parameters()
            parameters += self.model.ffns[l].parameters()
            parameters += self.model.layer_norm2[l].parameters()
            logger.info("Adding layer-%s parameters to optimizer" % (l + 1))

        logger.info("Optimizing on %i Transformer elements." % sum([p.nelement() for p in parameters]))

        return parameters

    def get_embeddings(self, x, lengths, positions=None, langs=None):
        """
        Inputs:
            `x`        : LongTensor of shape (slen, bs)
            `lengths`  : LongTensor of shape (bs,)
        Outputs:
            `sent_emb` : FloatTensor of shape (bs, out_dim)
        With out_dim == emb_dim
        """
        slen, bs = x.size()
        assert lengths.size(0) == bs and lengths.max().item() == slen

        # get transformer last hidden layer
        tensor = self.model('fwd', x=x, lengths=lengths, positions=positions, langs=langs, causal=False)
        assert tensor.size() == (slen, bs, self.out_dim)

        # single-vector sentence representation (first column of last layer)
        return tensor[0]



class LearningToNorm(nn.Module):
    def __init__(self, dims):
        super().__init__()

        self.dims = dims
        self.gamma = nn.Parameter(torch.ones(1))
        self.mean_weights = nn.Parameter(torch.zeros(dims + 1))
        self.var_weights = nn.Parameter(torch.zeros(dims + 1))
        self.sm = nn.Softmax(dim=-1)
        self.eps = 1e-12

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        # channelwise_norms && individual_norms: x --> (bs, dim, slen)
        # channelwise_norms && not individual_norms: x --> (bs, dim, slen, n_layer)
        # not channelwise_norms && individual_norms: x --> (bs, slen, dim)
        # not channelwise_norms && not individual_norms: x --> (bs, n_layer, slen, dim)
        # mask --> (bs, 1, slen)

        if mask is None:
            mask = x.new_ones(x.shape)
        else:
            while mask.dim() < x.dim():
                mask = mask.unsqueeze(-1)
            mask = mask.expand_as(x).type_as(x)

        mean_weights = self.sm(self.mean_weights)
        var_weights = self.sm(self.var_weights)

        masked_x = (x * mask).contiguous()

        mean = masked_x.new_zeros(masked_x.shape[:-1] + (1,))
        var = masked_x.new_full(masked_x.shape[:-1] + (1,), var_weights[0].item())

        for i in range(1, self.dims + 1):
            shape = masked_x.shape[:-i] + (-1, )
            vw = masked_x.view(shape)
            mask_vw = mask.view(shape)
            num = mask_vw.sum(dim=-1, keepdim=True) + self.eps
            curr_mean = vw.sum(dim=-1, keepdim=True) / num
            diff = (vw - curr_mean)
            diff = diff * mask_vw
            curr_var = (diff.pow(2).sum(dim=-1, keepdim=True) / (num - 1))

            final_shape = masked_x.shape[:-i] + (1,) * i

            mean += mean_weights[i] * curr_mean.view(final_shape)
            var += var_weights[i] * curr_var.view(final_shape)

        return self.gamma * (x - mean) / (var + self.eps).sqrt()


class ElmoTokenEmbedder(nn.Module):
    """
    This is an implementation of the ELMo module which allows learning how to combine hidden states of a language model
    to learn task-specific word representations.
    For more information see the paper here: http://arxiv.org/abs/1802.05365

    This implementation was inspired by the implementation in AllenNLP found here:
    https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md
    """

    def __init__(self, language_model, params):
                #  tune_lm: bool = False,
                #  weights_dropout: float = 0.,
                #  final_dropout: float = 0.,
                #  layer_norm: bool = True,
                #  affine_layer_norm: bool = False,
                #  apply_softmax: bool = True,
                #  channelwise_weights=False,
                #  scaled_sigmoid=False,
                #  individual_norms=False,
                #  channelwise_norm=False,
                #  init_gamma=1.0,
                #  ltn=False,
                #  ltn_dims=None,
                #  train_gamma=True,
                #  ):
        super().__init__()

        self.language_model = language_model
        self.padding_idx = params.pad_index
        self.tune_lm = params.elmo_tune_lm
        self.individual_norms = params.elmo_individual_norms
        self.channelwise_norm = params.elmo_channelwise_norm
        self.scaled_sigmoid = params.elmo_scaled_sigmoid
        self.ltn = None

        if not params.elmo_tune_lm or not language_model.training:
            for param in language_model.parameters():
                param.requires_grad = False
            language_model.eval()

        self.n_layers = language_model.n_layers + 1  # add 1 for token embedding layer
        assert self.n_layers > 0

        self.dim = language_model.dim
        self.embedding_dim = self.dim
        
        print(f'elmo {self.n_layers} x {self.dim}')

        self.weights_dropout = nn.Dropout(params.elmo_weights_dropout)
        self.final_dropout = nn.Dropout(params.elmo_final_dropout)

        self.layer_norm = None
        if params.elmo_layer_norm:
            sz = self.n_layers if self.channelwise_norm and not ltn else self.dim
            if self.individual_norms:
                assert params.elmo_affine_layer_norm
                self.layer_norm = nn.ModuleList(
                    nn.LayerNorm(sz, elementwise_affine=params.elmo_affine_layer_norm) for _ in range(self.n_layers)
                )
            else:
                self.layer_norm = nn.LayerNorm(sz, elementwise_affine=params.elmo_affine_layer_norm)

        self.channelwise_weights = params.elmo_channelwise_weights

        self.weights = None
        self.softmax = None

        if self.channelwise_weights:
            self.weights = nn.Parameter(torch.ones(self.dim, self.n_layers))
        else:
            self.weights = nn.Parameter(torch.Tensor(self.n_layers).fill_(1.0))
        self.softmax = nn.Softmax(dim=-1) if params.elmo_apply_softmax else None

        self.sigmoid_weights = nn.Parameter(torch.zeros(self.n_layers, self.dim)) if params.elmo_scaled_sigmoid else None

        self.gamma = nn.Parameter(torch.full((1,), params.elmo_init_gamma), requires_grad=params.elmo_train_gamma)
        self.projection = nn.Linear(self.dim, self.embedding_dim,
                                    bias=False) if self.embedding_dim != self.dim else None
        if params.elmo_ltn:
            if self.individual_norms:
                ltn_dims = params.elmo_ltn_dims or 3
                assert ltn_dims <= 3
                self.ltn = nn.ModuleList(
                    LearningToNorm(dims=ltn_dims) for _ in range(self.n_layers)
                )
            else:
                ltn_dims = ltn_dims or 4
                assert ltn_dims <= 4
                self.ltn = LearningToNorm(dims=ltn_dims)


    def reset_parameters(self):
        if self.projection:
            nn.init.xavier_uniform_(self.projection.weight)
        if self.softmax is None:
            nn.init.constant_(self.weights, 1 / (self.n_layers * 2))
        if self.scaled_sigmoid:
            nn.init.init.constant_(self.sigmoid_weights, 0)
        # if self.megaproj:
        #     for m in self.megaproj:
        #         if hasattr(m, 'weight'):
        #             nn.init.xavier_uniform_(m.weight)
        #         if hasattr(m, 'bias'):
        #             nn.init.constant_(m.bias, 0)

    def _lm_states(self, x, lengths, causal, src_enc, src_len, positions, langs, cache):
        if self.tune_lm and self.language_model.training:
            _, inner_states = self.language_model('fwd', x=x, lengths=lengths, causal=causal, src_enc=src_enc, src_len=src_len, positions=positions, langs=langs, cache=cache, return_states=True)

        else:
            with torch.no_grad():
                if self.language_model.training:
                    self.language_model.eval()
                _, inner_states = self.language_model('fwd', x=x, lengths=lengths, causal=causal, src_enc=src_enc, src_len=src_len, positions=positions, langs=langs, cache=cache, return_states=True)

        return inner_states

    def forward(self, mode, x, lengths, causal, src_enc=None, src_len=None, positions=None, langs=None, cache=None):

        # x --> (slen, bs)
        # states --> [(bs, slen, dim), ...] * self.n_layers
        states = self._lm_states(x, lengths, causal, src_enc, src_len, positions, langs, cache)
        

        ########## Layer Normalization Strategy ##########
        if self.layer_norm is not None:
            if self.ltn is None and self.channelwise_norm:
                states = torch.stack(states, dim=1).transpose(1, 3)  # (bs, dim, slen, n_layers)
                states = self.layer_norm(states)
                states = states.transpose(1, 3)  # (bs, n_layers, slen, dim)
                states = [_x.squeeze(1) for _x in torch.split(states, 1, dim=1)]  # [(bs, slen, dim), ...] * self.n_layers
            elif self.individual_norms:
                # normalize each layer using different layer_norm
                states = [self.layer_norm[i](states[i]) for i in range(len(states))]
            else:
                states = [self.layer_norm(s) for s in states]

        x = x.transpose(0, 1)  # (bs, slen)
        if self.ltn is not None:
            mask = x.ne(self.padding_idx)  # (bs, slen)

            if self.channelwise_norm:
                mask = mask.unsqueeze(1)  # (bs, 1, slen)
                states = torch.stack(states, dim=1)  # (bs, n_layer, slen, dim)
                if self.individual_norms:
                    states = states.transpose(2, 3)  # (bs, n_layer, dim, slen)
                    for i in range(len(self.ltn)):
                        states[:, i] = self.ltn[i](states[:, i], mask)
                    states = states.transpose(2, 3)
                else:
                    states = states.transpose(1, 3)  # (bs, dim, slen, n_layer)
                    states = self.ltn(states, mask)
                    states = states.transpose(1, 3)
                states = [_x.squeeze(1) for _x in torch.split(states, 1, dim=1)]
            else:
                if self.individual_norms:
                    states = [self.ltn[i](states[i], mask) for i in range(len(states))]
                else:
                    mask = mask.unsqueeze(1)  # (bs, 1, slen)
                    states = torch.stack(states, dim=1)  # (bs, n_layer, slen, dim)
                    states = self.ltn(states, mask)
                    states = [_x.squeeze(1) for _x in torch.split(states, 1, dim=1)]
        #################################################


        if self.softmax is not None:
            w = self.softmax(self.weights)
        else:
            w = self.weights

        if self.channelwise_weights:
            w = w.t()

        w = self.weights_dropout(w)  # (n_layers, dim) if channelwise_weights else (n_layers, )

        y = states[0].new_zeros(x.size() + (self.dim,))  # (bs, slen, dim)
        for i in range(len(states)):
            s = states[i]  # (bs, slen, dim)
            if self.sigmoid_weights is not None:
                sw = F.sigmoid(self.sigmoid_weights[i]) * 2  # (dim, )
                s = s * sw
            y += s * w[i]   # (bs, slen, dim) * `(dim, ) if channelwise_weights else scalar`

        # y = self._without_sentence_boundaries(y)

        if self.projection is not None:
            y = self.projection(y)

        if self.gamma:
            y = self.gamma * y

        y = self.final_dropout(y)
        y = y.transpose(0, 1)  # (slen, bs, dim)
        return y
