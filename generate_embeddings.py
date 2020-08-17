# Copyright (c) 2019-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from scipy import spatial
import json
import random
import argparse
import numpy as np

from src.slurm import init_signal_handler, init_distributed_mode
from src.data.loader import check_data_params, load_data, my_load_data
from src.utils import bool_flag, initialize_exp, set_sampling_probs, shuf_order
from src.model import check_model_params, build_model, build_seq2seq_model, build_clts_xencoder_model, build_clts_elmo_model
from src.model.memory import HashingMemory
from src.trainer import SingleTrainer, EncDecTrainer, MyEncDecTrainer, XLMCLTSEncDecTrainer
from src.evaluation.evaluator import SingleEvaluator, EncDecEvaluator, MyEncDecEvaluator, XLMCLTSEncDecEvaluator

import warnings
warnings.filterwarnings("ignore")


def get_parser():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser(description="Language transfer")

    # main parameters
    parser.add_argument("--dump_path", type=str, default="./dumped/",
                        help="Experiment dump path")
    parser.add_argument("--exp_name", type=str, default="",
                        help="Experiment name")
    parser.add_argument("--save_periodic", type=int, default=0,
                        help="Save the model periodically (0 to disable)")
    parser.add_argument("--exp_id", type=str, default="",
                        help="Experiment ID")

    # float16 / AMP API
    parser.add_argument("--fp16", type=bool_flag, default=False,
                        help="Run model with float16")
    parser.add_argument("--amp", type=int, default=-1,
                        help="Use AMP wrapper for float16 / distributed / gradient accumulation. Level of optimization. -1 to disable.")

    # only use an encoder (use a specific decoder for machine translation)
    parser.add_argument("--encoder_only", type=bool_flag, default=True,
                        help="Only use an encoder")

    # model parameters
    parser.add_argument("--emb_dim", type=int, default=512,
                        help="Embedding layer size")
    parser.add_argument("--n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--share_inout_emb", type=bool_flag, default=True,
                        help="Share input and output embeddings")
    parser.add_argument("--sinusoidal_embeddings", type=bool_flag, default=False,
                        help="Use sinusoidal embeddings")
    parser.add_argument("--use_lang_emb", type=bool_flag, default=True,
                        help="Use language embedding")

    # memory parameters
    parser.add_argument("--use_memory", type=bool_flag, default=False,
                        help="Use an external memory")
    if parser.parse_known_args()[0].use_memory:
        HashingMemory.register_args(parser)
        parser.add_argument("--mem_enc_positions", type=str, default="",
                            help="Memory positions in the encoder ('4' for inside layer 4, '7,10+' for inside layer 7 and after layer 10)")
        parser.add_argument("--mem_dec_positions", type=str, default="",
                            help="Memory positions in the decoder. Same syntax as `mem_enc_positions`.")

    # adaptive softmax
    parser.add_argument("--asm", type=bool_flag, default=False,
                        help="Use adaptive softmax")
    if parser.parse_known_args()[0].asm:
        parser.add_argument("--asm_cutoffs", type=str, default="8000,20000",
                            help="Adaptive softmax cutoffs")
        parser.add_argument("--asm_div_value", type=float, default=4,
                            help="Adaptive softmax cluster sizes ratio")

    # causal language modeling task parameters
    parser.add_argument("--context_size", type=int, default=0,
                        help="Context size (0 means that the first elements in sequences won't have any context)")

    # masked language modeling task parameters
    parser.add_argument("--word_pred", type=float, default=0.15,
                        help="Fraction of words for which we need to make a prediction")
    parser.add_argument("--sample_alpha", type=float, default=0,
                        help="Exponent for transforming word counts to probabilities (~word2vec sampling)")
    parser.add_argument("--word_mask_keep_rand", type=str, default="0.8,0.1,0.1",
                        help="Fraction of words to mask out / keep / randomize, among the words to predict")

    # input sentence noise
    parser.add_argument("--word_shuffle", type=float, default=0,
                        help="Randomly shuffle input words (0 to disable)")
    parser.add_argument("--word_dropout", type=float, default=0,
                        help="Randomly dropout input words (0 to disable)")
    parser.add_argument("--word_blank", type=float, default=0,
                        help="Randomly blank input words (0 to disable)")

    # data
    parser.add_argument("--data_path", type=str, default="",
                        help="Data path")
    parser.add_argument("--lgs", type=str, default="",
                        help="Languages (lg1-lg2-lg3 .. ex: en-fr-es-de)")
    parser.add_argument("--max_vocab", type=int, default=-1,
                        help="Maximum vocabulary size (-1 to disable)")
    parser.add_argument("--min_count", type=int, default=0,
                        help="Minimum vocabulary count")
    parser.add_argument("--lg_sampling_factor", type=float, default=-1,
                        help="Language sampling factor")

    # batch parameters
    parser.add_argument("--bptt", type=int, default=256,
                        help="Sequence length")
    parser.add_argument("--max_len", type=int, default=100,
                        help="Maximum length of sentences (after BPE)")
    parser.add_argument("--group_by_size", type=bool_flag, default=True,
                        help="Sort sentences by size during the training")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Number of sentences per batch")
    parser.add_argument("--max_batch_size", type=int, default=0,
                        help="Maximum number of sentences per batch (used in combination with tokens_per_batch, 0 to disable)")
    parser.add_argument("--tokens_per_batch", type=int, default=-1,
                        help="Number of tokens per batch")

    # training parameters
    parser.add_argument("--split_data", type=bool_flag, default=False,
                        help="Split data across workers of a same node")
    parser.add_argument("--optimizer", type=str, default="adam,lr=0.0001",
                        help="Optimizer (SGD / RMSprop / Adam, etc.)")
    parser.add_argument("--clip_grad_norm", type=float, default=5,
                        help="Clip gradients norm (0 to disable)")
    parser.add_argument("--epoch_size", type=int, default=100000,
                        help="Epoch size / evaluation frequency (-1 for parallel data size)")
    parser.add_argument("--max_epoch", type=int, default=100000,
                        help="Maximum epoch size")
    parser.add_argument("--stopping_criterion", type=str, default="",
                        help="Stopping criterion, and number of non-increase before stopping the experiment")
    parser.add_argument("--validation_metrics", type=str, default="",
                        help="Validation metrics")
    parser.add_argument("--accumulate_gradients", type=int, default=1,
                        help="Accumulate model gradients over N iterations (N times larger batch sizes)")

    # training coefficients
    parser.add_argument("--lambda_mlm", type=str, default="1",
                        help="Prediction coefficient (MLM)")
    parser.add_argument("--lambda_clm", type=str, default="1",
                        help="Causal coefficient (LM)")
    parser.add_argument("--lambda_pc", type=str, default="1",
                        help="PC coefficient")
    parser.add_argument("--lambda_ae", type=str, default="1",
                        help="AE coefficient")
    parser.add_argument("--lambda_mt", type=str, default="1",
                        help="MT coefficient")
    parser.add_argument("--lambda_bt", type=str, default="1",
                        help="BT coefficient")

    # training steps
    parser.add_argument("--clm_steps", type=str, default="",
                        help="Causal prediction steps (CLM)")
    parser.add_argument("--mlm_steps", type=str, default="",
                        help="Masked prediction steps (MLM / TLM)")
    parser.add_argument("--mt_steps", type=str, default="",
                        help="Machine translation steps")
    parser.add_argument("--ae_steps", type=str, default="",
                        help="Denoising auto-encoder steps")
    parser.add_argument("--bt_steps", type=str, default="",
                        help="Back-translation steps")
    parser.add_argument("--pc_steps", type=str, default="",
                        help="Parallel classification steps")
    parser.add_argument("--unsclts_steps", type=str, default="",
                        help="'en-EN', 'zh'-'ZH'")

    # reload pretrained embeddings / pretrained model / checkpoint
    parser.add_argument("--reload_emb", type=str, default="",
                        help="Reload pretrained word embeddings")
    parser.add_argument("--reload_model", type=str, default="",
                        help="Reload a pretrained model")
    parser.add_argument("--reload_checkpoint", type=str, default="",
                        help="Reload a checkpoint")

    # beam search (for MT only)
    parser.add_argument("--beam_size", type=int, default=1,
                        help="Beam size, default = 1 (greedy decoding)")
    parser.add_argument("--length_penalty", type=float, default=1,
                        help="Length penalty, values < 1.0 favor shorter sentences, while values > 1.0 favor longer ones.")
    parser.add_argument("--early_stopping", type=bool_flag, default=False,
                        help="Early stopping, stop as soon as we have `beam_size` hypotheses, although longer ones may have better scores.")

    # evaluation
    parser.add_argument("--eval_bleu", type=bool_flag, default=False,
                        help="Evaluate BLEU score during MT training")
    parser.add_argument("--eval_only", type=bool_flag, default=False,
                        help="Only run evaluations")

    # debug
    parser.add_argument("--debug_train", type=bool_flag, default=False,
                        help="Use valid sets for train sets (faster loading)")
    parser.add_argument("--debug_slurm", type=bool_flag, default=False,
                        help="Debug multi-GPU / multi-node within a SLURM job")
    parser.add_argument("--debug", help="Enable all debug flags",
                        action="store_true")

    # multi-gpu / multi-node
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="Multi-GPU - Local rank")
    parser.add_argument("--master_port", type=int, default=-1,
                        help="Master port (for multi-node SLURM jobs)")

    ########## added by chiamin ##########

    # general
    parser.add_argument("--share_encdec_emb", type=bool_flag,
                        default=False, help="Share encoder and decoder word embeddings")
    parser.add_argument("--eval_rouge", type=bool_flag, default=False,
                        help="Evaluate ROUGE-1 F1 score during TS training")
    parser.add_argument("--label_smoothing", type=float,
                        default=0., help="Label smoothing loss (0 to disable)")

    # separated word embedding
    parser.add_argument("--src_max_vocab", type=int, default=-1,
                        help="Maximum source vocabulary size (-1 to disable)")
    parser.add_argument("--tgt_max_vocab", type=int, default=-1,
                        help="Maximum target vocabulary size (-1 to disable)")
    parser.add_argument("--src_min_count", type=int, default=0,
                        help="Minimum source vocabulary count")
    parser.add_argument("--tgt_min_count", type=int, default=0,
                        help="Minimum target vocabulary count")

    # clts-xencoder
    parser.add_argument("--use_xencoder", type=bool_flag,
                        default=False, help="use cross-lingual encoder")
    parser.add_argument("--reload_xencoder", type=str, default="",
                        help="Reload pretrained xlm (cross-lingual encoder). Used in clts-xencoder")
    parser.add_argument("--ts_emb_dim", type=int, default=512,
                        help="text summarization embedding layer size")
    parser.add_argument("--ts_n_layers", type=int, default=4,
                        help="Number of Transformer layers")
    parser.add_argument("--ts_n_heads", type=int, default=8,
                        help="Number of Transformer heads")
    parser.add_argument("--ts_dropout", type=float, default=0,
                        help="Dropout")
    parser.add_argument("--ts_attention_dropout", type=float, default=0,
                        help="Dropout in the attention layer")
    parser.add_argument("--ts_gelu_activation", type=bool_flag, default=False,
                        help="Use a GELU activation instead of ReLU")
    parser.add_argument("--xencoder_optimizer", type=str, default="adam,lr=0.0001",
                        help="Cross-lingual Optimizer (SGD / RMSprop / Adam, etc.)")

    # clts-elmo
    parser.add_argument("--reload_elmo", type=str, default="",
                        help="Reload pretrained elmo. Used in clts-elmo evaluation")
    parser.add_argument("--elmo_tune_lm", type=bool_flag,
                        default=True, help="")
    parser.add_argument("--elmo_weights_dropout",
                        type=float, default=0.0, help="")
    parser.add_argument("--elmo_final_dropout",
                        type=float, default=0.0, help="")
    parser.add_argument("--elmo_layer_norm",
                        type=bool_flag, default=True, help="")
    parser.add_argument("--elmo_affine_layer_norm",
                        type=bool_flag, default=False, help="")
    parser.add_argument("--elmo_apply_softmax",
                        type=bool_flag, default=True, help="")
    parser.add_argument("--elmo_channelwise_weights",
                        type=bool_flag, default=False, help="")
    parser.add_argument("--elmo_scaled_sigmoid",
                        type=bool_flag, default=False, help="")
    parser.add_argument("--elmo_individual_norms",
                        type=bool_flag, default=False, help="")
    parser.add_argument("--elmo_channelwise_norm",
                        type=bool_flag, default=False, help="")
    parser.add_argument("--elmo_init_gamma", type=float, default=1.0, help="")
    parser.add_argument("--elmo_ltn", type=bool_flag, default=False, help="")
    parser.add_argument("--elmo_ltn_dims", type=str, default="", help="")
    parser.add_argument("--elmo_train_gamma",
                        type=bool_flag, default=True, help="")

    # generate embeddings
    parser.add_argument("--gen_word_emb", type=bool_flag, default=False)
    parser.add_argument("--gen_sent_emb", type=bool_flag, default=False)

    ######################################

    return parser


def most_similiar(target, embs, k=10):
    def consine_similarity(v1, v2):
        return 1 - spatial.distance.cosine(v1, v2)
        # return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

    similarity = {}
    for i, emb in enumerate(embs):
        if i % 10000 == 0:
            print(f'Processed {i} tokens')

        if (emb == target).all():
            continue

        similarity.update({i: consine_similarity(target, emb)})

    topk = sorted(similarity.items(), key=lambda x: x[1], reverse=True)[: k]

    return topk


def load_embeddings(path):

    print(f'Loading word embedding from {path}')
    tokens = []
    embs = []
    with open(path, 'r', encoding='utf-8') as infile:
        for line in infile:
            line = line.strip().split('\t')
            tokens.append(line[0])
            embs.append([float(v) for v in line[1:]])

    print('DONE')
    return np.array(embs), tokens


# def convert_to_text(batch, lengths, dico, params):
#     """
#     Convert a batch of sentences to a list of text sentences.
#     """


#     sentences = []
#     words = []
#     for k in range(1, lengths[j]):
#         if batch[k, j] == params.eos_index:
#             break
#         words.append(dico[batch[k, j]])
#     sentences.append(" ".join(words))
#     return sentences


def main(params):

    init_distributed_mode(params)

    # load data
    data = load_data(params)

    emb_weights = None
    # build model
    if params.encoder_only:
        model = build_model(params, data['dico'])
        emb_weights = model.embeddings.weight.data.cpu().numpy()
    else:
        encoder, decoder = build_model(params, data['dico'])
        emb_weights = encoder.embeddings.weight.data.cpu().numpy()

    if params.gen_word_emb:

        # metadata = open(
        #     "./pretrained_models/mlm_xnli15_1024/xlm15-metadata.txt", "w", encoding='utf-8')
        # embeddings = open(
        #     "./pretrained_models/mlm_xnli15_1024/embeddings.tsv", "w", encoding='utf-8')

        # with open("./pretrained_models/mlm_xnli15_1024/token_embeddings.tsv", "w", encoding='utf-8') as out:
        with open(f'./dumped/{params.exp_name}/{params.exp_id}/embeddings.tsv', 'w', encoding='utf-8') as out:
            for i in range(len(data['dico'])):
                word = data['dico'][i]
                emb = '\t'.join([str(v) for v in emb_weights[i]])

                out.write(f"{word}\t{emb}\n")
        #         metadata.write(f"{word}\n")
        #         embeddings(f"{emb}\n")

        # metadata.close()
        # embeddings.close()

    if params.gen_sent_emb:
        evaluator = SingleEvaluator(None, data, params)

        sents = {}
        sent_embs = {}
        with torch.no_grad():
            data_set = 'test'
            for lang1, lang2 in params.mlm_steps:
                # lang2 is None
                _sents, _sent_embs = evaluator.generate_sent_emb(
                    data_set, lang1, lang2)
                sents[lang1] = _sents
                sent_embs[lang1] = _sent_embs

        for lang1, lang2 in params.mlm_steps:
            out = open(
                f'./dumped/{params.exp_name}/{params.exp_id}/sent_embs-{lang1}.tsv', 'w', encoding='utf-8')
            for sent, emb in zip(sents[lang1], sent_embs[lang1]):
                emb = '\t'.join([str(v) for v in emb])
                out.write(f"{sent.strip()}\t{emb}\n")

            out.close()


def clts_xencoder_main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # load data
    data = load_data(params)

    # cross lingual  text summarization encoder, text summarization decoder
    xencoder, ts_encoder, ts_decoder = build_clts_xencoder_model(
        params, data['dico'])
    emb_weights = xencoder.embeddings.weight.data.cpu().numpy()

    # with open(f'./dumped/{params.exp_name}/{params.exp_id}/embeddings.tsv', 'w', encoding='utf-8') as out:
    metadata = open(
        "./pretrained_models/mlm_xnli15_1024/xlm15-metadata.txt", "w", encoding='utf-8')
    embeddings = open(
        "./pretrained_models/mlm_xnli15_1024/embeddings.tsv", "w", encoding='utf-8')
    with open("./pretrained_models/mlm_xnli15_1024/token_embeddings.tsv", "w", encoding='utf-8') as out:
        for i in range(len(data['dico'])):
            word = data['dico'][i]
            emb = '\t'.join([str(v) for v in emb_weights[i]])

            out.write(f"{word}\t{emb}\n")
            metadata.write(f"{word}\n")
            embeddings.write(f"{emb}\n")

    metadata.close()
    embeddings.close()


def clts_elmo_main(params):

    # initialize the multi-GPU / multi-node training
    init_distributed_mode(params)

    # load data
    data = load_data(params)

    # cross lingual  text summarization encoder, text summarization decoder
    elmo, ts_encoder, ts_decoder = build_clts_elmo_model(params, data['dico'])
    emb_weights = elmo.language_model.embeddings.weight.data.cpu().numpy()
    with open(f'./dumped/{params.exp_name}/{params.exp_id}/embeddings.tsv', 'w', encoding='utf-8') as out:
        for i in range(len(data['dico'])):
            word = data['dico'][i]
            emb = '\t'.join([str(v) for v in emb_weights[i]])

            out.write(f"{word}\t{emb}\n")


if __name__ == '__main__':

    # generate parser / parse parameters
    parser = get_parser()
    params = parser.parse_args()

    # debug mode
    if params.debug:
        params.exp_name = 'debug'
        params.exp_id = 'debug_%08i' % random.randint(0, 100000000)
        params.debug_slurm = True
        params.debug_train = True

    # check parameters
    check_data_params(params)
    check_model_params(params)

    # run experiment
    # main(params)
    # clts_xencoder_main(params)
    # clts_elmo_main(params)

    # embs, tokens = load_embeddings(
    #     f'./dumped/{params.exp_name}/{params.exp_id}/embeddings.tsv')

    embs, tokens = load_embeddings(
        './pretrained_models/mlm_xnli15_1024/xlm15-token_embeddings.tsv')

    target_token = 'development'
    # target_token = 'president'
    target = embs[tokens.index(target_token)]
    topk = most_similiar(target, embs, 60)

    print()
    print(target_token)
    # print(topk)
    for idx, sim in topk:
        print(idx, f'`{tokens[idx]}`', sim)

    ########## TensorBoard ##########
    # with open('embeddings.tsv', 'w', encoding='utf-8') as out, open('metadata.txt', 'w', encoding='utf-8') as meta:
    #     meta.write('token\n')
    #     for token, emb in zip(tokens, embs):
    #         out.write('\t'.join([str(e) for e in emb]) + '\n')
    #         meta.write(token + '\n')
