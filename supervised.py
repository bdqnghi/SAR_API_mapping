import os
import json
import argparse
from collections import OrderedDict
import torch

from src.utils import bool_flag, initialize_exp
from src.models import build_model
from src.trainer import Trainer
from src.evaluation import Evaluator
from src.utils import compute_candidates_for_method_similarity

# VALIDATION_METRIC = 'precision_at_1-nn'
VALIDATION_METRIC = 'precision_at_1-csls_knn_10'
# unsupervised criterion: 'mean_cosine-csls_knn_10-S2T-10000'
#   supervised criterion: 'precision_at_1-csls_knn_10'

# main
parser = argparse.ArgumentParser(description='Supervised training')
parser.add_argument("--seed", type=int, default=-1, help="Initialization seed")
parser.add_argument("--verbose", type=int, default=2, help="Verbose level (2:debug, 1:info, 0:warning)")
parser.add_argument("--exp_path", type=str, default="", help="Where to store experiment logs and models")
parser.add_argument("--exp_name", type=str, default="debug", help="Experiment name")
parser.add_argument("--exp_id", type=str, default="", help="Experiment ID")
parser.add_argument("--cuda", type=bool_flag, default=True, help="Run on GPU")
parser.add_argument("--export", type=str, default="txt", help="Export embeddings after training (txt / pth)")

# data
parser.add_argument("--src_lang", type=str, default='en', help="Source language")
parser.add_argument("--tgt_lang", type=str, default='es', help="Target language")
parser.add_argument("--emb_dim", type=int, default=300, help="Embedding dimension")
parser.add_argument("--max_vocab", type=int, default=200000, help="Maximum vocabulary size (-1 to disable)")
# training refinement
parser.add_argument("--n_refinement", type=int, default=5, help="Number of refinement iterations (0 to disable the refinement procedure)")
# dictionary creation parameters (for refinement)
parser.add_argument("--dico_train", type=str, default="default", help="Path to training dictionary (default: use identical character strings)")
parser.add_argument("--dico_eval", type=str, default="default", help="Path to evaluation dictionary")
parser.add_argument("--dico_method", type=str, default='csls_knn_10', help="Method used for dictionary generation (nn/invsm_beta_30/csls_knn_10)")
parser.add_argument("--dico_build", type=str, default='S2T&T2S', help="S2T,T2S,S2T|T2S,S2T&T2S")
parser.add_argument("--dico_threshold", type=float, default=0, help="Threshold confidence for dictionary generation")
parser.add_argument("--dico_max_rank", type=int, default=300, help="Maximum dictionary words rank (0 to disable)")
parser.add_argument("--dico_min_size", type=int, default=0, help="Minimum generated dictionary size (0 to disable)")
parser.add_argument("--dico_max_size", type=int, default=0, help="Maximum generated dictionary size (0 to disable)")
# reload pre-trained embeddings
parser.add_argument("--src_emb", type=str, default='', help="Reload source embeddings")
parser.add_argument("--tgt_emb", type=str, default='', help="Reload target embeddings")
parser.add_argument("--normalize_embeddings", type=str, default="", help="Normalize embeddings before training")


# parse parameters
params = parser.parse_args()

# check parameters
assert not params.cuda or torch.cuda.is_available()
assert params.dico_train in ["identical_char", "default"] or os.path.isfile(params.dico_train)
assert params.dico_build in ["S2T", "T2S", "S2T|T2S", "S2T&T2S"]
assert params.dico_max_size == 0 or params.dico_max_size < params.dico_max_rank
assert params.dico_max_size == 0 or params.dico_max_size > params.dico_min_size
assert os.path.isfile(params.src_emb)
assert os.path.isfile(params.tgt_emb)
assert params.dico_eval == 'default' or os.path.isfile(params.dico_eval)
assert params.export in ["", "txt", "pth"]

# build logger / model / trainer / evaluator
logger = initialize_exp(params)
src_emb, tgt_emb, mapping, _ = build_model(params, False)

trainer = Trainer(src_emb, tgt_emb, mapping, None, params)

evaluator = Evaluator(trainer)


logger.info("Compute candidate indices..............")
# compute_candidates_for_method_similarity(params.src_dico.word2id, params.tgt_dico.word2id, params)


# load a training dictionary. if a dictionary path is not provided, use a default
# one ("default") or create one based on identical character strings ("identical_char")
trainer.load_training_dico(params.dico_train)
# print(params.tgt_dico.id2word[0])

"""
Learning loop for Procrustes Iterative Learning
"""
# print("BUILD : " + str(params.dico_build))
for n_iter in range(params.n_refinement + 1):

    logger.info('Starting iteration %i...' % n_iter)

    # build a dictionary from aligned embeddings (unless
    # it is the first iteration and we use the init one)
    if n_iter > 0 or not hasattr(trainer, 'dico'):
        trainer.build_dictionary()

    # apply the Procrustes solution
    trainer.procrustes()

    # embeddings evaluation
    to_log = OrderedDict({'n_iter': n_iter})
    evaluator.all_eval(to_log)

    # JSON log / save best model / end of epoch
    logger.info("__log__:%s" % json.dumps(to_log))
    trainer.save_best(to_log, VALIDATION_METRIC)
    logger.info('End of iteration %i.\n\n' % n_iter)


# export embeddings
# if params.export:
#     trainer.reload_best()
#     trainer.export()
