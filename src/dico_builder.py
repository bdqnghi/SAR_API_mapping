# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from logging import getLogger
import torch

from .utils import get_nn_avg_dist
from .utils import compute_candidates_for_method_similarity

logger = getLogger()


def get_candidates_2(emb1, emb2, params, build="S"):

    print("Emb1 size : " + str(emb1.size()))
    print("Emb2 size : " + str(emb2.size()))
    src_candidate_indices = params.src_candidate_indices
    
    if build != "S":
        src_candidate_indices = params.tgt_candidate_indices

    all_scores = []
    all_targets = []

   
    candidate_embs = []

    # print("Getting candidates.........")
    # print(src_candidate_indices)
    for i in src_candidate_indices:
        
        candidate_embs.append(emb1[i])
        # scores = emb2.mm(emb1[[i]].transpose(0, 1)).transpose(0, 1)
        
        # best_scores, best_targets = scores.topk(topk, dim=1, largest=True, sorted=True)

        # print(best_scores[0])
        # print("Best score : " + str(best_scores.size()))
        # print("Best target : " + str(best_targets.size()))
        # update scores / potential targets
        # all_scores.append(best_scores.cpu())
        # all_targets.append(best_targets.cpu())

    # print("Len candidate embs : " + str(len(candidate_embs)))
  
    # print(candidate_embs)
    candidate_embs = torch.stack(candidate_embs)
    scores = emb1.mm(candidate_embs.transpose(0, 1)).transpose(0, 1)
    best_scores, best_targets = scores.topk(3, dim=1, largest=True, sorted=True)

    # print(best_scores)

    # print("------------------------------------------------------------------------")
    source_candidates = list()
    for best_target in best_targets.cpu():
        for b in best_target.numpy():
            source_candidates.append(b)

    source_candidates =  list(set(source_candidates))
    
    source_candidates_embs = list()
    for i in source_candidates:
        source_candidates_embs.append(emb1[i])

    source_candidates_embs = torch.stack(source_candidates_embs)
    scores = emb2.mm(source_candidates_embs.transpose(0, 1)).transpose(0, 1)
    best_scores, best_targets = scores.topk(1, dim=1, largest=True, sorted=True)
    
    target_candidates = list()
    for best_target in best_targets.cpu():
        for b in best_target.numpy():
            target_candidates.append(b)
    
    all_pairs = set(list(zip(source_candidates, target_candidates)))
    
    print("All pairs len : " + str(len(all_pairs)))
    return all_pairs


def get_candidates(emb1, emb2, params):
    """
    Get best translation pairs candidates.
    """
    bs = 128
    topk = 2

    all_scores = []
    all_targets = []

    # number of source words to consider
    n_src = emb1.size(0)
    if params.dico_max_rank > 0 and not params.dico_method.startswith('invsm_beta_'):
        n_src = params.dico_max_rank

    # nearest neighbors
    if params.dico_method == 'nn':
  
        # print("Size of emb 1 : " + str(emb1.size()))
        # print("Size of emb 2 : " + str(emb2.size()))
        # for every source word
        for i in range(0, n_src, bs):

            # print("Step : " + str(i))
            # compute target words scores
            # print("Min : " + str(min(n_src, i + bs)))
            # print("Size of candidates : " + str(emb1[i:min(n_src, i + bs)].size()))
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            # print("Scores size : " + str(scores.size()))
            best_scores, best_targets = scores.topk(topk, dim=1, largest=True, sorted=True)

            # print(best_scores[0])
            # print("Best score : " + str(best_scores.size()))
            # print("Best target : " + str(best_targets.size()))
            # print(best_targets)
            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        # print("All score size : " + str(len(all_scores)))
        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)
      

    # inverted softmax
    elif params.dico_method.startswith('invsm_beta_'):

        beta = float(params.dico_method[len('invsm_beta_'):])

        # for every target word
        for i in range(0, emb2.size(0), bs):

            # compute source words scores
            scores = emb1.mm(emb2[i:i + bs].transpose(0, 1))
            scores.mul_(beta).exp_()
            scores.div_(scores.sum(0, keepdim=True).expand_as(scores))

            best_scores, best_targets = scores.topk(topk, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append((best_targets + i).cpu())

        all_scores = torch.cat(all_scores, 1)
        all_targets = torch.cat(all_targets, 1)

        all_scores, best_targets = all_scores.topk(topk, dim=1, largest=True, sorted=True)
        all_targets = all_targets.gather(1, best_targets)

    # contextual dissimilarity measure
    elif params.dico_method.startswith('csls_knn_'):

        knn = params.dico_method[len('csls_knn_'):]
        assert knn.isdigit()
        knn = int(knn)

        # average distances to k nearest neighbors
        average_dist1 = torch.from_numpy(get_nn_avg_dist(emb2, emb1, knn))
        average_dist2 = torch.from_numpy(get_nn_avg_dist(emb1, emb2, knn))
        average_dist1 = average_dist1.type_as(emb1)
        average_dist2 = average_dist2.type_as(emb2)

        # for every source word
        for i in range(0, n_src, bs):

            # compute target words scores
            scores = emb2.mm(emb1[i:min(n_src, i + bs)].transpose(0, 1)).transpose(0, 1)
            scores.mul_(2)
            scores.sub_(average_dist1[i:min(n_src, i + bs)][:, None] + average_dist2[None, :])
            best_scores, best_targets = scores.topk(topk, dim=1, largest=True, sorted=True)

            # update scores / potential targets
            all_scores.append(best_scores.cpu())
            all_targets.append(best_targets.cpu())

        all_scores = torch.cat(all_scores, 0)
        all_targets = torch.cat(all_targets, 0)

    all_pairs = torch.cat([torch.arange(0, all_targets.size(0)).long().unsqueeze(1),all_targets[:, 0].unsqueeze(1)], 1)
    # print(all_targets)
    # print(all_scores.size())
    # print(all_pairs.size())
    # print((n_src, topk))
    # sanity check
    assert all_scores.size() == all_pairs.size() == (n_src, topk)

    # sort pairs by score confidence
    diff = all_scores[:, 0] - all_scores[:, 1]
    reordered = diff.sort(0, descending=True)[1]
    all_scores = all_scores[reordered]
    all_pairs = all_pairs[reordered]

    # max dico words rank
    if params.dico_max_rank > 0:
        selected = all_pairs.max(1)[0] <= params.dico_max_rank
        mask = selected.unsqueeze(1).expand_as(all_scores).clone()
        all_scores = all_scores.masked_select(mask).view(-1, 2)
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

    # max dico size
    if params.dico_max_size > 0:
        all_scores = all_scores[:params.dico_max_size]
        all_pairs = all_pairs[:params.dico_max_size]

    # min dico size
    diff = all_scores[:, 0] - all_scores[:, 1]
    if params.dico_min_size > 0:
        diff[:params.dico_min_size] = 1e9

    # confidence threshold
    if params.dico_threshold > 0:
        mask = diff > params.dico_threshold
        logger.info("Selected %i / %i pairs above the confidence threshold." % (mask.sum(), diff.size(0)))
        mask = mask.unsqueeze(1).expand_as(all_pairs).clone()
        all_pairs = all_pairs.masked_select(mask).view(-1, 2)

   
    return all_pairs

def get_word_pairs(index_pairs, params):
    src_id2word = params.src_dico.id2word
    tgt_id2word = params.tgt_dico.id2word

    pairs = list()
    for pair in list(index_pairs):
        src = pair[0]
        tgt = pair[1]
        src_name = src_id2word[src]
        tgt_name = tgt_id2word[tgt]
        pairs.append((src_name,tgt_name))
    return pairs

def build_dictionary(src_emb, tgt_emb, mutual_nn, params, s2t_candidates=None, t2s_candidates=None):
   
    """
    Build a training dictionary given current embeddings / mapping.
    """
    s2t = 'S2T' in params.dico_build
    t2s = 'T2S' in params.dico_build
    assert s2t or t2s

          
    if mutual_nn == 1:
        # This part is for the old pairs , switch back later
        s2t_candidates = get_candidates(src_emb, tgt_emb, params)
        t2s_candidates = get_candidates(tgt_emb, src_emb, params)
        t2s_candidates = torch.cat([t2s_candidates[:, 1:], t2s_candidates[:, :1]], 1)
        s2t_candidates = set([(a, b) for a, b in s2t_candidates.numpy()])
        t2s_candidates = set([(a, b) for a, b in t2s_candidates.numpy()])

        # s2t_candidates = get_candidates_2(src_emb, tgt_emb, params)
        # t2s_candidates = get_candidates_2(tgt_emb, src_emb, params)
        # t2s_candidates_temp = list()
        # for t in t2s_candidates:
        #     t2s_candidates_temp.append(t[::-1])
        # t2s_candidates = set(t2s_candidates_temp)



        temp_pairs = list()
        for i, position in enumerate(params.src_candidate_indices):
            pair = (position, params.tgt_candidate_indices[i])
            temp_pairs.append(pair)
      
        # if params.dico_build == 'S2T|T2S':

        final_pairs = s2t_candidates | t2s_candidates

        # else:
            # final_pairs = s2t_candidates & t2s_candidates

        final_pairs = list(final_pairs) 
        final_pairs.extend(temp_pairs)
        final_pairs = set(final_pairs)
        print(len(final_pairs))
        if len(final_pairs) == 0:
            logger.warning("Empty intersection ...")
            return None
        # dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))
    else:
        final_pairs = list()
        src_candidate_indices = params.src_candidate_indices
        tgt_candidate_indices = params.tgt_candidate_indices
        for i, position in enumerate(src_candidate_indices):
            pair = (position, tgt_candidate_indices[i])
            final_pairs.append(pair)

        # print("FINAL PAIRS FOR THE TEXT SIMILAR CASE")
        # print(final_pairs)
        final_pairs = list(set(final_pairs))
        print(len(final_pairs))
    dico = torch.LongTensor(list([[int(a), int(b)] for (a, b) in final_pairs]))

    logger.info('New train dictionary of %i pairs.' % dico.size(0))


    return dico.cuda() if params.cuda else dico


