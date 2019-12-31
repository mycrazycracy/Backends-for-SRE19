#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 22 11:33:56 2018

@author: Omid Sadjadi <omid.sadjadi@nist.gov>
"""

from itertools import product
import numpy as np
import sre_scorer as sc


def read_tsv_file(tsv_file, delimiter='\t', encoding='ascii'):
    return np.genfromtxt(tsv_file, names=True, delimiter=delimiter,
                         dtype=None, encoding=encoding)


def compute_partition_masks(trial_key, partitions, sub_part=None):
    mask_list = []
    psub_list = []
    nontar_mask = trial_key['targettype'] == 'nontarget'
    for part, part_elements in partitions.items():
        masks = []
        for elem in part_elements:
            msk = trial_key[part] == elem
            if elem == 'Y':
                diffnum_msk = ~msk
                # we exclude same-phone-number nontarget trials
                msk = np.logical_and(msk, ~nontar_mask)
                if sub_part == 'Y':
                    # injecting nontargets from different phone no. partition
                    diffnum_nontar = np.logical_and(diffnum_msk, nontar_mask)
                    msk = np.logical_or(msk, diffnum_nontar)
            masks.append(msk)
        psub_list.append(part_elements)
        mask_list.append(masks)
    psub_list = list(product(*psub_list))
    filt = np.array(list(map(lambda x: sub_part in x, psub_list)))
    mask_list = np.array(list(product(*mask_list)))
    if sub_part:
        mask_list = mask_list[filt]
    partition_masks = np.all(mask_list, axis=1)
    # filtering out empty partitions
    partition_masks = partition_masks[np.any(partition_masks, axis=1)]
    return partition_masks


def compute_equalized_act_cost(scores, partition_masks, tar_nontar_labs,
                               p_target, c_miss=1, c_fa=1):
    act_c = 0.
    for p_t in p_target:
        beta = c_fa * (1 - p_t) / (c_miss * p_t)
#        act_c_norm = np.zeros(len(partition_masks))
        fpr, fnr = np.zeros((2, len(partition_masks)))
        for ix, m in enumerate(partition_masks):
            part_scores = scores[m]
            part_labels = tar_nontar_labs[m]
            _, fpr[ix], fnr[ix] = sc.compute_actual_cost(part_scores,
                                                         part_labels, p_t,
                                                         c_miss, c_fa)
#        act_c += act_c_norm.mean()
        fnr_avg = 0. if np.all(np.isnan(fnr)) else np.nanmean(fnr)
        fpr_avg = 0. if np.all(np.isnan(fpr)) else np.nanmean(fpr)
        act_c += fnr_avg + beta * fpr_avg
    return act_c / len(p_target)


def compute_equalized_min_cost(scores, partition_masks, tar_nontar_labs, ptar):
    scores, labels, weights = compute_partition_weights(scores,
                                                        partition_masks,
                                                        tar_nontar_labs)
    fnr, fpr = sc.compute_pmiss_pfa_rbst(scores, labels, weights)
    eer = sc.compute_eer(fnr, fpr)
    min_c = 0.
    for pt in ptar:
        min_c += sc.compute_c_norm(fnr, fpr, pt)
    return eer, min_c / len(ptar)


def compute_partition_stats(partition_masks, tar_nontar_labs):
    max_tar_count = 0
    max_imp_count = 0
    for m in partition_masks:
        tar_cnt = tar_nontar_labs[m].sum()
        if tar_cnt > max_tar_count:
            max_tar_count = tar_cnt
        imp_cnt = np.sum(1 - tar_nontar_labs[m])
        if imp_cnt > max_imp_count:
            max_imp_count = imp_cnt
    return max_tar_count, max_imp_count


def compute_partition_weights(scores, partition_masks, tar_nontar_labs):
    max_tar_count, max_imp_count = compute_partition_stats(partition_masks,
                                                           tar_nontar_labs)
    part_scores = []
    part_labels = []
    count_weights = []
    for ix, mask in enumerate(partition_masks):
        labs = tar_nontar_labs[mask]
        num_targets = np.sum(labs)
        num_nontargets = labs.size - num_targets
        part_scores.append(scores[mask])
        part_labels.append(labs)
        tar_weight = max_tar_count/num_targets if num_targets > 0 else 0
        imp_weight = max_imp_count/num_nontargets if num_nontargets > 0 else 0
        acount_weights = np.empty(labs.shape, dtype='f')
        acount_weights[labs == 1] = tar_weight
        acount_weights[labs == 0] = imp_weight
        count_weights.append(acount_weights)
    part_scores = np.hstack(part_scores)
    part_labels = np.hstack(part_labels)
    count_weights = np.hstack(count_weights)
    return part_scores, part_labels, count_weights


def compute_numeric_labels(trial_key):
    tar_nontar_labs = np.zeros(trial_key['targettype'].size, dtype=np.int)
    tar_nontar_labs[trial_key['targettype'] == 'target'] = 1
    return tar_nontar_labs
