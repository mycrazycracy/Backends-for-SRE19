#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 21 14:40:51 2018

@author: Omid Sadjadi <omid.sadjadi@nist.gov>
"""

import argparse
import scoring_utils as st
from sre18_submission_validator import validate_me


def score_me(system_output_file, trial_key_file,
             configuration, partitions, sub_partition):
    trial_key = st.read_tsv_file(trial_key_file)
    sys_out = st.read_tsv_file(system_output_file)
    scores = sys_out['LLR']
    tar_nontar_labs = st.compute_numeric_labels(trial_key)
    results = {}
    for ds, p_target in configuration.items():
        dataset_mask = trial_key['data_source'] == ds
        ds_trial_key = trial_key[dataset_mask]
        ds_scores = scores[dataset_mask]
        ds_trial_labs = tar_nontar_labs[dataset_mask]
        if ds == 'cmn2':
            partition_masks = st.compute_partition_masks(ds_trial_key,
                                                         partitions,
                                                         sub_partition)
        else:
            partition_masks = [dataset_mask[dataset_mask]]
        act_c = st.compute_equalized_act_cost(ds_scores, partition_masks,
                                              ds_trial_labs, p_target)
        eer, min_c = st.compute_equalized_min_cost(ds_scores,
                                                   partition_masks,
                                                   ds_trial_labs, p_target)
        results[ds] = [eer, min_c, act_c]
    return results


def main():
    parser = argparse.ArgumentParser(description='SRE18 Submission Scorer.')
    parser.add_argument("-o", "--output", help="path to system output file",
                        type=str, required=True)
    parser.add_argument("-l", "--trials", help="path to the list of trials, "
                        "e.g., /path/to/LDC2018E46/doc/sre18_dev_trials.tsv",
                        type=str, required=True)
    parser.add_argument("-r", "--key", help="path to the trial key, e.g., "
                        "/path/to/LDC2018E46/doc/sre18_dev_trial_key.tsv",
                        type=str, required=True)
    allowed_choices = ['male', 'female', 'Y', 'N', '1', '3', 'pstn', 'voip']
    parser.add_argument("-p", "--subpartition", help="subpartition for which "
                        "the score should be computed, e.g., male. This "
                        "option can only be set for the CMN2",
                        type=str, choices=allowed_choices)
    parser.add_argument("-n", "--lines", help="Number of lines to print",
                        type=int, default=20)
    args = parser.parse_args()
    system_output_file = args.output
    trial_list_file = args.trials
    sub_partition = args.subpartition
    trial_key_file = args.key
    max_lines = args.lines
    if validate_me(system_output_file, trial_list_file, max_lines):
        print("System output failed the validation step. Exiting...\n")
        return
    # ---- configuration and partitions ----
    # BEGIN
    configuration = {'cmn2': [0.01, 0.005],
                     'vast': [0.05]}
    partitions = {'num_enroll_segs': ['1', '3'],
                  'phone_num_match': ['Y', 'N'],
                  'gender': ['male', 'female'],
                  'source_type': ['pstn', 'voip']}
    # END
    results = score_me(system_output_file, trial_key_file, configuration,
                       partitions, sub_partition)
    print('\nSet\tEER[%]\tmin_C\tact_C')
    cprimary, cnt = 0., 0.
    for ds, res in results.items():
        eer, minc, actc = res
        cprimary += actc
        cnt += 1
        print('{}\t{:05.2f}\t{:.3f}\t{:.3f}'.format(ds.upper(), eer*100,
              minc, actc))
    print('{}\t{}\t{}\t{:.3f}'.format('Both', '--', '--', cprimary/cnt))


if __name__ == '__main__':
    main()
