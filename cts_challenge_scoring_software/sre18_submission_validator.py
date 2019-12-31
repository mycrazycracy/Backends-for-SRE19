#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 18 15:12:37 2018

@author: Omid Sadjadi <omid.sadjadi@nist.gov>
"""

import argparse


def validate_me(system_output, trials_list, max_lines=20):
    invalid = False
    line_counter = 0
    err_str = ''
#    with open(trials_list) as fid1, open(system_output) as fid2:
    fid1 = open(trials_list)
    fid2 = open(system_output)
    line_no = 0
    for line in fid1:
        line_no += 1
        ref_list = split_line(line)
        sys_list = split_line(fid2.readline())
        # checking if the number of lines in two files match
        if sys_list == ['']:
            err_str += ('The system output has less lines than the trial '
                        'list.')
            invalid = True
            break
        # checking if the delimiter is TAB
        if len(sys_list) != len(ref_list) + 1:
            err_str += ('Line {}: Incorrect number of columns/fields. '
                        'Expected {}, got {} instead. TAB (\\t) delimiter '
                        'should be used.\n'.format(line_no,
                                                   len(ref_list)+1,
                                                   len(sys_list)))
            invalid = True
            line_counter += 1
        else:
            # checking if the fields match the reference
            if sys_list[:3] != ref_list:
                err_str += ('Line {}: Incorrect field(s). Expected "{}", '
                            'got "{}" instead.\n'
                            .format(line_no, '\t'.join(ref_list),
                                    '\t'.join(sys_list[:3])))
                invalid = True
                line_counter += 1
            if line_no == 1:
                # checking if "LLR" is in the header
                if sys_list[-1] != 'LLR':
                    err_str += ('Line {}: Expected "LLR" (case-sensitive) '
                                'in the header, got "{}" instead.\n'
                                .format(line_no, sys_list[-1]))
                    invalid = True
                    line_counter += 1
            else:
                # checking if the scores are floats
                if not is_float(sys_list[-1]):
                    err_str += ('Line {}: Expected float in the LLR '
                                'column, got "{}" instead.\n'
                                .format(line_no, sys_list[-1]))
                    invalid = True
                    line_counter += 1
        if line_counter >= max_lines:
            break
    ref_list = fid1.readline()
    sys_list = fid2.readline()
    # checking if the number of lines in two files match
    if sys_list and not ref_list:
        err_str += ('The system output has more lines than the trial list.')
        invalid = True
    fid1.close()
    fid2.close()
    if err_str and invalid:
        print("\n" + err_str)
    return invalid


def split_line(line, delimiter='\t'):
    return line.strip().split(delimiter)


def is_float(astr):
    try:
        float(astr)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description='SRE18 Submission Validator.')
    parser.add_argument("-o", "--output", help="path to system output file",
                        type=str, required=True)
    parser.add_argument("-l", "--trials", help="path to the list of trials, "
                        "e.g., /path/to/sre18_dev_trials.tsv",
                        type=str, required=True)
    parser.add_argument("-n", "--lines", help="Number of lines to print",
                        type=int, default=20)
    args = parser.parse_args()
    system_output = args.output
    trials_list = args.trials
    max_lines = args.lines
    validate_me(system_output, trials_list, max_lines)


if __name__ == '__main__':
    main()
