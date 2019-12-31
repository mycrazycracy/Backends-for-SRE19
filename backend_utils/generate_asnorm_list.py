import sys

if len(sys.argv) != 4:
    print('usage: %s utt2spk cohort_list trials_output')
    quit()

utt2spk = sys.argv[1]
cohort_list = sys.argv[2]
trials = sys.argv[3]

utts = set()
with open(utt2spk, 'r') as f:
    for line in f.readlines():
        utts.add(line.strip().split(' ')[0])

cohorts = set()
with open(cohort_list, 'r') as f:
    for line in f.readlines():
        cohorts.add(line.strip().split(' ')[0])

with open(trials, 'w') as f:
    for u in utts:
        for c in cohorts:
            f.write("%s %s\n" % (u, c))
