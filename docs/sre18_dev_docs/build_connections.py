import sys

enroll = sys.argv[1]
key = sys.argv[2]
spk2utt_file = sys.argv[3]

import pdb
pdb.set_trace()
spk2utt = {}
with open(enroll, 'r') as f:
    f.readline()
    for line in f.readlines():
        spk, utt, dummy = line.strip().split('\t')
        utt = utt.split('.')[0]
        if spk not in spk2utt:
            spk2utt[spk] = []
        spk2utt[spk].append(utt)

with open(key, 'r') as f:
    f.readline()
    for line in f.readlines():
        info = line.strip().split('\t')
        spk = info[0]
        utt = info[1].split('.')[0]
        if info[3] == 'target':
            spk2utt[spk].append(utt)

utt2spk = {}
for spk in spk2utt:
    for utt in spk2utt[spk]:
        if utt not in utt2spk:
            utt2spk[utt] = []
        utt2spk[utt].append(spk)


def find_connected(utt, utt2spk, spk2utt, utt_visited, spk_visited, utt_set):
    utt_visited[utt] = True
    for spk in utt2spk[utt]:
        if spk_visited[spk]:
            continue
        for u in spk2utt[spk]:
            if utt_visited[u]:
                continue
            utt_set.append(u)
            find_connected(u, utt2spk, spk2utt, utt_visited, spk_visited, utt_set)
        spk_visited[spk] = True


utts = list(utt2spk.keys())
index = 0 
utt_visited = {}
for utt in utts:
    utt_visited[utt] = False
spk_visited = {}
for spk in spk2utt:
    spk_visited[spk] = False
with open(spk2utt_file, 'w') as f:
    for utt in utts:
        if utt_visited[utt]:
            continue
        utt_set = [utt]
        find_connected(utt, utt2spk, spk2utt, utt_visited, spk_visited, utt_set)
    
        f.write('sre18_dev_%d' % (index))
        for u in utt_set:
            f.write(' sre18_dev_%d-%s' % (index, u))
        f.write('\n')
        index += 1

