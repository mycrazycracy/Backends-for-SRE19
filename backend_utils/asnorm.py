import sys
import operator
import numpy as np

if len(sys.argv) != 8:
    print('usage: %s asnorm_type start_portion end_portin enroll_scores test_scores scores asnorm_scores' % sys.argv[0])
    quit()

asnorm_type = sys.argv[1]
start_port = sys.argv[2]
end_port = sys.argv[3]
enroll_scores = sys.argv[4]
test_scores = sys.argv[5]
scores = sys.argv[6]
asnorm_scores = sys.argv[7]

asnorm_type = int(asnorm_type)
assert(asnorm_type == 1 or asnorm_type == 2), "The asnorm_type must be 1 or 2"

# If portion is a friction, it is the percentage of the total cohort set.
start_port = float(start_port)
end_port = float(end_port)
assert(end_port > 0), "The end portion must > 0"

# Process enrollment scores
enroll2scores = {}
with open(enroll_scores, 'r') as f:
    for line in f.readlines():
        e, c, s = line.strip().split(' ')
        if e not in enroll2scores:
            enroll2scores[e] = {}
        enroll2scores[e][c] = float(s)

if asnorm_type == 1:
    enroll2mean = {}
    enroll2std = {}
else:
    enroll2top_scores = {}
for e in enroll2scores:
    info = sorted(enroll2scores[e].items(), key=operator.itemgetter(1), reverse=True)
    if start_port < 1:
        start_num = min(int(start_port * len(info)), len(info)-1)
    else:
        start_num = min(int(start_port), len(info)-1)
    if end_port < 1:
        end_num = min(int(end_port * len(info)), len(info))
    else:
        end_num = min(int(end_port), len(info))
    if asnorm_type == 1:
        tmp = [i[1] for i in info[start_num:end_num]]
        enroll2mean[e] = np.mean(tmp)
        enroll2std[e] = np.std(tmp)
    else:
        enroll2top_scores[e] = info[start_num:end_num]


# Process test scores
test2scores = {}
with open(test_scores, 'r') as f:
    for line in f.readlines():
        t, c, s = line.strip().split(' ')
        if t not in test2scores:
            test2scores[t] = {}
        test2scores[t][c] = float(s)

if asnorm_type == 1:
    test2mean = {}
    test2std = {}
else:
    test2top_scores = {}
for t in test2scores:
    info = sorted(test2scores[t].items(), key=operator.itemgetter(1), reverse=True)
    if start_port < 1:
        start_num = min(int(start_port * len(info)), len(info)-1)
    else:
        start_num = min(int(start_port), len(info)-1)
    if end_port < 1:
        end_num = min(int(end_port * len(info)), len(info))
    else:
        end_num = min(int(end_port), len(info))
    if asnorm_type == 1:
        tmp = [i[1] for i in info[start_num:end_num]]
        test2mean[t] = np.mean(tmp)
        test2std[t] = np.std(tmp)
    else:
        test2top_scores[t] = info[start_num:end_num]

with open(asnorm_scores, 'w') as fp_out:
    with open(scores, 'r') as fp_in:
        for line in fp_in.readlines():
            e, t, s = line.strip().split(' ')
            s = float(s)
            if asnorm_type == 1:
                new_s = 0.5 * ((s - enroll2mean[e]) / enroll2std[e] + (s - test2mean[t]) / test2std[t])
            else:
                enroll_asnorm_scores = []
                for st in test2top_scores[t]:
                    enroll_asnorm_scores.append(enroll2scores[e][st[0]])
                enroll_mean = np.mean(enroll_asnorm_scores)
                enroll_std = np.std(enroll_asnorm_scores)

                test_asnorm_scores = []
                for st in enroll2top_scores[e]:
                    test_asnorm_scores.append(test2scores[t][st[0]])
                test_mean = np.mean(test_asnorm_scores)
                test_std = np.std(test_asnorm_scores)

                new_s = 0.5 * ((s - enroll_mean) / enroll_std + (s - test_mean) / test_std)

            fp_out.write("%s %s %f\n" % (e, t, new_s))

