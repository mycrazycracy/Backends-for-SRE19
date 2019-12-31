import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("extension", type=str)
    parser.add_argument("trials", type=str, help="The trials")
    parser.add_argument("input_score", type=str, help="The input score")
    parser.add_argument("output_score", type=str, help="The output score")
    args = parser.parse_args()

    scores = {}
    with open(args.input_score, 'r') as f:
        for line in f.readlines():
            info = line.strip().split(" ")
            t = info[0] + info[1]
            scores[t] = info[2]

    fp_out = open(args.output_score, 'w')
    # fp_out.write("modelid\tsegmentid\tside\tLLR\n")

    with open(args.trials, 'r') as f:
        for line in f.readlines():
            info = line.strip().split(' ')
            k = info[0] + info[1]
            fp_out.write('%s\t%s.%s\ta\t%s\n' % (info[0], info[1], args.extension, scores[k]))
    fp_out.close()
