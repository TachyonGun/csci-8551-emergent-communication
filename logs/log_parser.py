from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-i","--input_log",required=True, help="Path to input file")
    parser.add_argument("-t","--plot_title",required=True, help="Title of plot")
    parser.add_argument("-o","--output",required=True, help="Path to output losses")
    args = parser.parse_args()

    fP = open(args.input_log,"r")
    input_file = fP.read().strip()
    fP.close()
    lines = input_file.split("\n")
    valid_lines = list(filter(
        lambda line: line != "_________________________",
        lines
    ))
    losses = np.zeros(len(valid_lines))
    for i,line in enumerate(valid_lines):
        # 0 epoch 1 agent , landmarks 2 batch 3 last loss 4 min loss 5 min distance
        line_tokens = line.strip().replace('[','').split(']')
        loss = float(line_tokens[4].split(" ")[-1])
        losses[i] = loss

    plt.plot(np.arange(len(valid_lines)),losses)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title(args.plot_title)
    plt.savefig(args.output)

