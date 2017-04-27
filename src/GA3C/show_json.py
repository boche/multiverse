import matplotlib.pyplot as plt
import sys
import json
import numpy as np

def get_value(tag, line, f):
    digit_str = r":\s+([+-]?\d+(.\d+)?)"
    return f(re.search(tag + digit_str, line).group(1))

def myplot(ts_list, score_list, rscore_list):
    plt.plot(ts_list, score_list, label = "score")
    plt.plot(ts_list, rscore_list, label = "smooth score")
    plt.grid()
    plt.xlabel("seconds")
    plt.legend()
    plt.savefig("a3c-score.pdf")

def main():
    filename = sys.argv[1]
    ts_start = None
    ts_list, score_list = [], []
    rscore_list = []
    with open(filename) as f:
        data = json.load(f)
        for ts, _, score in data:
            if ts_start is None:
                ts_start = ts
            ts_list.append(int(ts - ts_start))
            score_list.append(score)
            rscore_list.append(np.mean(np.asarray(score_list[-10:])))
    myplot(ts_list, score_list, rscore_list)

if __name__ == "__main__":
    main()
