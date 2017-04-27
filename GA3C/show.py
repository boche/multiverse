import matplotlib.pyplot as plt
import re
import sys

def get_value(tag, line, f):
    digit_str = r":\s+([+-]?\d+(.\d+)?)"
    return f(re.search(tag + digit_str, line).group(1))

def myplot(ts_list, score_list, rscore_list, rpps_list, pps_list, tps_list, nt_list, np_list, na_list, filename):
    tag = re.search(r"(\d{5,})", filename).group(1)
    plt.clf()
    plt.plot(ts_list, score_list, label = "score")
    plt.plot(ts_list, rscore_list, label = "smooth score")
    plt.grid()
    plt.xlabel("seconds")
    plt.legend()
    plt.savefig("score-%s.pdf" % tag)

    plt.clf()
    plt.plot(ts_list, rpps_list, label = "smooth pps")
    plt.plot(ts_list, pps_list, label = "pps")
    plt.plot(ts_list, tps_list, label = "tps")
    plt.legend()
    plt.xlabel("seconds")
    plt.savefig("rate-%s.pdf" % tag)

    plt.clf()
    plt.plot(ts_list, nt_list, label = "#trainer")
    plt.plot(ts_list, np_list, label = "#predictor")
    plt.plot(ts_list, na_list, label = "#agent")
    plt.legend()
    plt.xlabel("seconds")
    plt.savefig("schedule-%s.pdf" % tag)

def main():
    filename = sys.argv[1]
    ts_list, score_list, rscore_list = [], [], []
    rpps_list, pps_list, tps_list = [], [], []
    nt_list, np_list, na_list = [], [], []
    with open(filename) as f:
        for line in f:
            if line.startswith("[Time:"):
                ts = get_value("Time", line, int)
                episode = get_value("Episode", line, int)
                score = get_value(" Score", line, float)
                rscore = get_value("RScore", line, float)
                rpps = get_value("RPPS", line, int)
                pps = get_value("\[PPS", line, int)
                tps = get_value("TPS", line, int)
                nt = get_value("NT", line, int)
                np = get_value("NP", line, int)
                na = get_value("NA", line, int)
                ts_list.append(ts)
                score_list.append(score)
                rscore_list.append(rscore)
                rpps_list.append(rpps)
                pps_list.append(pps)
                tps_list.append(tps)
                nt_list.append(nt)
                np_list.append(np)
                na_list.append(na)
                # print(episode, score, rscore)
                # print(rpps, pps, tps)
                # print(nt, np, na)
                # print(line)
    myplot(ts_list, score_list, rscore_list, rpps_list, pps_list, tps_list, nt_list, np_list, na_list, filename)

if __name__ == "__main__":
    main()
