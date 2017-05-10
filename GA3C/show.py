import matplotlib.pyplot as plt
import re
import sys

def get_value(tag, line, f):
    digit_str = r":\s+([+-]?\d+(.\d+)?)"
    return f(re.search(tag + digit_str, line).group(1))

def plot_all(ts_list, score_list, rscore_list, rpps_list, pps_list, tps_list, nt_list, np_list, na_list, filename, nepisode=35000):
    ts_list = ts_list[:nepisode]
    score_list = score_list[:nepisode]
    rscore_list = rscore_list[:nepisode]
    rpps_list = rpps_list[:nepisode]
    pps_list = pps_list[:nepisode]
    tps_list = tps_list[:nepisode]
    nt_list = nt_list[:nepisode]
    np_list = np_list[:nepisode]
    na_list = na_list[:nepisode]

    tag = re.search(r"(\d{5,})", filename).group(1)
    plt.clf()
    plt.plot(ts_list, score_list, label = "score")
    plt.plot(ts_list, rscore_list, label = "smooth score")
    plt.grid()
    plt.xlabel("seconds")
    plt.legend()
    plt.savefig("score-%s.png" % tag)

    plt.clf()
    plt.plot(ts_list, rpps_list, label = "smooth pps")
    plt.plot(ts_list, pps_list, label = "pps")
    plt.plot(ts_list, tps_list, label = "tps")
    plt.legend()
    plt.xlabel("seconds")
    plt.savefig("rate-%s.png" % tag)

    plt.clf()
    plt.plot(ts_list, nt_list, label = "#trainer")
    plt.plot(ts_list, np_list, label = "#predictor")
    plt.plot(ts_list, na_list, label = "#agent")
    plt.legend()
    plt.xlabel("seconds")
    plt.savefig("schedule-%s.png" % tag)

def plot_cmp_score(t1, t2, s1, s2, file1, file2, nepisode = 20000):
    t1 = t1[:nepisode]
    t2 = t2[:nepisode]
    s1 = s1[:nepisode]
    s2 = s2[:nepisode]
    tag1 = re.search(r"(\d{5,})", file1).group(1)
    tag2 = re.search(r"(\d{5,})", file2).group(1)
    plt.clf()
    plt.plot(t1, s1, label = tag1)
    plt.plot(t2, s2, label = tag2)
    plt.grid()
    plt.xlabel("seconds")
    plt.legend()
    plt.savefig("score-%s-%s.png" % (tag1, tag2))

def get_list(filename):
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
    return ts_list, score_list, rscore_list, rpps_list, pps_list, tps_list, nt_list, np_list, na_list

def main():
    if len(sys.argv) == 3:
        file1 = sys.argv[1]
        file2 = sys.argv[2]
        t1, _, s1, _, _, _, _, _, _ = get_list(file1)
        t2, _, s2, _, _, _, _, _, _ = get_list(file2)
        plot_cmp_score(t1, t2, s1, s2, file1, file2)
    elif len(sys.argv) == 2:
        filename = sys.argv[1]
        ts_list, score_list, rscore_list, rpps_list, pps_list, tps_list, nt_list, np_list, na_list = get_list(filename)
        plot_all(ts_list, score_list, rscore_list, rpps_list, pps_list, tps_list, nt_list, np_list, na_list, filename)

if __name__ == "__main__":
    main()
