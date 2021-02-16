import numpy as np
from typing import List, Dict
import os
from matplotlib import pyplot as plt
import numpy as np

from regexes import *

BASE_PATH = os.path.dirname(os.path.abspath(__file__))


def normalize_infs(arr: List[float]) -> List[float]:
    arr = list(map(lambda val: 1e20 if val >= 1e20 else val, arr))
    arr = list(map(lambda val: -1e20 if val <= -1e20 else val, arr))
    return arr


def print_bounds(lbs: List[float], ubs: List[float], prnt_name: str = "", num_print: int = 10) -> None:
    print(prnt_name, " lbs: ", lbs[:num_print])
    print(prnt_name, " ubs: ", ubs[:num_print])


def compare_arrays_diff_idx(arr1: List[float], arr2: List[float], arr_name: str = "") -> bool:
    assert len(arr1) == len(arr2)
    res_eq = True
    for i in range(len(arr1)):
        if not np.isclose(arr1[i], arr2[i]):
            print(arr_name, " index: ", i, ", val1: ", arr1[i], ", val2:", arr2[i])
            res_eq = False
    return res_eq


def num_inf_bounds(lbs: List[float], ubs: List[float]) -> int:
    num = 0
    for i, lb in enumerate(lbs):
        if lbs[i] <= -1e20:
            num += 1
        if ubs[i] >= 1e20:
            num += 1
    return num


def get_alg_progress_measure_dict(output: str, alg: str) -> Dict:
    outdict = {} # data struct to hold plotting data
    with_measure_output = get_regex_result(with_measure_output_pattern(alg), output)
    without_measure_output = get_regex_result(without_measure_output_pattern(alg), output)

    num_rounds = int(get_regex_result(num_rounds_pattern(alg), with_measure_output, "nrounds"))
    assert(num_rounds == int(get_regex_result(num_rounds_pattern(alg), without_measure_output, "nrounds")))

    for prop_round in range(1, num_rounds+1):
        score = float(get_regex_result(round_measures_pattern(prop_round), with_measure_output, "score"))
        k = int(get_regex_result(round_measures_pattern(prop_round), with_measure_output, "k"))
        n = int(get_regex_result(round_measures_pattern(prop_round), with_measure_output, "n"))
        timestamp = int(get_regex_result(round_timestamp_pattern(prop_round, alg), without_measure_output, "timestamp"))
        outdict[prop_round] = [score, k, n, timestamp]
    return outdict


def create_progress_plots(prob_name: str, data: Dict) -> None:
    fig = plt.figure(figsize=(14, 14))
    plt.style.use('bmh')

    ###### 1 ########
    fig.add_subplot(321)
    for algorithm in data:
        rounds = list(map(lambda x: x, data[algorithm]))
        scores = list(map(lambda x: data[algorithm][x][0], data[algorithm]))
        plt.plot(rounds, scores, label=str(prob_name) + "-" + str(algorithm))

    plt.legend(fancybox=True, framealpha=0.5)
    plt.ylabel("Score value")
    plt.xlabel("Round")
    plt.tick_params(which='minor', left=False)

    fig.add_subplot(323)
    for algorithm in data:
        rounds = list(map(lambda x: x, data[algorithm]))
        ks = scores = list(map(lambda x: data[algorithm][x][1], data[algorithm]))
        plt.plot(rounds, ks, label=str(prob_name) + "-" + str(algorithm))

    plt.legend(fancybox=True, framealpha=0.5)
    plt.ylabel("k value")
    plt.xlabel("Round")
    plt.tick_params(which='minor', left=False)

    fig.add_subplot(325)
    for algorithm in data:
        rounds = list(map(lambda x: x, data[algorithm]))
        ns = list(map(lambda x: data[algorithm][x][2], data[algorithm]))
        plt.plot(rounds, ns, label=prob_name + "-" + str(algorithm))

    plt.legend(fancybox=True, framealpha=0.5)
    plt.ylabel("n value")
    plt.xlabel("Round")
    plt.tick_params(which='minor', left=False)

    ###### 2 ########
    fig.add_subplot(322)
    for algorithm in data:
        timestamps = list(map(lambda x: data[algorithm][x][3], data[algorithm]))
        scores = list(map(lambda x: data[algorithm][x][0], data[algorithm]))
        plt.plot(timestamps, scores, label=str(prob_name) + "-" + str(algorithm))

    plt.legend(fancybox=True, framealpha=0.5)
    plt.ylabel("Score value")
    plt.xlabel("Time in nanoseconds")
    plt.tick_params(which='minor', left=False)

    fig.add_subplot(324)
    for algorithm in data:
        timestamps = list(map(lambda x: data[algorithm][x][3], data[algorithm]))
        ks = list(map(lambda x: data[algorithm][x][1], data[algorithm]))
        plt.plot(timestamps, ks, label=str(prob_name) + "-" + str(algorithm))

    plt.legend(fancybox=True, framealpha=0.5)
    plt.ylabel("k value")
    plt.xlabel("Time in nanoseconds")
    plt.tick_params(which='minor', left=False)

    fig.add_subplot(326)
    for algorithm in data:
        timestamps = list(map(lambda x: data[algorithm][x][3], data[algorithm]))
        ns = list(map(lambda x: data[algorithm][x][2], data[algorithm]))
        plt.plot(timestamps, ns, label=str(prob_name) + "-" + str(algorithm))

    plt.legend(fancybox=True, framealpha=0.5)
    plt.ylabel("n value")
    plt.xlabel("Time in nanoseconds")
    plt.tick_params(which='minor', left=False)

    # save fig
    fig.suptitle(prob_name + ' progress measure plots')
    plt.tight_layout()
    folder = 'progress_plots'
    if not os.path.exists(folder):
        os.makedirs(folder)
    fig.savefig(os.path.join(folder, prob_name + ".pdf"), format='pdf', bbox_inches='tight', pad_inches=1)


def plot_progress_save_pdf(output: str) -> None:
    if not get_regex_result(seq_to_ato_pattern, output, 'match'):
        print("Results don't match. Aborting plotting progress and returning.")
        return

    prob_name = get_regex_result(prob_name_pattern, output, "prob_file").split("/")[-1]

    plot_data = {
        "cpu_seq" : get_alg_progress_measure_dict(output, "cpu_seq"),
        "gpu_atomic" : get_alg_progress_measure_dict(output, "gpu_atomic")
    }
    # cpu_seq
    create_progress_plots(prob_name, plot_data)




