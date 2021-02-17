import json
# import numpy as np
import math
import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter
from typing import List, Tuple
import os
import sys

# add parent directory to path, need it to load some functions
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from regexes import result_pattern

cpu_seq_key = "cpu_seq"
cpu_omp_key = "cpu_omp"
gpu_reduction_key = "gpu_reduction"
gpu_atomic_key = "gpu_atomic"

cpu_seq_time_key = "cpu_seq_time"
cpu_omp_time_key = "cpu_omp_time"
gpu_reduction_time_key = "gpu_reduction_time"
gpu_atomic_time_key = "gpu_atomic_time"
cpu_seq_rounds_key = "cpu_seq_rounds"
cpu_omp_rounds_key = "cpu_omp_rounds"
gpu_reduction_rounds_key = "gpu_reduction_rounds"
gpu_atomic_rounds_key = "gpu_atomic_rounds"

machine_to_time = {
    cpu_seq_key: cpu_seq_time_key,
    cpu_omp_key: cpu_omp_time_key,
    gpu_reduction_key: gpu_reduction_time_key,
    gpu_atomic_key: gpu_atomic_time_key
}


# Helper methods
def getsetovernvarscons(test_set, threashold):
    new_test_set = {}
    for f in test_set.keys():
        data = list(filter(lambda x: x["nvars"] >= threashold or x["ncons"] >= threashold, test_set[f]))
        print("num instance for ", f, " with at least ", threashold, " cons or vars: ", len(data))
        new_test_set[f] = data
    return new_test_set


def reducesetnnz(test_set, nnz):
    new_test_set = {}
    for f in test_set.keys():
        new_test_set[f] = list(filter(lambda x: (x["nnz"] >= nnz), test_set[f]))
        print("num instance for ", f, " with at least ", nnz, " nnz: ", len(new_test_set[f]))
    return new_test_set


def reduceset(test_set, lb, ub):
    new_test_set = {}
    for f in test_set.keys():
        reduce_f = lambda x: (x["nvars"] >= lb or x["ncons"] >= lb) and (x["nvars"] < ub and x["ncons"] < ub)
        data = list(filter(reduce_f, test_set[f]))
        new_test_set[f] = data
    return new_test_set


def geo_mean_overflow(iterable):
    assert len(iterable) != 0
    a = np.log(iterable)
    return np.exp(a.sum() / len(a))


def get_speedups(seq_times: List[Tuple], par_times: List[Tuple]) -> List[float]:
    # input [(prob_name, time), (prob_name, time) ...]

    speedups = []
    # iterate over all instances:
    for instance in seq_times:
        instance_name = instance[0]
        seq_time = instance[1]

        par_time = None
        # get corresponding instance in par_times, if exists
        for par_instance in par_times:
            if par_instance[0] == instance_name:
                par_time = par_instance[1]

        if par_time is not None:
            speedups.append(float(seq_time) / par_time)

    return speedups


def create_plots(dist_data, speedups):
    def get_y_ticks(data, num_points=6):
        data = list(filter(lambda x: x > 0.0, data))
        lst = list(map(lambda x: truncate(x, 2) if x > 0.01 else x, np.geomspace(min(data), max(data), num_points)))
        lst.append(1.00)
        return sorted(lst)

    def truncate(number, decimals=0):
        """
        Returns a value truncated to a specific number of decimal places.
        """
        if not isinstance(decimals, int):
            raise TypeError("decimal places must be an integer.")
        elif decimals < 0:
            raise ValueError("decimal places has to be 0 or more.")
        elif decimals == 0:
            return math.trunc(number)

        factor = 10.0 ** decimals
        return math.trunc(number * factor) / factor

    fig = plt.figure()
    plt.style.use('bmh')

    ### Subplot A ###

    ax = fig.add_subplot(121)
    plt.text(0.5, 1.05, "(a)", transform=ax.transAxes)
    ys = []
    for algorithm in speedups:
        for machine in speedups[algorithm]:
            ys += speedups[algorithm][machine][1]
            plt.plot(np.arange(len(speedups[algorithm][machine][0])), speedups[algorithm][machine][1],
                     label=str(algorithm) + "-" + str(machine))

    plt.yscale('log')
    yticks = get_y_ticks(ys)
    plt.yticks(yticks, yticks)
    plt.tick_params(which='minor', left=False)
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend(fancybox=True, framealpha=0.5)

    ### Subplot B ###

    ax2 = fig.add_subplot(122)
    plt.text(0.5, 1.05, "(b)", transform=ax2.transAxes)

    # create x and y axis arrays
    ys = []
    for algorithm in dist_data:
        for machine in dist_data[algorithm]:
            ys += dist_data[algorithm][machine]
            plt.plot(np.arange(len(dist_data[algorithm][machine])), dist_data[algorithm][machine],
                     label=str(algorithm) + "-" + str(machine))

    plt.legend(fancybox=True, framealpha=0.5)
    plt.yscale('log')
    yticks = get_y_ticks(ys, 7)
    plt.yticks(yticks, yticks)
    plt.tick_params(which='minor', left=False)

    rc('font', **{'family': 'serif', 'serif': ['Times']})
    rc('text', usetex=True)
    ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.show()


# Main
if __name__ == "__main__":
    # regexes used to parse the log files
    results_regex = re.compile(result_pattern)

    # load config
    with open("plot_config.json") as json_file:
        data = json.load(json_file)
        log_files_data = data["log_files"]
        base_case_run = data["base_case"]

    ### building the main data struct ###

    test_sets = {}
    false_results = {}
    max_num_rounds = {}

    # For each log file, parse the contents and save the data in the main data struct
    for log_file in log_files_data:
        test_sets[log_file] = []
        false_results[log_file] = 0
        max_num_rounds[log_file] = 0
        with open(log_file, 'r') as f:
            results_file = f.read()

        # build results dict
        for g1 in re.finditer(results_regex, results_file):
            #  prob_file = str(g1.group('prob_file'))
            #  prob_name = prob_file.split("/")[-1]
            res_eq = str(g1.group('results_correct'))
            instace = {
                "nvars": int(g1.group('n_vars')),
                "ncons": int(g1.group('n_cons')),
                "nnz": int(g1.group('nnz')),
                "prob_name": str(g1.group('prob_file')).split("/")[-1],
                cpu_seq_time_key: float(g1.group('cpu_seq_time')),
                cpu_omp_time_key: float(g1.group('cpu_omp_time')),
              #  gpu_reduction_time_key: float(g1.group('gpu_reduction_time')),
                gpu_atomic_time_key: float(g1.group('gpu_atomic_time')),
                cpu_seq_rounds_key: int(g1.group('cpu_seq_rounds')),
                cpu_omp_rounds_key: int(g1.group('cpu_omp_rounds')),
            #    gpu_reduction_rounds_key: int(g1.group('gpu_reduction_rounds')),
                gpu_atomic_rounds_key: int(g1.group('gpu_atomic_rounds')),
                "res_eq": res_eq
            }

            # Only add the instance to the data struct if the results of all algorithms match
            if res_eq == "True" and instace[cpu_seq_rounds_key] < 100 and instace[cpu_omp_rounds_key] < 100 and instace[gpu_atomic_rounds_key] < 100:
                assert(instace[cpu_seq_time_key] >= 2)
                assert(instace[cpu_omp_time_key] >= 2)
                assert(instace[gpu_atomic_time_key] >= 2)
                test_sets[log_file].append(instace)
            elif instace[cpu_seq_rounds_key] == 100 or instace[cpu_omp_rounds_key] == 100 or instace[gpu_atomic_rounds_key] == 100:
                max_num_rounds[log_file]+=1
            elif res_eq == "False" and instace[cpu_seq_rounds_key] < 100 and instace[cpu_omp_rounds_key] < 100 and instace[gpu_atomic_rounds_key] < 100:
                false_results[log_file]+=1

    print("Finished parsing log files. Number of instances with correct/wrong results:")
    for log_file in test_sets:
        print(log_file, " correct: ", len(test_sets[log_file]), ", incorrect: ", false_results[log_file], ", max num rounds: ", max_num_rounds[log_file])

    print("Average number of rounds for ", log_file)
    for log_file in test_sets:
        print("cpu_seq:",       sum(map(lambda x: x[cpu_seq_rounds_key], test_sets[log_file])) / len(test_sets[log_file]),
              "cpu_omp:",       sum(map(lambda x: x[cpu_omp_rounds_key], test_sets[log_file])) / len(test_sets[log_file]),
#              "gpu_reduction:", sum(map(lambda x: x[gpu_reduction_rounds_key], test_sets[log_file])) / len(test_sets[log_file]),
              "gpu_atomic:",    sum(map(lambda x: x[gpu_atomic_rounds_key], test_sets[log_file])) / len(test_sets[log_file]))


    # remove small instances from the test set
    print("\nRemoving small instances from the test sets")
    test_sets = getsetovernvarscons(test_sets, 1000)

    ####  gather data for plotting sub-figure b) - speedup distributions ####

    # initi output data struct
    speedup_distros = {}
    for log_file in log_files_data:
        for alg in log_files_data[log_file]:
            speedup_distros[alg] = {}

    seq_base = list(map(lambda x: (x["prob_name"], x["cpu_seq_time"]), test_sets[base_case_run]))

    for log_file in log_files_data:
        for alg in log_files_data[log_file]:
            machine = log_files_data[log_file][alg]

            speedup_distros[alg][machine] = \
                sorted(
                    get_speedups(
                        seq_base,
                        list(map(lambda x: (x["prob_name"], x[machine_to_time[alg]]), test_sets[log_file]))
                    )

                )

    #### gather data for sub-figure a) - geometric means of speedups per test set

    # buckets for partitioning the set in 8 subsets
    buckets = [0, 10000, 20000, 40000, 80000, 160000, 320000, 640000, 10000000000000]
    speedups = {}

    for i in range(len(buckets) - 1):
        lb = buckets[i]
        ub = buckets[i + 1]
        redset = reduceset(test_sets, lb, ub)
        key = "Set-" + str(i + 1)

        # initi output data struct
        avg_speedups = {}
        for log_file in log_files_data:
            for alg in log_files_data[log_file]:
                avg_speedups[alg] = {}

        seq_base = list(map(lambda x: (x["prob_name"], x["cpu_seq_time"]), redset[base_case_run]))

        for log_file in log_files_data:
            for alg in log_files_data[log_file]:
                machine = log_files_data[log_file][alg]

                avg_speedups[alg][machine] = \
                    geo_mean_overflow(
                        get_speedups(
                            seq_base,
                            list(map(lambda x: (x["prob_name"], x[machine_to_time[alg]]), redset[log_file]))
                        )

                    )

        speedups[key] = avg_speedups

    speedups_plot_dict = {}

    subsets = speedups.keys()
    for subset in speedups.keys():

        for alg in speedups[subset].keys():
            if alg not in speedups_plot_dict.keys():
                speedups_plot_dict[alg] = {}
            for machine in speedups[subset][alg].keys():
                if machine not in speedups_plot_dict[alg].keys():
                    speedups_plot_dict[alg][machine] = ([], [])

                speedups_plot_dict[alg][machine][0].append(subset)
                speedups_plot_dict[alg][machine][1].append(speedups[subset][alg][machine])

    create_plots(speedup_distros, speedups_plot_dict)
