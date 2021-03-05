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
import pprint
import matplotlib.pylab as pylab
params = {'legend.fontsize': 'x-large',
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}
pylab.rcParams.update(params)

# add parent directory to path, need it to load some functions
currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.append(parentdir)

from regexes import result_pattern, papilo_output_pattern, papilo_found_more_changes_pattern

cpu_seq_key = "cpu_seq"
cpu_omp_key = "cpu_omp"
gpu_reduction_key = "gpu_reduction"
gpu_atomic_key = "gpu_atomic"
papilo_key = "papilo"

cpu_seq_time_key = "cpu_seq_time"
cpu_omp_time_key = "cpu_omp_time"
gpu_reduction_time_key = "gpu_reduction_time"
gpu_atomic_time_key = "gpu_atomic_time"
papilo_time_key = "papilo_time"

cpu_seq_rounds_key = "cpu_seq_rounds"
cpu_omp_rounds_key = "cpu_omp_rounds"
gpu_reduction_rounds_key = "gpu_reduction_rounds"
gpu_atomic_rounds_key = "gpu_atomic_rounds"
papilo_rounds_key = "papilo_rounds"

machine_to_time = {
    cpu_seq_key: cpu_seq_time_key,
    cpu_omp_key: cpu_omp_time_key,
    gpu_reduction_key: gpu_reduction_time_key,
    gpu_atomic_key: gpu_atomic_time_key,
    papilo_key: papilo_time_key
}


def get_linestyle(algorithm, machine):
    if 'fastmath' in machine:
        return 'dotted'
    elif algorithm == 'cpu_omp' or any(x in machine for x in ['single', 'gpu_loop', '8thrds']):
        return 'dashed'
    else:
        return 'solid'


def get_line_color(machine, algorithm):
    if algorithm == 'papilo':
        return 'tab:red'

    if 'V100' in machine:
        return 'tab:blue'
    elif 'TITAN' in machine:
        return 'tab:orange'
    elif 'P400' in machine or 'gpu_loop' in machine:
        return 'tab:red'
    elif 'RTXsuper' in machine:
        return 'tab:green'
    elif 'xeon' in machine:
        return 'tab:purple'
        #'tab:blue'
    elif 'amdtr' in machine:
        return 'tab:blue'
        #return 'tab:red'
    elif 'i7-9700K' in machine:
        return 'tab:pink'
       # return 'tab:green'
    else:
        raise Exception("Unknown machine: ", machine)


def print_stats(test_sets, false_results, max_num_rounds):
    print("Number of instances with correct/wrong results:")
    for log_file in test_sets:
        print(log_file, " correct: ", len(test_sets[log_file]), ", incorrect: ", false_results[log_file], ", max num rounds: ", max_num_rounds[log_file])


    for log_file in test_sets:

        cpu_seq_rounds = list(map(lambda x: x[cpu_seq_rounds_key], test_sets[log_file]))
        cpu_omp_rounds = list(map(lambda x: x[cpu_omp_rounds_key], test_sets[log_file]))
        #gpu_red_rounds = list(map(lambda x: x[gpu_reduction_rounds_key], test_sets[log_file]))
        gpu_ato_rounds = list(map(lambda x: x[gpu_atomic_rounds_key], test_sets[log_file]))
        print(log_file, ": ")
        print("Average number of rounds for cpu_seq:",      sum(cpu_seq_rounds) / len(test_sets[log_file]),
              "cpu_omp:",       sum(cpu_omp_rounds) / len(test_sets[log_file]),
              #              "gpu_reduction:", sum(gpu_red_rounds) / len(test_sets[log_file]),
              "gpu_atomic:",    sum(gpu_ato_rounds) / len(test_sets[log_file]))

        print("maximum increase factor: ", max([gpu_ato_rounds[i] / cpu_seq_rounds[i] for i in range(len(gpu_ato_rounds))]))


def print_stats_papilo(test_sets, max_num_rounds, false_results, additional_changes_found):
    print("Number of instances with correct/wrong results:")
    base = list(test_sets.keys())[0]
    for log_file in test_sets:
        print(log_file, " correct: ", len(test_sets[log_file]), ", incorrect: ", len(false_results[log_file]), ", max num rounds: ", len(max_num_rounds[log_file]), "additional changes found by papilo: ", len(additional_changes_found[log_file]))



        #print(base, " vs ", log_file, ": ", set(additional_changes_found[base]) == set(additional_changes_found[log_file]))
    # base = false_results["17_02_papilo_1thread_double_ada.log"]
    # rational = false_results["17_02_papilo_rationals_ada.log"]
    # diff_set = np.setdiff1d(rational,base)
    # print(diff_set)
    # for inst in diff_set:
    #     print(inst, " correct previously: ", inst in map(lambda x: x['prob_name'], test_sets["17_02_papilo_1thread_double_ada.log"]))



# Helper methods
def getsetovernvarscons(test_set, threashold):
    print("\nRemoving small instances from the test sets")
    new_test_set = {}
    for f in test_set.keys():
        data = list(filter(lambda x: x["nvars"] >= threashold or x["ncons"] >= threashold, test_set[f]))
        print("num instance for ", f, " with at least ", threashold, " cons or vars: ", len(data), ". Num removed instances: ", len(test_set[f]) - len(data))
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
    plot_a = True
    ### Subplot A ###
    if plot_a:
        ax = fig.add_subplot(111)
       # plt.text(0.5, 1.05, "(a)", transform=ax.transAxes, size='x-large')
        ys = []
        for algorithm in speedups:
            for machine in speedups[algorithm]:
                ys += speedups[algorithm][machine][1]
                plt.plot(np.arange(len(speedups[algorithm][machine][0])), speedups[algorithm][machine][1],
                         label=str(algorithm) + "-" + str(machine), linestyle=get_linestyle(algorithm, machine), color=get_line_color(machine, algorithm))

        plt.yscale('log')
        yticks = get_y_ticks(ys, 10)
        plt.yticks(yticks, yticks)
        plt.xticks(np.arange(len(speedups[algorithm][machine][0])), map(lambda x: "Set-" + str(x+1), np.arange(len(speedups[algorithm][machine][0]))))
        plt.tick_params(which='minor', left=False)
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        plt.legend(fancybox=True, framealpha=0.5)

    ### Subplot B ###
    plot_b = False
    if plot_b:
        ax2 = fig.add_subplot(122)
        plt.text(0.5, 1.05, "(b)", transform=ax2.transAxes, size='x-large')
        # create x and y axis arrays
        ys = []
        for algorithm in dist_data:
            for machine in dist_data[algorithm]:
                ys += dist_data[algorithm][machine]
                plt.plot(np.arange(len(dist_data[algorithm][machine])), dist_data[algorithm][machine],
                         label=str(algorithm) + "-" + str(machine), linestyle=get_linestyle(algorithm, machine), color=get_line_color(machine, algorithm))
          #  print("machine ", machine, " alg: ", algorithm, " min speedup: ", min(dist_data[algorithm][machine]))

       # plt.legend(fancybox=True, framealpha=0.5)
        plt.yscale('log')
        yticks = get_y_ticks(ys, 10)
        plt.yticks(yticks, yticks)
        plt.tick_params(which='minor', left=False)
        plt.minorticks_off()
        rc('font', **{'family': 'serif', 'serif': ['Times']})
        rc('text', usetex=True)
        ax2.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.show()


def parse_results_GDP(log_files_data):
    # regexes used to parse the log files
    results_regex = re.compile(result_pattern)
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

        cons = []
        vars = []
        zz = []

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
            cons.append(instace['ncons'])
            vars.append(instace['nvars'])
            zz.append(instace['nnz'])

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
    print(f"av cons {sum(cons)/len(cons)}, vars {sum(vars)/len(vars)}, nnz: {sum(zz)/len(zz)}")
    print("len: ", len(cons))
    return test_sets, max_num_rounds, false_results

def parse_results_papilo(log_files_data):
    # regexes used to parse the log files
    results_regex = re.compile(papilo_output_pattern)
    additional_changes_regex = re.compile(papilo_found_more_changes_pattern)

    test_sets = {}
    false_results = {}
    max_num_rounds = {}
    additional_changes = {}

    # For each log file, parse the contents and save the data in the main data struct
    for log_file in log_files_data:
        test_sets[log_file] = []
        additional_changes[log_file] = []
        false_results[log_file] = []
        max_num_rounds[log_file] = []
        with open(log_file, 'r') as f:
            results_file = f.read()

        # build results dict
        for g1 in re.finditer(results_regex, results_file):
            prob_name = str(g1.group('prob_file')).split("/")[-1]
            res_eq = str(g1.group('results_correct'))
            papilo_time = float(g1.group('papilo_time'))
            #papilo reports in seconds, we need nanoseconds:
            papilo_time = papilo_time * 1e9
            instace = {
                "nvars": int(g1.group('n_vars')),
                "ncons": int(g1.group('n_cons')),
                "nnz": int(g1.group('nnz')),
                "prob_name": str(g1.group('prob_file')).split("/")[-1],
                cpu_seq_time_key  : float(g1.group('cpu_seq_time')),
                cpu_omp_time_key  : float(g1.group('cpu_omp_time')),
                papilo_time_key   : papilo_time,
                cpu_seq_rounds_key: int(g1.group('cpu_seq_rounds')),
                cpu_omp_rounds_key: int(g1.group('cpu_omp_rounds')),
              #  papilo_rounds_key : int(g1.group('papilo_rounds')),
                "res_eq": res_eq
            }

            # Only add the instance to the data struct if the results of all algorithms match
            if res_eq == "True" and instace[cpu_seq_rounds_key] < 100 and instace[cpu_omp_rounds_key] < 100:
                assert(instace[cpu_seq_time_key] >= 2)
                assert(instace[cpu_omp_time_key] >= 2)
                test_sets[log_file].append(instace)
            elif instace[cpu_seq_rounds_key] == 100 or instace[cpu_omp_rounds_key] == 100:
                max_num_rounds[log_file].append(prob_name)
            elif res_eq == "False" and instace[cpu_seq_rounds_key] < 100 and instace[cpu_omp_rounds_key] < 100:
                false_results[log_file].append(prob_name)

        for g1 in re.finditer(additional_changes_regex, results_file):
            prob_name = str(g1.group('prob_file')).split("/")[-1]
            additional_changes[log_file].append(prob_name)

    return test_sets, max_num_rounds, false_results, additional_changes

def get_plots_data(log_files_data, test_sets):
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
        #print(key, " size: ")
        ##for key in redset.keys():
        #  print(len(redset[key]))

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

    # Printing data for speedup table
    print("\nData for the speedups table: ")
    for alg in speedups_plot_dict:
        for machine in speedups_plot_dict[alg]:
            print(alg, " ", machine, ":")
            print("    ", speedups_plot_dict[alg][machine][0])
            print("    ", speedups_plot_dict[alg][machine][1])
            print("     percentiles: 5%=", np.percentile(speedup_distros[alg][machine], 5), " 50%=", np.percentile(speedup_distros[alg][machine], 50), " 95%=", np.percentile(speedup_distros[alg][machine], 95))
            print("     average speedup: ", geo_mean_overflow(speedup_distros[alg][machine]))

    return speedup_distros, speedups_plot_dict

def process_data_GDP(log_files_data):
    ### building the main data struct ###
    test_sets, max_num_rounds, false_results = parse_results_GDP(log_files_data)
    print_stats(test_sets, max_num_rounds, false_results)
    # remove small instances from the test set
    test_sets = getsetovernvarscons(test_sets, 1000)

    speedup_distros, speedups_plot_dict = get_plots_data(log_files_data, test_sets)

    create_plots(speedup_distros, speedups_plot_dict)

def process_data_papilo(log_files_data):
    ### building the main data struct ###
    test_sets, max_num_rounds, false_results, additional_changes = parse_results_papilo(log_files_data)
    print_stats_papilo(test_sets, max_num_rounds, false_results, additional_changes)
    # remove small instances from the test set
   # test_sets = getsetovernvarscons(test_sets, 1000)

    speedup_distros, speedups_plot_dict = get_plots_data(log_files_data, test_sets)

    create_plots(speedup_distros, speedups_plot_dict)

# Main
if __name__ == "__main__":
    # load config
    with open("plot_config.json") as json_file:
        data = json.load(json_file)
        log_files_data = data["log_files"]
        base_case_run = data["base_case"]

    process_data_GDP(log_files_data)
   # process_data_papilo(log_files_data)




