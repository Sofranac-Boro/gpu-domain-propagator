import json
import re
import os

import numpy as np
from scipy import interpolate
from typing import Dict, List

from utils import EPSGE, EPSLE, check_monotonic_increase, get_slope, EPSEQ, EPSGT, EPSLT, \
    num_instances_with_numerics_tag_in_miplib
from regexes import *
from plotter.plot_results import geo_mean_overflow

from matplotlib import pyplot as plt
from matplotlib import rc
from matplotlib.ticker import FormatStrFormatter

instance_output_pattern = "(?s)========== Starting measure executions for the  (?P<file>.*?)  file. ==========(.*?)========== End measure executions for the  (?P<endfile>.*?)  file. =========="


def remove_duplicates_take_first(scores: List, timestamps: List):
    i = 0
    while True:
        if i == len(scores) - 1:
            return
        if EPSEQ(scores[i], scores[i+1]):
            del scores[i+1]
            del timestamps[i+1]
        elif EPSGT(scores[i], scores[i+1]):
            raise Exception("error, decresing score")
        else:
            i += 1

class RunData():
    def __init__(self):
        self.prob_name = None
        self.cpu_data = {0: {"score": 0.0, "k": 0, "n":0, "timestamp": 0}}
        self.gpu_data = {0: {"score": 0.0, "k": 0, "n":0, "timestamp": 0}}
        self.cpu_num_rounds = 0
        self.gpu_num_rounds = 0
        self.max_k = 0

        self.cpu_scores_list = []
        self.cpu_timestamps_list = []
        self.cpu_k_list = []

        self.gpu_scores_list = []
        self.gpu_timestamps_list = []
        self.gpu_k_list = []

        self.cpu_fun = None
        self.gpu_fun = None
        self.cpu_k_fun = None
        self.gpu_k_fun = None

    def add_round(self, alg, round, score, k, n, timestamp):
        assert timestamp > 0
        assert k >= 0
        assert n >= 0
        assert EPSGE(score, 0.0) and EPSLE(score, 100.0)
        assert round > 0

        data = {"score": score, "k": k, "n":n, "timestamp": timestamp}
        if alg == "cpu_seq":
            assert round not in self.cpu_data.keys()
            self.cpu_data[round] = data
            self.cpu_num_rounds = max(round, self.cpu_num_rounds)
        elif alg == "gpu_atomic":
            assert round not in self.gpu_data.keys()
            self.gpu_data[round] = data
            self.gpu_num_rounds = max(round, self.gpu_num_rounds)
        else:
            raise Exception("Unknown alg: ", alg)

    def remove_second_to_last_round(self):
        # we don't compare runs with 1 round, they found no bound changes
        assert self.cpu_num_rounds > 1
        assert self.gpu_num_rounds > 1

        # remove second to last round. The progress should not have changed in these two rounds
        for key in ["score", "k", "n"]:
            assert self.cpu_data[self.cpu_num_rounds - 1][key] == self.cpu_data[self.cpu_num_rounds][key]
            assert self.gpu_data[self.gpu_num_rounds - 1][key] == self.gpu_data[self.gpu_num_rounds][key]

        self.cpu_data[self.cpu_num_rounds - 1] = self.cpu_data[self.cpu_num_rounds]
        self.gpu_data[self.gpu_num_rounds - 1] = self.gpu_data[self.gpu_num_rounds]
        del self.cpu_data[self.cpu_num_rounds]
        del self.gpu_data[self.gpu_num_rounds]
        self.cpu_num_rounds -= 1
        self.gpu_num_rounds -= 1

    def prepare_score_for_analysis(self):
        # the scores and times should now be both strictly monotonically increasing
        self.cpu_scores_list = list(map(lambda x: self.cpu_data[x]["score"], range(0, self.cpu_num_rounds + 1)))
        self.cpu_timestamps_list = list(map(lambda x: self.cpu_data[x]["timestamp"], range(0, self.cpu_num_rounds + 1)))
        self.gpu_scores_list = list(map(lambda x: self.gpu_data[x]["score"], range(0, self.gpu_num_rounds + 1)))
        self.gpu_timestamps_list = list(map(lambda x: self.gpu_data[x]["timestamp"], range(0, self.gpu_num_rounds + 1)))

        # if same score appears more than once, take the timestamp of the first appearence
        remove_duplicates_take_first(self.cpu_scores_list, self.cpu_timestamps_list)
        remove_duplicates_take_first(self.gpu_scores_list, self.gpu_timestamps_list)

        # non-decreasing
        check_monotonic_increase(self.cpu_scores_list)
        check_monotonic_increase(self.gpu_scores_list)
        check_monotonic_increase(self.cpu_timestamps_list)
        check_monotonic_increase(self.gpu_timestamps_list)

    def prepare_k_for_analysis(self):
        assert self.cpu_data[self.cpu_num_rounds]["k"] == self.max_k
        assert self.gpu_data[self.gpu_num_rounds]["k"] == self.max_k

        self.cpu_k_list = list(map(lambda x: self.cpu_data[x]["k"], range(0, self.cpu_num_rounds + 1)))
        self.cpu_timestamps_list = list(map(lambda x: self.cpu_data[x]["timestamp"], range(0, self.cpu_num_rounds + 1)))
        self.gpu_k_list = list(map(lambda x: self.gpu_data[x]["k"], range(0, self.gpu_num_rounds + 1)))
        self.gpu_timestamps_list = list(map(lambda x: self.gpu_data[x]["timestamp"], range(0, self.gpu_num_rounds + 1)))

        # normalize with max_k so that we have 0 to a 100 range.
        self.cpu_k_list = list(map(lambda x: x * 100.0 / self.max_k, self.cpu_k_list))
        self.gpu_k_list = list(map(lambda x: x * 100.0 / self.max_k, self.gpu_k_list))

        remove_duplicates_take_first(self.cpu_k_list, self.cpu_timestamps_list)
        remove_duplicates_take_first(self.gpu_k_list, self.gpu_timestamps_list)

        check_monotonic_increase(self.cpu_k_list)
        check_monotonic_increase(self.gpu_k_list)

    def last_scores_100(self):
        if np.isclose(self.cpu_data[self.cpu_num_rounds]["score"], 100.0) and np.isclose(self.gpu_data[self.gpu_num_rounds]["score"], 100.0):
            return True
        else:
            print(self.prob_name)
            return False

    def last_k_100(self):
        return np.isclose(self.cpu_k_list[-1], 100.0) and np.isclose(self.gpu_k_list[-1], 100.0)

    def get_speedup_final(self):
        cpu = self.cpu_data[self.cpu_num_rounds]["timestamp"]
        gpu = self.gpu_data[self.gpu_num_rounds]["timestamp"]
        assert not np.isclose(gpu, 0.0)
        return float(cpu) / float(gpu)

    def get_speedup_at(self, progress_val: int) -> float:
        if self.cpu_fun is None or self.gpu_fun is None:
            self.cpu_fun = interpolate.interp1d(self.cpu_scores_list, self.cpu_timestamps_list, fill_value="extrapolate", bounds_error=False)
            self.gpu_fun = interpolate.interp1d(self.gpu_scores_list, self.gpu_timestamps_list, fill_value="extrapolate", bounds_error=False)
        cpu_timestamp = self.cpu_fun(progress_val)
        gpu_timestamp = self.gpu_fun(progress_val)
        assert not np.isclose(gpu_timestamp, 0.0)
        return float(cpu_timestamp) / float(gpu_timestamp)

    def get_k_speedup_at(self, progress_val: int) -> float:
        if self.cpu_k_fun is None or self.gpu_k_fun is None:
            self.cpu_k_fun = interpolate.interp1d(self.cpu_k_list, self.cpu_timestamps_list, fill_value="extrapolate", bounds_error=False)
            self.gpu_k_fun = interpolate.interp1d(self.gpu_k_list, self.gpu_timestamps_list, fill_value="extrapolate", bounds_error=False)
        cpu_timestamp = self.cpu_k_fun(progress_val)
        gpu_timestamp = self.gpu_k_fun(progress_val)
        assert not np.isclose(gpu_timestamp, 0.0)
        return float(cpu_timestamp) / float(gpu_timestamp)

    def check_finite_progress(self):
        if np.isclose(self.cpu_data[self.cpu_num_rounds]["score"], 0.0):
            assert np.isclose(self.gpu_data[self.gpu_num_rounds]["score"], 0.0)
            # at least one infinite change. otherwise there is no finite nor infinite progress. This case should've been already handled
            assert self.cpu_data[self.cpu_num_rounds]["k"] > 0
            assert self.gpu_data[self.gpu_num_rounds]["k"] > 0
            return False
        else:
            return True

    def check_infinite_progress(self):
        if self.cpu_data[self.cpu_num_rounds]["k"] == 0:
            assert self.gpu_data[self.gpu_num_rounds]["k"] == 0
            # at least one finite change, otherwise there is no finite nor infinite progress. This case should've been already handled
            assert np.isclose(self.cpu_data[self.cpu_num_rounds]["score"], 100.0)
            assert np.isclose(self.gpu_data[self.gpu_num_rounds]["score"], 100.0)
            return False
        else:
            return True

    def get_average_cpu_slope(self):
        cpu_slopes = list(map(lambda i: get_slope(self.cpu_timestamps_list[i], self.cpu_timestamps_list[i+1], self.cpu_scores_list[i], self.cpu_scores_list[i+1]), range(0, self.cpu_num_rounds)))
        assert len(cpu_slopes) != 0
        return np.mean(cpu_slopes)

    def get_average_gpu_slope(self):
        gpu_slopes = list(map(lambda i: get_slope(self.gpu_timestamps_list[i], self.gpu_timestamps_list[i+1], self.gpu_scores_list[i], self.gpu_scores_list[i+1]), range(0, self.gpu_num_rounds)))
        assert len(gpu_slopes) != 0
        return np.mean(gpu_slopes)

    def stalling_cpu(self,p,q):
        scores = list(map(lambda x: self.cpu_data[x]["score"], range(0, self.cpu_num_rounds + 1)))
        timestamps = list(map(lambda x: self.cpu_data[x]["timestamp"], range(0, self.cpu_num_rounds + 1)))
        last_timestamp = timestamps[-1]
        # normalize timestamps
        timestamps = list(map(lambda x: x * 100 / last_timestamp, timestamps ))

        dy_1 = np.gradient(scores, timestamps)
        dy_2 = np.gradient(dy_1, timestamps)

        assert np.all(EPSGE(dy_1, 0.0)) # progress can't decrease
        assert len(scores) == len(timestamps) == self.cpu_num_rounds + 1
        assert EPSEQ(timestamps[-1], 100.0)
        assert EPSEQ(scores[-1], 100.0)

        for round in range(1, self.cpu_num_rounds):

            inf_reduction = False if self.cpu_data[round]["k"] == 0 else self.cpu_data[round]["k"] > self.cpu_data[round-1]["k"]

            # if there is small progress: f' < p and no infinite tightenings
            if EPSLT(dy_1[round], p) and not inf_reduction:

                # if there is increase in progress speed thereafter: f'' > q
                if np.any(EPSGT(np.array(dy_2[round+1:]), q)):
                   # print(self.prob_name, " stalling in round ", round)
                    #print(scores)
                    #print(timestamps)
                    return True
        return False

    def stalling_gpu(self,p,q):
        scores = list(map(lambda x: self.gpu_data[x]["score"], range(0, self.gpu_num_rounds + 1)))
        timestamps = list(map(lambda x: self.gpu_data[x]["timestamp"], range(0, self.gpu_num_rounds + 1)))
        last_timestamp = timestamps[-1]
        # normalize timestamps
        timestamps = list(map(lambda x: x * 100 / last_timestamp, timestamps ))

        dy_1 = np.gradient(scores, timestamps)
        dy_2 = np.gradient(dy_1, timestamps)

        assert np.all(EPSGE(dy_1, 0.0)) # progress can't decrease
        assert len(scores) == len(timestamps) == self.gpu_num_rounds + 1
        assert EPSEQ(timestamps[-1], 100.0)
        assert EPSEQ(scores[-1], 100.0, 1e-3)

        for round in range(1, self.gpu_num_rounds):

            inf_reduction = False if self.gpu_data[round]["k"] == 0 else self.gpu_data[round]["k"] > self.gpu_data[round-1]["k"]

            # if there is small progress: f' < p, score < 95 and no infinite tightenings
            if EPSLT(dy_1[round], p) and not inf_reduction:

                # if there is increase in progress speed thereafter: f'' > q
                if np.any(EPSGT(np.array(dy_2[round+1:]), q)):
                    # print(self.prob_name, " stalling in round ", round)
                    #print(scores)
                    #print(timestamps)
                    return True
        return False



def get_run_object(output: str) -> RunData:
    runData = RunData()
    runData.prob_name = get_regex_result(prob_name_pattern, output, "prob_file")
    for alg in ["cpu_seq", "gpu_atomic"]:
        with_measure_output = get_regex_result(with_measure_output_pattern(alg), output)
        without_measure_output = get_regex_result(without_measure_output_pattern(alg), output)

        num_rounds = int(get_regex_result(num_rounds_pattern(alg), with_measure_output, "nrounds"))
        if (num_rounds != int(get_regex_result(num_rounds_pattern(alg), without_measure_output, "nrounds"))):
            print(runData.prob_name)
        assert(num_rounds == int(get_regex_result(num_rounds_pattern(alg), without_measure_output, "nrounds")))

        max_ks = get_all_regex_result(max_measure_pattern, output, "k")
        max_ks = list(map(lambda x: int(x), max_ks))
        # make sure the same k was found in cpu and gpu alg in the end
        assert(max_ks.count(max_ks[0]) == len(max_ks))
        runData.max_k = max_ks[0]

        for prop_round in range(1, num_rounds+1):
            score = float(get_regex_result(round_measures_pattern(prop_round), with_measure_output, "score"))
            k = int(get_regex_result(round_measures_pattern(prop_round), with_measure_output, "k"))
            n = int(get_regex_result(round_measures_pattern(prop_round), with_measure_output, "n"))
            timestamp = int(get_regex_result(round_timestamp_pattern(prop_round, alg), without_measure_output, "timestamp"))
            runData.add_round(alg, prop_round, score, k, n, timestamp)
    return runData


def parse_results_progress_run(results_file):
    def no_max_rounds(instance_output):
        cpu_seq_output = get_regex_result(without_measure_output_pattern("cpu_seq"), instance_output)
        gpu_ato_output = get_regex_result(without_measure_output_pattern("gpu_atomic"), instance_output)

        cpu_rounds = int(get_regex_result(num_rounds_pattern("cpu_seq"), cpu_seq_output, "nrounds"))
        atomic_rounds = int(get_regex_result(num_rounds_pattern("gpu_atomic"), gpu_ato_output, "nrounds"))

        return cpu_rounds < 100 and atomic_rounds < 100

    def no_no_output(instance_output):
        cpu_seq_output = get_regex_result(without_measure_output_pattern("cpu_seq"), instance_output)
        gpu_ato_output = get_regex_result(without_measure_output_pattern("gpu_atomic"), instance_output)

        # e.g. nothing to propagate, 0 size problem. Results will match but should not be used
        if cpu_seq_output is None or gpu_ato_output is None:
            return False
        else:
            return True

    def no_one_round(instance_output):
        cpu_seq_output = get_regex_result(without_measure_output_pattern("cpu_seq"), instance_output)
        gpu_ato_output = get_regex_result(without_measure_output_pattern("gpu_atomic"), instance_output)

        cpu_rounds = int(get_regex_result(num_rounds_pattern("cpu_seq"), cpu_seq_output, "nrounds"))
        atomic_rounds = int(get_regex_result(num_rounds_pattern("gpu_atomic"), gpu_ato_output, "nrounds"))

        return cpu_rounds > 1 or atomic_rounds > 1

    # put output of every instance in a list
    instances_output = list(map(lambda x: str(x.group()), re.finditer(re.compile(instance_output_pattern), results_file)))
    print("num intances: ", len(instances_output))

    instances_output = list(filter(no_no_output, instances_output))
    print("num instances after removing those with no results available: ", len(instances_output))

    #non_matching = list(filter(lambda output: get_regex_result(seq_to_ato_pattern, output, 'match') == "False", instances_output))
    #non_matching = list(map(lambda output: get_regex_result(prob_name_pattern, output, "prob_file").split("/")[-1], non_matching))
    #num_instances_with_numerics_tag_in_miplib(non_matching)

    # remove instances with non-mathcing results
    instances_output = list(filter(lambda output: get_regex_result(seq_to_ato_pattern, output, 'match') == "True", instances_output))
    print("num intances after removing those with non-matching reuslts: ", len(instances_output))

    instances_output = list(filter(no_max_rounds, instances_output))
    print("num instances after removing those with max num rounds reached: ", len(instances_output))

    instances_output = list(filter(no_one_round, instances_output))
    print("num instances after removing those with no bound changes found (1 propagation rounds): ", len(instances_output))

    # list of instances with differing num rounds for with and without measure runs. Probably some weird numerical trouble. Remove these form the test set
    rem = ["momentum1", "neos-4335793-snake", 'neos-5273874-yomtsa', 'ns2034125']
    instances_output = list(filter(lambda output: get_regex_result(prob_name_pattern, output, "prob_file").split("/")[-1] not in rem, instances_output))
    print("num instances after manual removal (numerical difficulties): ", len(instances_output))

    # FINITE PROGRESS
    finite_run_data_objects = list(map(lambda x: get_run_object(x), instances_output))
    assert len(finite_run_data_objects) == len(instances_output)

    # remove instances with only infinite progress
    finite_run_data_objects = list(filter(lambda x: x.check_finite_progress(), finite_run_data_objects))
    print("num instances after removing instances with only infinite progress: ", len(finite_run_data_objects))

    # remove insance where the last score is not 100
    finite_run_data_objects = list(filter(lambda x: x.last_scores_100(), finite_run_data_objects))
    print("num instances after removing instances with last score not equal to 100: ", len(finite_run_data_objects))


    for obj in finite_run_data_objects:
        obj.remove_second_to_last_round()
        obj.prepare_score_for_analysis()


    pq = [(1e20, 0.0), (0.1, 0.0), (0.1, 0.2), (0.1, 0.5), (0.5, 0.5), (0.5, 2.0)]
    print("\nStalling test set: ", len(finite_run_data_objects), " instances.")
    for p, q in pq:
        num_stalls_cpu = 0
        num_stalls_gpu = 0
        for obj in finite_run_data_objects:
            if obj.stalling_cpu(p, q):
                num_stalls_cpu+=1
            if obj.stalling_gpu(p, q):
                num_stalls_gpu+=1
        print("Number of stalls for p=",p," and q=", q, ":  cpu: ", num_stalls_cpu, ", gpu: ", num_stalls_gpu)

    for val in range(10,110, 10):
        print("Geo mean speedup at ", val, "% : ", geo_mean_overflow(list(map(lambda x: x.get_speedup_at(val), finite_run_data_objects))))
    print("Geo mean speedup at ", 100, "% : ",geo_mean_overflow(list(map(lambda x: x.get_speedup_at(100), finite_run_data_objects))))
    print("Geo mean speedup at the end ", geo_mean_overflow(list(map(lambda x: x.get_speedup_final(), finite_run_data_objects))))


    # INFINITE PROGRESS
    #remove instances with only finite progress
    infinite_run_data_objects = list(map(lambda x: get_run_object(x), instances_output))
    infinite_run_data_objects = list(filter(lambda x: x.check_infinite_progress(), infinite_run_data_objects))
    print("num instances after removing instances with only finite progress: ", len(infinite_run_data_objects))

    for obj in infinite_run_data_objects:
        obj.remove_second_to_last_round()
        obj.prepare_k_for_analysis()

    infinite_run_data_objects = list(filter(lambda x: x.last_k_100(), infinite_run_data_objects))
    print("num instances after removing instances where last k is not 100: ", len(infinite_run_data_objects))
    #
    # for val in range(10,110,10):
    #     print("Geo mean k speedup at ", val, "% : ", geo_mean_overflow(list(map(lambda x: x.get_k_speedup_at(val), infinite_run_data_objects))))
    print("Geo mean k speedup at ", 100, "% : ", geo_mean_overflow(list(map(lambda x: x.get_speedup_final(), infinite_run_data_objects))))

  # plotting
    fig = plt.figure()
    plt.style.use('bmh')
    rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman'], 'size': 10})
    rc('text', usetex=True)
    ax = fig.add_subplot(111)
    x = list(np.linspace(10, 99, num=100, endpoint=False))
    x = x + list(np.linspace(99, 100, num=20, endpoint=True))
    #x = list(np.linspace(10, 100, num=100, endpoint=False))
    s = []
    k = []

    for val in x:
        s.append(geo_mean_overflow(list(map(lambda x: x.get_speedup_at(val), finite_run_data_objects))))
        k.append(geo_mean_overflow(list(map(lambda x: x.get_k_speedup_at(val), infinite_run_data_objects))))

    plt.plot(x, s, label="finite domain reductions mean speedup",color='tab:blue')
    plt.plot(x, k, label="infinite domain reductions mean speedup",color='tab:red', linestyle='dashed')

    yticks = list(np.linspace(min(min(s), min(k)), max(max(s), max(k)), 10))
    xticks = list(range(10,110,10))
    plt.yticks(yticks, yticks)
    plt.xticks(xticks, xticks)
    plt.xlabel("progress (\%)")
    plt.ylabel("speedup")
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    plt.legend(fancybox=True, framealpha=0.5)
    plt.show()



if __name__ == "__main__":
    # load config
    with open("plotter/plot_config.json") as json_file:
        data = json.load(json_file)
        log_files_data = data["log_files"]
        base_case_run = data["base_case"]

    for log_file in log_files_data:
        with open(os.path.join("plotter", log_file), 'r') as f:
            results_file = f.read()
        print("log file: ", log_file)
        parse_results_progress_run(results_file)