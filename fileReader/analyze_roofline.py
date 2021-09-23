import os
from matplotlib import pyplot as plt
from regexes import *

V100_peak_flops  = 7800.0 # GFLOPS
#V100_peak_flops = 15700.0 # GFLOPS
V100_peak_bandwidth = 900.0 # GB/s
V100_machine_balance = V100_peak_flops/V100_peak_bandwidth
print("V100 machine balance: ", V100_machine_balance)

# unit conversion dict to GB/s or s
units = {
    'B/s': 1e9,
    'KB/s': 1e6,
    'MB/s': 1e3,
    'GB/s': 1,
    'us': 1e6,
    'ms': 1e3,
    's':1
}


def frange(start, stop, step=1.0):
    f = start
    while f < stop:
        f += step
        yield f


def to_byte_per_s(val, unit):
    if unit == 'GB/s':
        return val * 1e9
    elif unit == 'MB/s':
        return val * 1e6
    elif unit == 'KB/s':
        return val * 1e3
    elif unit == 'B/s':
        return val
    else:
        raise Exception("Unknown unit ", unit)


def to_seconds(val, unit):
    if unit == 's':
        return val
    elif unit == 'ms':
        return val / 1e3
    elif unit == 'us':
        return val / 1e6


def plot_roofline():
    fig = plt.figure(frameon=False)
    ax = fig.add_subplot(1, 1, 1)
    yticks_labels = []
    yticks = []
    xticks_labels = []
    xticks = [2.**i for i in range(-6, 6)]
    ax.set_xlabel('arithmetic intensity [FLOP/byte]')
    ax.set_ylabel('performance [FLOP/s]')

    # Upper bound
    x = list(frange(min(xticks), max(xticks), 0.01))
    ax.plot(x, [min(V100_peak_bandwidth*x, float(V100_peak_flops)) for x in x])


    ax.set_xscale('log', basex=2)
    ax.set_yscale('log')
    ax.set_xlim(min(xticks), max(xticks))
    # ax.set_yticks([perf, float(max_flops)])
    ax.set_xticks(xticks)
    ax.grid(axis='x', alpha=0.7, linestyle='--')
    # fig.savefig('out.pdf')
    plt.show()


class RunData:
    def __init__(self):
        self.prob_name = None
        self.num_invocation = None
        self.flops = None
        self.DR = None
        self.TR = None
        self.TR_unit = None
        self.DW = None
        self.TW = None
        self.TW_unit = None
        self.runtime = None
        self.runtime_unit = None
        self.total_mem = None
        self.total_mem_unit = None

        self.AI = None
        self.attainable_perf = None
        self.achieve_perf = None
        self.achieved_perf_percent = None

    def print(self):
        print(self.prob_name, ":   invocations:", self.num_invocation, ", flops:", self.flops, ", DR:", self.DR, ", DR_unit:", self.DR_unit, ", DW:", self.DW, ", DW_unit", self.DW_unit)

    def compute_metrics(self):

        runtime_in_s = to_seconds(self.runtime, self.runtime_unit)

        data_movement = (to_byte_per_s(self.TR, self.TR_unit) + to_byte_per_s(self.TW, self.TW_unit)) * runtime_in_s
        self.AI = self.flops / data_movement
        #self.AI = self.flops / ((self.DR + self.DW)*32)

        self.attainable_perf = min(V100_peak_bandwidth * self.AI, float(V100_peak_flops)) # in GB/s
        self.achieve_perf = (self.flops/1e9) / runtime_in_s
        self.achieved_perf_percent = (self.achieve_perf / self.attainable_perf) * 100


def get_run_object(output: str) -> RunData:
    runData = RunData()
    runData.prob_name = get_regex_result(prob_name_pattern, output, "prob_file")
    runData.num_invocation = get_regex_result(roofline_flops_pattern, output, "invocations")
    runData.flops = int(float(get_regex_result(roofline_flops_pattern, output, "avg")))
    runData.DR = int(get_regex_result(roofline_DR_pattern, output, "avg"))
    runData.TR = float(get_regex_result(roofline_TR_pattern, output, "avg"))
    runData.TR_unit = get_regex_result(roofline_TR_pattern, output, "avg_unit").strip()
    runData.DW = int(get_regex_result(roofline_DW_pattern, output, "avg"))
    runData.TW = float(get_regex_result(roofline_TW_pattern, output, "avg"))
    runData.TW_unit = get_regex_result(roofline_TW_pattern, output, "avg_unit").strip()

    return runData


def add_runtime_data_to_run_objects(output: str, run_data_objects):
    prob_name = get_regex_result(prob_name_pattern, output, "prob_file")

    obj = [obj for obj in run_data_objects if obj.prob_name == prob_name]
    assert len(obj) == 1
    obj = obj[0]

    obj.runtime = float(get_regex_result(roofline_runtime_pattern, output, "avg"))
    obj.runtime_unit = get_regex_result(roofline_runtime_pattern, output, "avg_unit")


if __name__ == "__main__":

    throughput_log_file = "04_08_2021_gpu2_double_roofline.log"
    runtime_log_file = "05_08_2021_gpu2_roofline_runtimes_double.log"

    with open(os.path.join("plotter", throughput_log_file), 'r') as f:
       throughput_results_file = f.read()
    with open(os.path.join("plotter", runtime_log_file), 'r') as f:
        runtime_results_file = f.read()

    throughput_instances_output = list(map(lambda x: str(x.group()), re.finditer(re.compile(roofline_prob_out_pattern), throughput_results_file)))
    runtime_instances_output = list(map(lambda x: str(x.group()), re.finditer(re.compile(roofline_prob_out_pattern), runtime_results_file)))

    throughput_instances_output = list(filter(lambda output: get_regex_result(roofline_success_run, output) is not None, throughput_instances_output))
    runtime_instances_output = list(filter(lambda output: get_regex_result(roofline_success_run, output) is not None, runtime_instances_output))
    print("num instances after removing those with no results available: ", len(throughput_instances_output))

    throughput_instances_output = list(filter(lambda output: int(get_regex_result(nnz_pattern, output, 'nnz')) >2.5*1e5, throughput_instances_output))
    runtime_instances_output = list(filter(lambda output: int(get_regex_result(nnz_pattern, output, 'nnz')) >2.5*1e5, runtime_instances_output))
    print("num instances after removing those with nnz <= 1.5*1e5: ", len(throughput_instances_output))

    run_data_objects = list(map(lambda x: get_run_object(x), throughput_instances_output))
    assert len(run_data_objects) == len(throughput_instances_output)

    #add runtime data to objects

    for output in runtime_instances_output:
        add_runtime_data_to_run_objects(output, run_data_objects)

    run_data_objects = list(filter(lambda obj: obj.runtime is not None, run_data_objects))

    print("num isntances with both runtime and throughput info: ", len(run_data_objects))

    # run_data_objects = list(filter(lambda obj: obj.TR_unit == 'GB/s' and obj.TW_unit == 'GB/s', run_data_objects))
    # print("num isntances with GB/s throughput: ", len(run_data_objects))
    # highest mem throughput
    for obj in run_data_objects:
        obj.compute_metrics()

    AIs = list(map(lambda obj: obj.AI, run_data_objects))
    print("arithemtic intensity: min: ", min(AIs), ", max: ", max(AIs), ", avr: ", sum(AIs)/len(AIs))

    perf_percentages = list(map(lambda obj: obj.achieved_perf_percent, run_data_objects))
    print("achieved performance percentages: min: ", min(perf_percentages), ", max: ", max(perf_percentages), ", avr: ", sum(perf_percentages)/len(perf_percentages))



    # AI= calc_AI(flop,DR,DW)
    # print("Arithmetic intensity is: ", AI, ". Machine balance is: ", V100_machine_balance)
    #
    # achieved_perf = get_achieved_perf(flop,runtime)
    # attainable_perf = get_attainable_perf(AI)
    # print("achieved perf in percentages: ", get_achieved_per_in_percent_to_attainable(achieved_perf,attainable_perf))

