import os
import sys
import threading
import time
import re

nnz_pattern = r"nnz     :  (?P<nnz>\d+)\n"

#result_pattern = r"Reding of  (?P<prob_file>.*)  model done!\nnum vars:  (?P<n_vars>\d*)\nnum cons:  (?P<n_cons>\d*)\nnnz     :  (?P<nnz>\d*)\n\n.*\n.*cpu_seq propagation done. Num rounds: (?P<cpu_seq_rounds>\d*)\ncpu_seq execution time : (?P<cpu_seq_time>\d*).*\n\n.*\n.*cpu_omp propagation done. Num rounds: (?P<cpu_omp_rounds>\d*)\ncpu_omp execution time : (?P<cpu_omp_time>\d*).*\n\n.*\n.*gpu_reduction propagation done. Num rounds: (?P<gpu_reduction_rounds>\d*)\ngpu_reduction execution time : (?P<gpu_reduction_time>\d*).*\n\n.*\n.*gpu_atomic propagation done. Num rounds: (?P<gpu_atomic_rounds>\d*)\ngpu_atomic execution time : (?P<gpu_atomic_time>\d*).*\n\n.*\n.*\n.*\nall results match:  (?P<results_correct>.*)"
result_pattern = r"Reading of  (?P<prob_file>.*)  model done!\nnum vars:  (?P<n_vars>\d*)\nnum cons:  (?P<n_cons>\d*)\nnnz     :  (?P<nnz>\d*)\n\n.*\n.*cpu_seq propagation done. Num rounds: (?P<cpu_seq_rounds>\d*)\ncpu_seq execution time : (?P<cpu_seq_time>\d*).*\n\n.*\n.*cpu_omp propagation done. Num rounds: (?P<cpu_omp_rounds>\d*)\ncpu_omp execution time : (?P<cpu_omp_time>\d*).*\n\n.*\n.*gpu_atomic propagation done. Num rounds: (?P<gpu_atomic_rounds>\d*)\ngpu_atomic execution time : (?P<gpu_atomic_time>\d*).*\n\ncpu_seq to cpu_omp results match:  (?P<dsadasdas>.*)\ncpu_seq to gpu_atomic results match:  (?P<dadadas>.*)\nall results match:  (?P<results_correct>.*)"
seq_to_omp_pattern = r"cpu_seq to cpu_omp results match:  (?P<match>.*)"
seq_to_papilo_pattern = r"cpu_seq to pailo results match:  (?P<match>.*)"
seq_to_red_pattern = r"cpu_seq to gpu_reduction results match:  (?P<match>.*)"
seq_to_ato_pattern = r"cpu_seq to gpu_atomic results match:  (?P<match>.*)"
seq_to_dis_pattern = r"cpu_seq to cpu_seq_dis results match:  (?P<match>.*)"
prob_name_pattern = r"Reading of  (?P<prob_file>.*)  model done!"

# regexes for progress measure plotting
max_measure_pattern = "Maximum measure: score=(?P<score>\d+.\d+), k=(?P<k>\d+)"
def with_measure_output_pattern(alg): return "(?s)====   Running the {} with measure  ====(.*?)====   end {} with measure  ====".format(alg, alg)
def without_measure_output_pattern(alg): return "(?s)====   Running the {} without measure  ====(.*?)====   end {} without measure  ====".format(alg, alg)
def num_rounds_pattern(alg): return "{} propagation done\. Num rounds: (?P<nrounds>\d+)".format(alg)
def round_measures_pattern(prop_round): return "round {} total score: (?P<score>\d+.\d+), k=(?P<k>\d+), n=(?P<n>\d+)".format(prop_round)
def round_timestamp_pattern(prop_round, alg): return "Propagation round: {}, {} execution time : (?P<timestamp>\d+) nanoseconds".format(prop_round, alg)


# PaPILO
papilo_results_pattern = r"        propagation            (?P<rounds>\d+)               (?P<b>\d+.\d+)               (?P<c>\d+)              (?P<d>\d+.\d+)              (?P<time>\d+.\d+)"
papilo_success_pattern = r"presolving finished after (?P<time>\d+.\d+) seconds"
papilo_solve_stats = r"presolved[ \t]+(?P<rounds>\d+) rounds:[ \t]+(?P<del_cols>\d+) del cols,[ \t]+(?P<del_rows>\d+) del rows,[ \t]+(?P<chg_bounds>\d+) chg bounds,[ \t]+(?P<chg_sides>\d+) chg sides,[ \t]+(?P<chg_coeffs>\d+) chg coeffs,[ \t]+(?P<tsx_applied>\d+) tsx applied,[ \t]+(?P<tsx_conflicts>\d+) tsx conflicts"

papilo_output_pattern = r"Reading of  (?P<prob_file>.*)  model done!\nnum vars:  (?P<n_vars>\d*)\nnum cons:  (?P<n_cons>\d*)\nnnz     :  (?P<nnz>\d*)\n\n.*\n.*cpu_omp propagation done. Num rounds: (?P<cpu_omp_rounds>\d*)\ncpu_omp execution time : (?P<cpu_omp_time>\d*).*\n\n.*\n.*cpu_seq propagation done. Num rounds: (?P<cpu_seq_rounds>\d*)\ncpu_seq execution time : (?P<cpu_seq_time>\d*) nanoseconds\n\nRunning papilo after cpu_seq...\npapilo run successful!\npapilo did not find any bound changes after cpu_seq!\n\npapilo execution start...\npapilo run successful!\npapilo propagation done. Num rounds:  (?P<papilo_rounds>.*)\npapilo execution time :  (?P<papilo_time>\d*.\d+)  seconds\ncpu_seq to cpu_omp results match:  (?P<seq_omp_match>.*)\ncpu_seq to pailo results match:  (?P<seq_papilo_match>.*)\nall results match:  (?P<results_correct>.*)"
papilo_found_more_changes_pattern = r"execution of  (?P<prob_file>.*)  failed\. Exception: \npapilo found additional bound changes after GDP\.\n\*\*\* print_tb:\n  File \"run_propagation\.py\", line 209, in <module>\n    papilo_comparison_run\(args\.file, datatype\)\n  File \"run_propagation\.py\", line 124, in papilo_comparison_run\n    \(seq_new_lbs, seq_new_ubs, stdout\) = propagateSequentialWithPapiloPostsolve\(reader, n_vars, n_cons, nnz, col_indices, row_ptrs, coeffs, lhss, rhss,\n  File \"\/home\/bzfsofra\/ada_tests\/gpu-domain-propagator\/fileReader\/GPUDomPropInterface\.py\", line 277, in propagateSequentialWithPapiloPostsolve\n    raise Exception\(\"papilo found additional bound changes after GDP\.\"\)"

no_bdchgs_after_papilo_pattern = r"papilo did not find any bound changes after cpu_seq!"


# Roofline analysis regexes
roofline_prob_out_pattern = r"(?s)read with 0 errors\n(.*?)Reding lp file"
roofline_flops_pattern = r"[ ]+(?P<invocations>\d*)[ ]+flop_count_dp[ ]+Floating Point Operations\(Double Precision\)[ ]+(?P<min>\d+.\d+e\+\d+|\d+)[ ]+(?P<max>\d+.\d+e\+\d+|\d+)[ ]+(?P<avg>\d+.\d+e\+\d+|\d+)\n"
roofline_flops_float_pattern = r"[ ]+(?P<invocations>\d*)[ ]+flop_count_sp[ ]+Floating Point Operations\(Single Precision\)[ ]+(?P<min>\d+.\d+e\+\d+|\d+)[ ]+(?P<max>\d+.\d+e\+\d+|\d+)[ ]+(?P<avg>\d+.\d+e\+\d+|\d+)\n"
roofline_TR_pattern = r"[ ]+(?P<invocations>\d*)[ ]+dram_read_throughput[ ]+Device Memory Read Throughput[ ]+(?P<min>\d+.\d+)(?P<min_unit>.*)[ ]+(?P<max>\d+.\d+)(?P<max_unit>.*)[ ]+(?P<avg>\d+.\d+)(?P<avg_unit>.*)\n"
roofline_TW_pattern = r"[ ]+(?P<invocations>\d*)[ ]+dram_write_throughput[ ]+Device Memory Write Throughput[ ]+(?P<min>\d+.\d+)(?P<min_unit>.*)[ ]+(?P<max>\d+.\d+)(?P<max_unit>.*)[ ]+(?P<avg>\d+.\d+)(?P<avg_unit>.*)\n"
roofline_DR_pattern = r"[ ]+(?P<invocations>\d*)[ ]+dram_read_transactions[ ]+Device Memory Read Transactions[ ]+(?P<min>\d+)[ ]+(?P<max>\d+)[ ]+(?P<avg>\d+)\n"
roofline_DW_pattern = r"[ ]+(?P<invocations>\d*)[ ]+dram_write_transactions[ ]+Device Memory Write Transactions[ ]+(?P<min>\d+)[ ]+(?P<max>\d+)[ ]+(?P<avg>\d+)\n"
roofline_success_run = r"gpu_atomic propagation done."
roofline_runtime_pattern = r" GPU activities:[ ]+\d+.\d+\%[ ]+\d+.\d+[a-z]+[ ]+\d+[ ]+(?P<avg>\d+.\d+)(?P<avg_unit>[a-z]+)[ ]+(?P<min>\d+.\d+)(?P<min_unit>[a-z]+)[ ]+(?P<max>\d+.\d+)(?P<max_unit>[a-z]+)[ ]+void GPUAtomicDomainPropagation"

def get_regex_result(regex_string: str, search_string: str, group_name: str = None):
    m = re.compile(regex_string).search(search_string)

    if m is not None:
        return m.group(group_name) if group_name is not None else m.group()
    else:
        return None


def get_all_regex_result(regex_string: str, search_string: str, group_name: str = None):
    def fun(m):
        if m is not None:
            return m.group(group_name) if group_name is not None else m.group()
        else:
            return None

    ms = re.compile(regex_string).finditer(search_string)
    return list(map(fun, ms))


class OutputGrabber(object):
    """
    Class used to grab standard output or another stream.
    """
    escape_char = "\b"

    def __init__(self, stream=None, threaded=False):
        self.origstream = stream
        self.threaded = threaded
        if self.origstream is None:
            self.origstream = sys.stdout
        self.origstreamfd = self.origstream.fileno()
        self.capturedtext = ""
        # Create a pipe so the stream can be captured:
        self.pipe_out, self.pipe_in = os.pipe()

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def start(self):
        """
        Start capturing the stream data.
        """
        self.capturedtext = ""
        # Save a copy of the stream:
        self.streamfd = os.dup(self.origstreamfd)
        # Replace the original stream with our write pipe:
        os.dup2(self.pipe_in, self.origstreamfd)
        if self.threaded:
            # Start thread that will read the stream:
            self.workerThread = threading.Thread(target=self.readOutput)
            self.workerThread.start()
            # Make sure that the thread is running and os.read() has executed:
            time.sleep(0.01)

    def stop(self):
        """
        Stop capturing the stream data and save the text in `capturedtext`.
        """
        # Print the escape character to make the readOutput method stop:
        self.origstream.write(self.escape_char)
        # Flush the stream to make sure all our data goes in before
        # the escape character:
        self.origstream.flush()
        if self.threaded:
            # wait until the thread finishes so we are sure that
            # we have until the last character:
            self.workerThread.join()
        else:
            self.readOutput()
        # Close the pipe:
        os.close(self.pipe_in)
        os.close(self.pipe_out)
        # Restore the original stream:
        os.dup2(self.streamfd, self.origstreamfd)
        # Close the duplicate stream:
        os.close(self.streamfd)

    def readOutput(self):
        """
        Read the stream data (one byte at a time)
        and save the text in `capturedtext`.
        """
        while True:
            char = os.read(self.pipe_out, 1).decode(self.origstream.encoding)
            if not char or self.escape_char in char:
                break
            self.capturedtext += char
