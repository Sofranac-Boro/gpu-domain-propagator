import os
import sys
import threading
import time
import re

result_pattern = r"Reding of  (?P<prob_file>.*)  model done!\nnum vars:  (?P<n_vars>\d*)\nnum cons:  (?P<n_cons>\d*)\nnnz     :  (?P<nnz>\d*)\n\n.*\n.*cpu_seq propagation done. Num rounds: (?P<cpu_seq_rounds>\d*)\ncpu_seq execution time : (?P<cpu_seq_time>\d*).*\n\n.*\n.*cpu_omp propagation done. Num rounds: (?P<cpu_omp_rounds>\d*)\ncpu_omp execution time : (?P<cpu_omp_time>\d*).*\n\n.*\n.*gpu_reduction propagation done. Num rounds: (?P<gpu_reduction_rounds>\d*)\ngpu_reduction execution time : (?P<gpu_reduction_time>\d*).*\n\n.*\n.*gpu_atomic propagation done. Num rounds: (?P<gpu_atomic_rounds>\d*)\ngpu_atomic execution time : (?P<gpu_atomic_time>\d*).*\n\n.*\n.*\n.*\nall results match:  (?P<results_correct>.*)"

seq_to_omp_pattern = r"cpu_seq to cpu_omp results match:  (?P<match>.*)"
seq_to_red_pattern = r"cpu_seq to gpu_reduction results match:  (?P<match>.*)"
seq_to_ato_pattern = r"cpu_seq to gpu_atomic results match:  (?P<match>.*)"
seq_to_dis_pattern = r"cpu_seq to cpu_seq_dis results match:  (?P<match>.*)"
prob_name_pattern = r"Reding of  (?P<prob_file>.*)  model done!"


# regexes for progress measure plotting
def with_measure_output_pattern(alg): return "(?s)====   Running the {} with measure  ====(.*?)====   end {} with measure  ====".format(alg, alg)
def without_measure_output_pattern(alg): return "(?s)====   Running the {} without measure  ====(.*?)====   end {} without measure  ====".format(alg, alg)
def num_rounds_pattern(alg): return "{} propagation done\. Num rounds: (?P<nrounds>\d+)".format(alg)
def round_measures_pattern(prop_round): return "round {} total score: (?P<score>\d+.\d+), k=(?P<k>\d+), n=(?P<n>\d+)".format(prop_round)
def round_timestamp_pattern(prop_round, alg): return "Propagation round: {}, {} execution time : (?P<timestamp>\d+) nanoseconds".format(prop_round, alg)


def get_regex_result(regex_string: str, search_string: str, group_name: str = None):
    m = re.compile(regex_string).search(search_string)

    if m is not None:
        return m.group(group_name) if group_name is not None else m.group()
    else:
        return None


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
