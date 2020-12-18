import os
import re
import sys
import threading
import time
import unittest

from plotter.plot_results import result_pattern, seq_to_omp_pattern, seq_to_red_pattern, seq_to_ato_pattern, seq_to_dis_pattern
from run_propagation import prop_compare_seq_gpu

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = r'test_data'


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


class TestGDP(unittest.TestCase):

    def run_instance_checck_result(self, instance_name: str) -> None:
        out = OutputGrabber()
        with out:
            prop_compare_seq_gpu(os.path.join(BASE_PATH, TEST_DATA_PATH, instance_name))

        seqomp_regex = re.compile(seq_to_omp_pattern)
        seqred_regex = re.compile(seq_to_red_pattern)
        seqato_regex = re.compile(seq_to_ato_pattern)
        self.assertTrue(
            bool(seqomp_regex.search(out.capturedtext)) and bool(seqred_regex.search(out.capturedtext)) and bool(
                seqato_regex.search(out.capturedtext)))
        for g1 in re.finditer(seqomp_regex, out.capturedtext):
            self.assertEqual(str(g1.group('match')), 'True')
        for g1 in re.finditer(seqred_regex, out.capturedtext):
            self.assertEqual(str(g1.group('match')), 'True')
        for g1 in re.finditer(seqato_regex, out.capturedtext):
            self.assertEqual(str(g1.group('match')), 'True')

        # comment out for release. only for internal testing

      #  seqdis_regex = re.compile(seq_to_dis_pattern)
      #  self.assertTrue(bool(seqdis_regex.search(out.capturedtext)))
      #  for g1 in re.finditer(seqdis_regex, out.capturedtext):
      #      self.assertEqual(str(g1.group('match')), 'True')

    def test_regex(self):
        input_file = r'drayage-25-27.mps.gz'
        out = OutputGrabber()
        with out:
            prop_compare_seq_gpu(os.path.join(BASE_PATH, TEST_DATA_PATH, input_file))
        self.assertTrue(bool(re.search(result_pattern, out.capturedtext)))

    def test_drayage_25_27(self):
        self.run_instance_checck_result('drayage-25-27.mps.gz')

    def test_b1c1s1(self):
        self.run_instance_checck_result('b1c1s1.mps.gz')

    def test_ip036(self):
        self.run_instance_checck_result('gen-ip036.mps.gz')

    def test_p0201(self):
        self.run_instance_checck_result('p0201.mps.gz')

    # temporarily disabled until the bug with atomic race condition in CSR-Vector is resolved
   # def test_ofi(self):
   #     self.run_instance_checck_result('ofi.mps.gz')

    def test_osorio_cta(self):
        self.run_instance_checck_result('osorio-cta.mps.gz')

    def test_reblock166(self):
        self.run_instance_checck_result('reblock166.mps.gz')

    def test_square37(self):
        self.run_instance_checck_result('square37.mps.gz')

    def test_traininstance6(self):
        self.run_instance_checck_result('traininstance6.mps.gz')

    def test_van(self):
        self.run_instance_checck_result('van.mps.gz')


if __name__ == '__main__':
    unittest.main()
