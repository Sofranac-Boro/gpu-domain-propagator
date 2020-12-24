import os
import re
import unittest
import time
import shutil

from run_propagation import prop_compare_seq_gpu, propagation_measure_run
from regexes import *

BASE_PATH = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_PATH = r'test_data'


class TestGDP(unittest.TestCase):
    def setUp(self):
        self.startTime = time.time()

    def tearDown(self):
        t = time.time() - self.startTime
        print('%s runtime: %.3f' % (self.id(), t))

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

    def test_progress_measure_with_regex(self):
        input_file = r'drayage-25-27.mps.gz'
        propagation_measure_run(os.path.join(BASE_PATH, TEST_DATA_PATH, input_file))

        self.assertTrue(os.path.exists(os.path.join(BASE_PATH, "progress_plots", "drayage-25-27.mps.gz.pdf")))
        shutil.rmtree(os.path.join(BASE_PATH, "progress_plots"))

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
