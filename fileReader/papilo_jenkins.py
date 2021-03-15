import os
import re
import unittest
from parameterized import parameterized
from run_propagation import papilo_comparison_run
from regexes import *

BASE_PATH = os.path.dirname(os.path.abspath(__file__))

MIPLIB_PATH = "/nfs/optimi/kombadon/IP/miplib2017"
#MIPLIB_PATH = "/home/bzfsofra/miplib2017"
PAPILO_PATH = "/home/optimi/bzfsofra/papilo"
#PAPILO_PATH = "/home/bzfsofra/papilo"
files = [f for f in os.listdir(MIPLIB_PATH) if os.path.isfile(os.path.join(MIPLIB_PATH, f))]

class TestGDP(unittest.TestCase):


    def run_instance_checck_result(self, instance_name: str) -> None:
        out = OutputGrabber()
        with out:
            papilo_comparison_run(os.path.join(MIPLIB_PATH, instance_name), PAPILO_PATH)

        print(out.capturedtext)

        seqpapilo_regex = re.compile(seq_to_papilo_pattern)
        nobdchgs_regex = re.compile(no_bdchgs_after_papilo_pattern)
        rounds_regex_cpu_seq = re.compile

        # papilo finds no additional bound changes:
        self.assertTrue(bool(nobdchgs_regex.search(out.capturedtext)))
        # cpu_seq finishes successfully.
        self.assertTrue(bool(re.compile("cpu_seq execution time :").search(out.capturedtext)))
        # papilo finishes successfully
        self.assertTrue(bool(re.compile("papilo execution time :").search(out.capturedtext)))
        # results match
        self.assertTrue(bool(seqpapilo_regex.search(out.capturedtext)))
        for g1 in re.finditer(seqpapilo_regex, out.capturedtext):
            self.assertEqual(str(g1.group('match')), 'True')

    @parameterized.expand(files[0:10])
    def test_papilo(self, input_file):
        print("\n=========== instance ", input_file, " ==========")
        self.run_instance_checck_result(input_file)
        print("=========== end instance ", input_file, " ==========")


if __name__ == '__main__':
    unittest.main()
