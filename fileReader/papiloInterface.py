import os
import subprocess
import tempfile
import shutil
import pathlib
from typing import List, Tuple, Union
from regexes import *
from readerInterface import FileReaderInterface, get_reader


def propagatePapilo(input_file_path: str)-> Tuple[List[float]]:
    papilo = PapiloInterface("/home/bzfsofra/papilo", input_file_path)
    papilo.run_papilo()
    return papilo.get_presolved_bounds()


class PapiloInterface():
    def __init__(self, papilo_path: str, input_file: str):
        self.input_file = input_file
        _, instance_name = os.path.split(input_file)
        self.instance_name = instance_name
        self.tmp_papilo_dir = tempfile.mkdtemp()

        #self.output_file = os.path.join(self.tmp_papilo_dir, "presolved" + self.instance_name)
        self.output_file = os.path.join("/home/bzfsofra/", "presolved" + self.instance_name)
        self.papilo_binary = os.path.join(papilo_path, "build/bin/papilo")
        self.parameters_file = os.path.join(pathlib.Path(__file__).parent.absolute(), "papilo_params.txt")

        self.num_rounds = None
        self.exec_time = None
        self.run_successful = False

    def __del__(self):
        shutil.rmtree(self.tmp_papilo_dir)

    def run_papilo(self) -> None:
        args = f"{self.papilo_binary} presolve -f {self.input_file} -r {self.output_file} -p {self.parameters_file}"
        print("command: " , args)
        p = subprocess.run(args.split(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)

        if get_regex_result(papilo_success_pattern, p.stdout, "time"):
            self.run_successful = True
            self.num_rounds = get_regex_result(papilo_results_pattern, p.stdout, "rounds")
            self.exec_time = get_regex_result(papilo_results_pattern, p.stdout, "time")
            print(p.stdout)
        else:
            print("Papilo run failed")

    def get_presolved_bounds(self) -> Union[Tuple[List[float]], None]:
        if not self.run_successful:
            return None
        reader: FileReaderInterface = get_reader(self.output_file)
        return reader.get_var_bounds()




