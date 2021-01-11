import os
import subprocess


class PapiloInterface():
    def __init__(self, papilo_path: str):
        self.papilo_binary = os.path.join(papilo_path, "build/bin/papilo")

    def run_papilo(self, input_file: str, output_file: str) -> None:
        args = f"{self.papilo_binary} -f {input_file} -r {output_file}"
        #p = subprocess.run(args, stdout=subprocess.PIPE)
        subprocess.call(args, shell=True, env=os.environ)