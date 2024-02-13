import subprocess
import sys
import os
import shutil
import pandas as pd
import warnings 
import numpy as np
from scipy.stats import norm


def shell_do(command, print_cmd=True, log=False, return_log=False, err=False):
    if print_cmd:
        print(f'Executing: {(" ").join(command.split())}', file=sys.stderr)

    res=subprocess.run(command.split(), stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    output = res.stdout.decode('utf-8') + res.stderr.decode('utf-8')

    if log:
        print(output)
    if return_log:
        return output
    if err:
        return res.stderr.decode('utf-8')

def main():
    app_cmd = 'streamlit run Home.py'
    shell_do(app_cmd)

if __name__ == "__main__":
    main()