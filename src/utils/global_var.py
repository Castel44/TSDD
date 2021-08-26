import os
import shutil
import subprocess

######################################################################################################################

BASE_PATH = subprocess.check_output('git rev-parse --show-toplevel', shell=True).decode('ascii').split('\n', 1)[0]
OUTPATH = os.path.join(BASE_PATH, 'results')
DATAPATH = os.path.join(BASE_PATH, 'data')

columns = shutil.get_terminal_size().columns
