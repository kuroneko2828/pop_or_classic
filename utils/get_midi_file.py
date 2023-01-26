import glob
import re
import joblib


def get_pop_files(POP909_path):
    files = []
    dirs = glob.glob(POP909_path+'POP909/*/')
    for dir_ in dirs:
        no = re.match(r'.*?/(\d+?)/$', dir_)
        if no is not None:
            files.append(dir_+no.group(1)+'.mid')
    return files


def get_classic_files(maestro_path):
    files = glob.glob(maestro_path+'*/*.midi')
    return files