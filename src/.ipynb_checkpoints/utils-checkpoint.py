import os
import sys


def add_to_path(src_dir):
    for root, dirs, files in os.walk(src_dir):
        for dir in dirs:
            sys.path.insert(0, os.path.join(root, dir))