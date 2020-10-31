'''
	Joint Recognition of inner and outer boundary
'''

import argparse
from train_utils import *

if __name__ == '__main__':

	# parsing the command to format the dataset
    parser = argparse.ArgumentParser(description='Input among the following: train, test1, test2')