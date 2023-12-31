import sys
import pandas as pd
import argparse
import mean
from mean.locate_neurons import extract_ROI
from mean.gaussian_kernel_convolution import gaussian_kernel_convolution
parser = argparse.ArgumentParser(description="Arguments",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-c1", "--condition1", type=str, help="File path to condition 1")
parser.add_argument("-c2", "--condition2", type=str, help="File path to condition 2")
parser.add_argument("-o", "--output", type=str, help="File path to where to store output")

args = vars(parser.parse_args())
fname_c1 = args['condition1']
fname_c2 = args['condition2']
output_path = args['output']

df_crops = extract_ROI(fname_c1, fname_c2)
df_convolved = gaussian_kernel_convolution(df_crops)
df_convolved.to_csv(output_path)                        