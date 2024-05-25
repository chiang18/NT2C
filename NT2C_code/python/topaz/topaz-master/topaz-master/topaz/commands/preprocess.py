from __future__ import print_function
import sys
sys.path.append('/home/m112040012/vscode/topaz/topaz-master/topaz-master')
import argparse
from topaz.commands.normalize import add_arguments, main

name = 'preprocess'
help = 'downsample and normalize images in one step'


if __name__ == '__main__':
    parser = add_arguments()
    parser.prog = 'Script for performing image downsampling and normalization in one step'
    args = parser.parse_args()
    main(args)