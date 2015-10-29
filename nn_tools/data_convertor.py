#!/usr/bin/env python

__author__ = 'Tomas Novacik'

import argparse
import h5py
import kaldi_io
import sys


def k2hdf(options):
    with h5py.File(options.output_file, "w") as o_file:
        max_tag_len = 0
        tags = []
        mats = []
        sizes = []
        l_dict = {}
        l_list = []
        for tag, l in kaldi_io.read_ali_ark(options.labels):
            l_dict[tag] = l
        for tag, mat in kaldi_io.read_mat_scp(options.features):
            if len(tag) > max_tag_len:
                max_tag_len = len(tag)
            tags.append(tag)
            mats.append(mat)
            sizes.append(len(mat))
            if len(mat) != len(l_dict[tag]):
                raise ValueError("Labels lengths does not match data length.")
            l_list.extend(l_dict[tag])

        # must be done like this because torch does not support strings nor attributes:'(
        for index, tag in enumerate(tags):
            g = o_file.create_group(tag)
            g.create_dataset("labels", data=l_dict[tag])
            g.create_dataset("data", data=mats[index].flatten())
            g.create_dataset("cols", data=[len(mats[index][0])])
            g.create_dataset("rows", data=[len(mats[index])])


def hdf2k(options):
    with h5py.File(options.net_output) as i_file:
        with open(options.output_file, "w") as o_file:
            for key, item in i_file.items():
                if key != 'i_size':
                    mat, cols, rows = item["data"], item["cols"][0], item["rows"][0]
                    kaldi_io.write_mat(o_file, mat.value.reshape(cols, rows), key)

usage = 'usage: %prog [options]'
parser = argparse.ArgumentParser(usage)
parser.add_argument('-f', '--features', dest='features', action='store', type=str,  help='input features kaldi scp', required=False)
parser.add_argument('-l', '--labels', dest='labels', action='store', type=str, help='labels of the data', required=False)
parser.add_argument('--net-output', dest='net_output', action='store', type=str, help='output of neural network in hdf file format')
parser.add_argument('--kaldi-to-hdf', dest='k2hdf', action="store_true", default=False, help='transfer kaldi inputs to hdfs file format')
parser.add_argument('--hdf-to-kaldi', dest='hdf2k', action="store_true", default=False, help='transfer hdfs nn output to kaldi file format')
parser.add_argument('-o', '--output-file', dest='output_file', action='store', type=str,  help='output file name', required=True)

options = parser.parse_args()

if options.k2hdf and options.hdf2k:
    print("Error:Both transfer options chosen, only one can be chosen at the time.")
    sys.exit(1)

if not options.k2hdf and not options.hdf2k:
    print("Error:One of the transfer option must be set.")
    sys.exit(1)

if options.k2hdf and (not options.features or not options.labels):
    print("Error:Missing features or label files.")
    sys.exit(1)

if options.hdf2k and not options.net_output:
    print("Error:Missing net output filename.")
    sys.exit(1)

if options.k2hdf:
    k2hdf(options)
else:
    hdf2k(options)

# eof
