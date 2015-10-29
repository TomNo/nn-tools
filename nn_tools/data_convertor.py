#!/usr/bin/env python

__author__ = 'Tomas Novacik'

import argparse
import h5py
import kaldi_io

usage = 'usage: %prog [options]'
parser = argparse.ArgumentParser(usage)
parser.add_argument('-f', '--features', dest='features', action='store', type=str,  help='input features kaldi scp', required=True)
parser.add_argument('-l', '--labels', dest='labels', action='store', type=str, help='labels of the data', required=True)
parser.add_argument('-o', '--output-file', dest='output_file', action='store', type=str,  help='output hdf5 file name', required=True)

options = parser.parse_args()

first = True

with h5py.File(options.output_file, "w") as o_file:
    max_tag_len = 0
    tags = []
    mats = []
    sizes = []
    features = []
    l_dict = {}
    l_list = []
    for tag, l in kaldi_io.read_ali_ark(options.labels):
        l_dict[tag] = l
    for tag, mat in kaldi_io.read_mat_scp(options.features):
        if len(tag) > max_tag_len:
            max_tag_len = len(tag)
        tags.append(tag)
        mats.append(mat.flatten())
        sizes.append(len(mat))
        if len(mat) != len(l_dict[tag]):
            raise ValueError("Labels lengths does not match data length.")
        l_list.extend(l_dict[tag])

    i_size = len(mat[0]) # upsy xD
    # must be done like this because torch does not support strings nor attributes:'(
    for index, tag in enumerate(tags):
        g = o_file.create_group(tag)
        g.create_dataset("labels", data=l_dict[tag])
        g.create_dataset("data", data=mats[index])
    o_file.create_dataset("i_size", data=[i_size])

# eof
