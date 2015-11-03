#!/usr/bin/env python

__author__ = 'Tomas Novacik'

import argparse
import h5py
import kaldi_io
import sys
import numpy as np
from Scientific.IO.NetCDF import NetCDFFile

def get_kaldi_data(options):
    """Gets kaldi tags, features and alignments."""
    tags = []
    mats = []
    sizes = []
    l_dict = {}
    if not options.forward_pass:
        for tag, l in kaldi_io.read_ali_ark(options.labels):
            l_dict[tag] = l
    for tag, mat in kaldi_io.read_mat_scp(options.features):
        tags.append(tag)
        mats.append(mat)
        sizes.append(len(mat))

        if not options.forward_pass:
            if len(mat) != len(l_dict[tag]):
                raise ValueError("Labels lengths does not match data length.")
    return tags, mats, sizes, l_dict


def k2nc(options):
    """Convert kaldi data to netcdf format (currennt)."""
    o_file =  NetCDFFile(options.output_file, 'w')
    tags, mats, sizes, l_dict = get_kaldi_data(options)
    seq_lengths = [len(mat) for mat in mats]
    inputs = []
    for mat in mats:
        for line in mat:
            inputs.append(line)
    num_labels = 0
    if not options.forward_pass:
        num_labels = max(max(l_dict.values(), key=max)) + 1

    def c_var(vname, data, vtype, dims):
        nc_var = o_file.createVariable (vname,vtype,dims)
        nc_var.assignValue(data)

    labels = []
    if options.forward_pass:
        for i in seq_lengths:
            labels +=[0] * i
    else:
        for i in tags:
            labels += l_dict[i].tolist()
    o_file.createDimension('numSeqs', len(seq_lengths))
    o_file.createDimension('numTimesteps', len(inputs))
    o_file.createDimension('inputPattSize', len(inputs[0]))
    o_file.createDimension('numLabels', num_labels)

    c_var('inputs', inputs, 'f', ('numTimesteps', 'inputPattSize'))
    c_var('seqLengths', seq_lengths, 'i', ('numSeqs',))
    c_var('targetClasses', labels, 'i',('numTimesteps',))

    m_s_length = len(max(tags, key=len))
    n_tags = []
    for tag in tags:
        n_tags.append(list(tag) +['\0']*(m_s_length - len(tag)))
    o_file.createDimension('maxSeqTagLength', m_s_length)
    c_var('seqTags', np.array(n_tags), 'c', ('numSeqs', 'maxSeqTagLength'))
    o_file.close()


def k2hdf(options):
    """Converts kaldi data to hdf5"""
    with h5py.File(options.output_file, "w") as o_file:
        tags, mats, sizes, l_dict = get_kaldi_data(options)
        # must be done like this because torch-hdf5 does not support strings nor attributes:'(
        for index, tag in enumerate(tags):
            g = o_file.create_group(tag)
            if not options.forward_pass:
                g.create_dataset("labels", data=l_dict[tag])
            g.create_dataset("data", data=mats[index].flatten())
            g.create_dataset("cols", data=[len(mats[index][0])])
            g.create_dataset("rows", data=[len(mats[index])])


def hdf2k(options):
    """Converts hdf5 format to kaldi matrix."""
    with h5py.File(options.net_output) as i_file:
        with open(options.output_file, "w") as o_file:
            for key, item in i_file.items():
                if key != 'i_size':
                    mat, cols, rows = item["data"], item["cols"][0], item["rows"][0]
                    kaldi_io.write_mat(o_file, mat.value.reshape(cols, rows), key)

usage = 'usage: %prog [options]'
parser = argparse.ArgumentParser(usage)
parser.add_argument('-a', '--action', dest="action", choices=('kaldi-to-hdf', 'hdf-to-kaldi', 'kaldi-to-nc'), required=True)
parser.add_argument('-f', '--features', dest='features', action='store', type=str,  help='input features kaldi scp')
parser.add_argument('--forward-pass', dest='forward_pass', action='store_true', help='in case of forward pass no alignments are necessary', default=False)
parser.add_argument('-l', '--labels', dest='labels', action='store', type=str, help='labels of the data')
parser.add_argument('--net-output', dest='net_output', action='store', type=str, help='output of neural network in hdf file format')
parser.add_argument('-o', '--output-file', dest='output_file', action='store', type=str,  help='output file name', required=True)

options = parser.parse_args()

if options.action.startswith("kaldi") and (not options.features or not options.labels):
    if options.forward_pass and not options.features:
        print("Error:Missing features.")
        sys.exit(1)
    elif not options.forward_pass:
        print("Error:Missing features or label files.")
        sys.exit(1)

if not options.action.startswith("kaldi") and not options.net_output:
    print("Error:Missing net output filename.")
    sys.exit(1)

actions = dict([('kaldi-to-hdf', k2hdf) ,('hdf-to-kaldi', hdf2k),
                ('kaldi-to-nc', k2nc)])
actions[options.action](options)
# eof
