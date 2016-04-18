#!/usr/bin/env python

__author__ = 'Tomas Novacik'

import argparse
import h5py
import kaldi_io
import sys
import numpy as np

# TODO how to pass std/mean to test data

class Convertor(object):
    def __init__(self, options):
        self.options = options

    def split_sequence(self, mat, tag, labels):
        counter = 0
        seq_size = self.options.resequence
        assert len(mat) == len(labels), "Labels and seqs count does not match."
        while True:
            start = counter * seq_size
            n_mat = mat[start: (counter + 1) * seq_size]
            if len(n_mat) == 0:
                break

            if len(n_mat) < self.options.resequence:
                # in an event of shorter sequence add more data
                miss = self.options.resequence - len(n_mat)
                start -= miss
                n_mat = mat[start: (counter + 1) * seq_size]

            self.mats.append(n_mat)
            n_tag = tag + "_%s" % str(counter)
            self.tags.append(n_tag)
            if n_tag in self.l_dict:
                raise KeyError("Label: %s already exists." % n_tag)
            n_labels = labels[start: (counter + 1) * seq_size]
            assert len(n_labels) != 0, "Empty labels"
            self.l_dict[n_tag] = n_labels
            counter += 1


    def convert(self):
        if self.options.action.startswith("kaldi"):
            self.kaldi_convert()
        elif self.options.action == 'hdf-to-kaldi':
            self.hdf2k()
        else:
            raise Exception("Unsupported action.")

    def kaldi_convert(self):
        self.get_kaldi_data()
        if self.options.deltas_order > 0:
            self.add_deltas()
        if self.options.normalize:
            self.normalize()
        if self.options.action.endswith("nc"):
            self.k2nc()
        elif self.options.action.endswith("hdf"):
            self.k2hdf()

    def add_deltas(self):
        pass

    def normalize(self):
        m = 0
        std = 0
        if not self.options.mean:
            for mat in self.mats:
                m+=mat.mean()
            m /= float(len(self.mats))
        else:
            m=self.options.mean

        if not self.options.std:
            for mat in self.mats:
                std+=mat.std()
            std /= float(len(self.mats))
        else:
            std = self.options.std

        for i in xrange(len(self.mats)):
            self.mats[i] = self.mats[i] - m
            self.mats[i] = self.mats[i] / std
        print("Std: " + str(std))
        print("Mean: " + str(m))

    def get_kaldi_data(self):
        """Gets kaldi tags, features and alignments."""
        tags = []
        mats = []
        sizes = []
        l_dict = {}
        skipped = 0
        missing_labels = []
        if self.options.features.endswith(".scp"):
            f_reader = kaldi_io.read_mat_scp
        else:
            f_reader = kaldi_io.read_mat_ark
        if self.options.labels:
            for tag, l in kaldi_io.read_ali_ark(self.options.labels):
                if tag not in l_dict:
                    l_dict[tag] = l
                else:
                    raise KeyError("Tag is: " + tag + "is already present.")

        for tag, mat in f_reader(self.options.features):
            if not self.options.forward_pass and tag not in l_dict:
                skipped += 1
                missing_labels.append(tag)
                continue
            tags.append(tag)
            mats.append(mat)
            sizes.append(len(mat))

            if not self.options.forward_pass:
                if len(mat) != len(l_dict[tag]):
                    raise ValueError("Labels lengths does not match data length.")
        if self.options.resequence:
            # re-sequencing should be used only in the training mode
            # not in forward pass
            self.tags = []
            self.mats = []
            self.l_dict = {}
            del(sizes)
            for index in xrange(len(mats)):
                mat = mats.pop()
                tag = tags.pop()
                self.split_sequence(mat, tag, l_dict[tag])
                del(mat)
                del(l_dict[tag])
                del(tag)
        else:
            self.tags = tags
            self.mats = mats
            self.l_dict = l_dict
        if skipped != 0:
            print("Missing labels: " + str(skipped))
            print(missing_labels)

    def k2nc(self):
        """Convert kaldi data to netcdf format (currennt)."""
        o_file =  NetCDFFile(self.options.output_file, 'w')
        seq_lengths = [len(mat) for mat in self.mats]
        inputs = []
        for mat in self.mats:
            for line in mat:
                inputs.append(line)
        num_labels = max(max(self.l_dict.values(), key=max)) + 1

        def c_var(vname, data, vtype, dims):
            nc_var = o_file.createVariable (vname,vtype,dims)
            nc_var[:] = data[:]

        labels = []
        if self.options.forward_pass:
            for i in seq_lengths:
                labels +=[0] * i
        else:
            for i in self.tags:
                labels += self.l_dict[i].tolist()
        o_file.createDimension('numSeqs', len(seq_lengths))
        o_file.createDimension('numTimesteps', len(inputs))
        o_file.createDimension('inputPattSize', len(inputs[0]))
        o_file.createDimension('numLabels', num_labels)

        c_var('inputs', inputs, 'f', ('numTimesteps', 'inputPattSize'))
        c_var('seqLengths', seq_lengths, 'i', ('numSeqs',))
        c_var('targetClasses', labels, 'i',('numTimesteps',))

        m_s_length = len(max(self.tags, key=len))
        n_tags = []
        for tag in self.tags:
            n_tags.append(list(tag) + ['\0']*(m_s_length - len(tag)))
        o_file.createDimension('maxSeqTagLength', m_s_length)
        c_var('seqTags', np.array(n_tags), 'c', ('numSeqs', 'maxSeqTagLength'))
        o_file.close()

    def k2hdf(self):
        """Converts kaldi data to hdf5"""
        with h5py.File(self.options.output_file, "w") as o_file:
            r_count = 0
            # test dataset - must be done like this because torch-hdf5 does
            # not support strings nor attributes:'(
            if self.options.forward_pass:
                for index, tag in enumerate(self.tags):
                    g = o_file.create_group("/" + str(index) + "/" + tag)
                    g.create_dataset("data", data=self.mats[index].flatten())
                    g.create_dataset("cols", data=[len(self.mats[index][0])])
                    g.create_dataset("rows", data=[len(self.mats[index])])
                    r_count+= len(self.mats[index])
                o_file.create_dataset("rows", data=[r_count])
                o_file.create_dataset("cols", data=[len(self.mats[-1][0])])
            else: # optimized version for training/cv datasets
                cols = len(self.mats[0][0])
                t_map = [(self.mats[i], self.tags[i]) for i in xrange(len(self.mats))]
                # sort by size -- bigger samples near each other to optimize
                # mini batches
                t_map = sorted(t_map, key=lambda x:-len(x[0]))
                s_mats = [i[0] for i in t_map]
                s_tags = [i[1] for i in t_map]
                labels = np.concatenate([np.array(self.l_dict[tag]) for tag in s_tags])
                seq_sizes = [len(mat) for mat in s_mats]
                lab_sizes = [len(self.l_dict[tag]) for tag in s_tags]
                all_mats = np.concatenate(s_mats)
                del(self.mats)
                rows = len(all_mats)
                all_mats.resize((rows, cols))
                o_file.create_dataset("cols", data=[cols])
                o_file.create_dataset("labels", data=labels, compression="gzip", compression_opts=9)
                o_file.create_dataset("features", data=all_mats, compression="gzip", compression_opts=9)
                o_file.create_dataset("rows", data=[rows])
                o_file.create_dataset("seq_sizes", data=[seq_sizes])
                o_file.create_dataset("lab_sizes", data=[lab_sizes])

    def hdf2k(self):
        """Converts hdf5 format to kaldi matrix."""
        with h5py.File(self.options.net_output) as i_file:
            with open(self.options.output_file, "w") as o_file:
                for key, item in i_file.items():
                    mat = item["data"]
                    kaldi_io.write_mat(o_file, mat.value, key)

usage = 'usage: %prog [options]'
parser = argparse.ArgumentParser(usage)
parser.add_argument('-a', '--action', dest="action", choices=('kaldi-to-hdf', 'hdf-to-kaldi', 'kaldi-to-nc'), required=True)
parser.add_argument('-f', '--features', dest='features', action='store', type=str,  help='input features kaldi scp')
parser.add_argument('--forward-pass', dest='forward_pass', action='store_true', help='in case of forward pass no alignments are necessary', default=False)
parser.add_argument('-l', '--labels', dest='labels', action='store', type=str, help='labels of the data')
parser.add_argument('--net-output', dest='net_output', action='store', type=str, help='output of neural network in hdf file format')
parser.add_argument('-o', '--output-file', dest='output_file', action='store', type=str,  help='output file name', required=True)
parser.add_argument('--normalize', dest='normalize', action='store', type=int, default=1, help='normalize input, only usable for kaldi conversion')
parser.add_argument('--deltas-order', dest='deltas_order', action='store', type=int, help='add deltas, only usable for from kaldi conversion', default=0)
parser.add_argument('-r', '--resequence', dest="resequence", action="store", type=int, default=None, help='Re-sequence whole data set to given sequence length')
parser.add_argument('--mean', dest="mean", action="store", type=float, help="mean that will be substracted from the input data", default=None)
parser.add_argument('--std', dest="std", action="store", type=float, help="standard deviation that will divide input input data", default=None)

options = parser.parse_args()

if "nc" in options.action:
    # try:
    #     from Scientific.IO.NetCDF import NetCDFFile
    # except ImportError:
    from netCDF4 import Dataset as NetCDFFile

if options.action.startswith("kaldi") and (not options.features or not options.labels):
    if options.action.endswith("nc"):
        print("Error:Missing features or label files.")
        sys.exit(1)
    elif not options.features:
        print("Error:Missing features.")
        sys.exit(1)

if not options.action.startswith("kaldi"):
    if options.normalize:
        print("Error:Normalization is done only when converting data from kaldi.")
        sys.exit(1)
    if options.deltas_order != 0:
        print("Error:Deltas order must be 0 (Only used for from kaldi conversion)")
        sys.exit(1)

if not options.action.startswith("kaldi") and not options.net_output:
    print("Error:Missing net output filename.")
    sys.exit(1)

if (options.mean or options.std) and not options.normalize:
    print("Mean and std specified although normalization was not selected.")
    sys.exit(1)

convertor = Convertor(options)
convertor.convert()

# eof
