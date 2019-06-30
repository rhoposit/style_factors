################################################################################
#                https://github.com/rhoposit/style_factors
#
#                Centre for Speech Technology Research
#                     University of Edinburgh, UK
#                      Copyright (c) 2014-2015
#                        All Rights Reserved.
#
# The system as a whole and most of the files in it are distributed
# under the following copyright and conditions
#
#  Permission is hereby granted, free of charge, to use and distribute
#  this software and its documentation without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of this work, and to
#  permit persons to whom this work is furnished to do so, subject to
#  the following conditions:
#
#   - Redistributions of source code must retain the above copyright
#     notice, this list of conditions and the following disclaimer.
#   - Redistributions in binary form must reproduce the above
#     copyright notice, this list of conditions and the following
#     disclaimer in the documentation and/or other materials provided
#     with the distribution.
#   - The authors' names may not be used to endorse or promote products derived
#     from this software without specific prior written permission.
#
#  THE UNIVERSITY OF EDINBURGH AND THE CONTRIBUTORS TO THIS WORK
#  DISCLAIM ALL WARRANTIES WITH REGARD TO THIS SOFTWARE, INCLUDING
#  ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS, IN NO EVENT
#  SHALL THE UNIVERSITY OF EDINBURGH NOR THE CONTRIBUTORS BE LIABLE
#  FOR ANY SPECIAL, INDIRECT OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
#  WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN
#  AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION,
#  ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE OF
#  THIS SOFTWARE.
################################################################################
import sys, os
import numpy as np
import glob
from collections import defaultdict
from sklearn.model_selection import train_test_split


###########################################################################
# This script reads in a text file of Kaldi utterance vectors
# and a text file of Kaldi, based on the reference file
#
# To run:
# python 02_vecs2npy_iemocap.py kaldi_utterance_file.txt reference.csv x_or_i
###########################################################################

utts_input_file = sys.argv[1]
reference_csv = sys.argv[2]
x_or_i = sys.argv[3]

def load_utts(infile, y_dict, y_hat):
    a = y_dict.values()
    cats = {x:a.count(x) for x in a}
    input = open(infile, "r")
    UTTS_RAW = input.read().split("]\n")[:-1]
    input.close()
    uttvecs = []
    X, y, ysid, spkutt = [], [], [], []
    for i in range(0, len(UTTS_RAW)):
        uttID = UTTS_RAW[i].split(" ")[0]
        uttvec = np.array(UTTS_RAW[i].split(" ")[3:-1]).astype(np.float)
        label = y_dict[uttID.split("-")[-1]]
        sid = y_hat.index(uttID.split("-")[0])
        if label <= 30:
            X.append(uttvec)
            y.append(label)
            ysid.append(sid)
            spkutt.append(uttID)
    return np.array(X), np.array(y), np.array(ysid), np.array(spkutt, dtype=np.str)


def load_spkrs(infile):
    input = open(infile, "r")
    SPKS_RAW = input.read().split("\n")[:-1]
    input.close()
    SPKS = defaultdict(np.array)
    S = []
    for i in range(0, len(SPKS_RAW)):
        spk = SPKS_RAW[i].split(" ")[0]
        spkvec = np.array(SPKS_RAW[i].split(" ")[3:-1]).astype(np.float)
        SPKS[spk] = spkvec
        S.append(spkvec)
    return np.array(S)


def load_csv(infile):
    reference = []
    input = open(infile, "r")
    data = input.read().split("\n")[:-1]
    input.close()
    y_dict = defaultdict(int)
    y_hat = []
    for d in data:
        info = d.split(",")
        label = info[-1]
        if label in reference:
            labelID = reference.index(label)
        else:
            reference.append(label)
            labelID = reference.index(label)
        uttID = info[-2]
        sid = info[-3]
        y_dict[uttID] = labelID
        y_hat.append(sid)
    return y_dict, y_hat

def write_to_file(xdata, ydata, outfile, outfile2):
    output = open(outfile, "w")
    output2 = open(outfile2, "w")
    for spk, utt in zip(ydata, xdata):
        utt_string = " ".join(list(map(str, utt)))
        outstring = spk+"  [ "+str(utt_string)+" ]\n"
        outstring2 = spk+" "+spk.split("-")[0]+"\n"
        output.write(outstring)
        output2.write(outstring2)
    output2.close()


y_dict, y_hat = load_csv(reference_csv)
y_hat = list(set(y_hat))
print("num speakers: ", str(len(y_hat)))
X,y, ysid, spkutt = load_utts(utts_input_file, y_dict, y_hat)

# split into train/test/dev preserving class priors
indices = np.arange(X.shape[0])
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(X, y, indices, stratify=y, test_size=0.25, random_state=42)
indices = np.arange(X_train.shape[0])
ysid_train = np.take(ysid, idx_train)
ysid_test = np.take(ysid, idx_test)
spkutt_train = np.take(spkutt, idx_train)
spkutt_test = np.take(spkutt, idx_test)
X_train, X_valid, y_train, y_valid, idx_train, idx_valid = train_test_split(X_train, y_train, indices, stratify=y_train, test_size=0.15, random_state=42)
ysid_train = np.take(ysid, idx_train)
ysid_valid = np.take(ysid, idx_valid)
spkutt_train = np.take(spkutt, idx_train)
spkutt_valid = np.take(spkutt, idx_valid)

# save the vectors and ground truth for PLDA analysis
write_to_file(X_train, spkutt_train, "iemocap_basic4/"+x_or_i+"_utts_train.txt", "iemocap_basic4/"+x_or_i+"_utt2spk_train")
write_to_file(X_valid, spkutt_valid, "iemocap_basic4/"+x_or_i+"_utts_valid.txt", "iemocap_basic4/"+x_or_i+"_utt2spk_valid")
write_to_file(X_test, spkutt_test, "iemocap_basic4/"+x_or_i+"_utts_test.txt", "iemocap_basic4/"+x_or_i+"_utt2spk_test")

# save the vectors and ground truth as numpy segments for machine learning
# save the feature vector
utts_output_file = utts_input_file.split(".")[0]+"_X_train.npy"
np.save(utts_output_file, X_train)
utts_output_file = utts_input_file.split(".")[0]+"_X_valid.npy"
np.save(utts_output_file, X_valid)
utts_output_file = utts_input_file.split(".")[0]+"_X_test.npy"
np.save(utts_output_file, X_test)

# save a label for style type
utts_output_file = utts_input_file.split(".")[0]+"_y_train.npy"
np.save(utts_output_file, y_train)
utts_output_file = utts_input_file.split(".")[0]+"_y_valid.npy"
np.save(utts_output_file, y_valid)
utts_output_file = utts_input_file.split(".")[0]+"_y_test.npy"
np.save(utts_output_file, y_test)

# save an ID for speaker
#utts_output_file = utts_input_file.split(".")[0]+"_ysid_train.npy"
#np.save(utts_output_file, ysid_train)
#utts_output_file = utts_input_file.split(".")[0]+"_ysid_valid.npy"
#np.save(utts_output_file, ysid_valid)
#utts_output_file = utts_input_file.split(".")[0]+"_ysid_test.npy"
#np.save(utts_output_file, ysid_test)


