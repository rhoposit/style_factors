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
import os
import argparse
from os import listdir
from os.path import isfile, join
import numpy as np
from collections import defaultdict
from operator import itemgetter
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import glob
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


#####################################################################
# This script creates an utt2spk file for use with Kaldi
#
# To run:
# mkdir kaldi_files
# python 00_IVIE_file_sorting /path/to/wavs
#####################################################################


path_to_wavs = sys.argv[1]


outfile = "kaldi_files/utt2spk"
outfile2 = "kaldi_files/wav.scp"
output = open(outfile, "wb")
output2 = open(outfile2, "wb")
wavs = glob.glob(path_to_wavs+"*.wav")

for w in wavs:
    print w
    speakerID = w.split("/")[-1].split("-")[0]
    utt = w.split("/")[-1].split(".wav")[0]
    outstring = utt+" " + speakerID + "\n"
    output.write(outstring)
    fullpath = full_audio_path+"/"+w
    output2.write(utt+" "+fullpath+"\n")

output.close()
output2.close()


