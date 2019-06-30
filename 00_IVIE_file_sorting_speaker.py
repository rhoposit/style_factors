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
import glob, sys


#####################################################################
# This script reads renames IVIE filenames to work well with the code
# and sort by speaker. You need to provide a path to the IVIE wavs
#
# To run:
# mkdir ivie_wavs
# python 00_IVIE_file_sorting /path/to/IVIE/wavs
#####################################################################

in_wavs = sys.argv[1]
flist = glob.glob(in_wavs+"/*.wav")


loc_map = {"b":"belfast", "p":"bradford", "c":"cambridge", "w":"cardiff", "m":"dublin", "l":"leeds", "s":"liverpool", "j":"london", "n":"newcastle"}
style_map = {"f":"free_conversation", "m":"map_task", "r":"retold", "p":"read", "s":"statement", "q":"question", "w":"wh_question", "i":"inversion", "c":"coordination"}

outfile = "reference_ivie.csv"
output = open(outfile, "wb")
new_CSV = []
speakers = []
count = 1

for f in flist:
    print(f)
    identifying_info = f.split("/")[-1].split(".wav")[0]
    if len(identifying_info) == 5:
        sty, section, loc, spk_f,spk_l = identifying_info
    else:
        sty, section, a_b, loc, spk_f,spk_l = identifying_info
        section = section+a_b
    spk = loc+spk_f+spk_l
    if spk in speakers:
        speakerID = speakers.index(spk)
    else:
        speakers.append(spk)
        speakerID = speakers.index(spk)
    speakerID = "spk"+str(speakerID).zfill(4)
    new_fname = "ivie_wavs/"+speakerID+"-"+str(count).zfill(4)+".wav"
    uttID = str(count).zfill(4)
    command = "cp "+f+" "+new_fname
    os.system(command)
    print command

    
    ref_info = [new_fname, speakerID, uttID, style_map[sty], loc_map[loc], section]
    new_CSV.append(",".join(ref_info))
    count += 1

output.write("\n".join(new_CSV))
output.close()







