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
from __future__ import print_function, division
import warnings
warnings.filterwarnings('ignore')

from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)

import keras
from keras.utils import plot_model
from keras import regularizers
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, merge, LSTM
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn import decomposition


import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import numpy as np
np.set_printoptions(suppress=True)


#########################################################################################
# To Run:
# python 04_PCA.py [x|i] [iemocap|ivie] [orig|dnn-aev|dnn-ae1|dnn-ae2|dnn-ae3|dnn-aec]
# Examples:
# python 04_PCA.py x ivie dnn-aev
# python 04_PCA.py i ivie orig
#########################################################################################

x_or_i = sys.argv[1] # select x or i vectors to work with
exp = sys.argv[2] # helps where to find the data: orig, dnn-ae1, dnn-ae2, etc
dataset = sys.argv[3]

l2_val = 0.0001


    
# load up the data
def load_numpy(X_file, y_file):
    print(X_file)
    print(y_file)
    X_train = np.load(X_file+"_train.npy")
    X_valid = np.load(X_file+"_valid.npy")
    X_test = np.load(X_file+"_test.npy")
    y_train = np.load(y_file+"_train.npy")
    y_valid = np.load(y_file+"_valid.npy")
    y_test = np.load(y_file+"_test.npy")

    return X_train, X_valid, X_test, y_train, y_valid, y_test


def write_to_file(xdata, ydata, outfile):
    output = open(outfile, "w")
    for spk, utt in zip(ydata, xdata):
        utt_string = " ".join(list(map(str, utt)))
        outstring = spk+"  [ "+str(utt_string)+" ]\n"
        output.write(outstring)
    output.close()


def min_categorical_crossentropy(y_true, y_pred):
    return -K.categorical_crossentropy(y_true, y_pred)


def special_scaling(train, valid, test):
    maxi = train.max()
    mini = train.min()
    meani = train.mean()
    stdi = train.std()
    X_train = (train - meani) / stdi
    X_valid = (valid - meani) / stdi
    X_test = (test - meani) / stdi
    return np.array(X_train), np.array(X_valid), np.array(X_test), meani, stdi, maxi, mini


def load_spk_data(exp_folder, x_or_i, dataset):
    ret = []
    yfile = exp_folder+"/"+x_or_i+"_utt2spk_"+dataset
    input = open(yfile, "r")
    y = input.read().split("\n")
    input.close()
    for item in y:
        spk = item.split(" ")[0]
        ret.append(spk)
    return ret

def scale_back(train, valid, test, meani, stdi, maxi, mini):
#    train = np.array([(x_i * (maxi - mini)) + mini for x_i in train])
#    valid = np.array([(x_i * (maxi - mini)) + mini for x_i in valid])
#    test = np.array([(x_i * (maxi - mini)) + mini for x_i in test])
    X_train = (train * stdi) + meani
    X_valid = (valid * stdi) + meani
    X_test = (test * stdi) + meani
    return X_train, X_valid, X_test


def scale_back_test(test, meani, stdi, maxi, mini):
#    test = np.array([(x_i * (maxi - mini)) + mini for x_i in test])
    X_test = (test * stdi) + meani
    return X_test

def save_ydata(exp_folder, x_or_i, spk_train, spk_valid, spk_test):
    # save y-data
    outfile = exp_folder+"/"+x_or_i+"_y_train.npy"
    np.save(outfile, spk_train)
    outfile = exp_folder+"/"+x_or_i+"_y_valid.npy"
    np.save(outfile, spk_valid)
    outfile = exp_folder+"/"+x_or_i+"_y_test.npy"
    np.save(outfile, spk_test)


def save_train_data(exp_folder,x_or_i,latent_dim,train_ae,valid_ae,test_ae,y_train, y_valid, y_test,recon, spk_train, spk_valid, spk_test):
    # save txt versions of reconstructed data
    train_outfile = exp_folder+"/"+x_or_i+"_utts_"+recon+"_"+str(latent_dim)+"_train.txt"
    valid_outfile = exp_folder+"/"+x_or_i+"_utts_"+recon+"_"+str(latent_dim)+"_valid.txt"
    test_outfile = exp_folder+"/"+x_or_i+"_utts_"+recon+"_"+str(latent_dim)+"_test.txt"
    write_to_file(train_ae, spk_train, train_outfile)
    write_to_file(valid_ae, spk_valid, valid_outfile)
    write_to_file(test_ae, spk_test, test_outfile)

    # save numpy versions of reconstructed data
    outfile = train_outfile.split(".txt")[0]+".npy"
    np.save(outfile, train_ae)
    outfile = valid_outfile.split(".txt")[0]+".npy"
    np.save(outfile, valid_ae)
    outfile = test_outfile.split(".txt")[0]+".npy"
    np.save(outfile, test_ae)



# define this DNN
class DNN():
    def __init__(self, num_styles, img_shape):
        # define the shape of the input, num classes, etc
        self.channels = 1
        self.num_styles = num_styles
        self.img_shape = img_shape

        optimizer = Adam(0.0002)
        img = Input(shape=self.img_shape)

        print("shape of input in init: ", img.shape)
        self.style_classifier_DNN = self.build_style_classifier_DNN()
        labels = self.style_classifier_DNN(img)
        self.style_classifier_DNN.compile(loss='categorical_crossentropy',
            optimizer=optimizer,
            metrics=['accuracy'])        

    
    def build_style_classifier_DNN(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(256,kernel_regularizer=regularizers.l2(l2_val), activation='relu'))
        model.add(Dense(256,kernel_regularizer=regularizers.l2(l2_val), activation='relu'))
        model.add(Dense(256,kernel_regularizer=regularizers.l2(l2_val), activation='relu'))
        model.add(Dense(self.num_styles, activation='softmax',))
        model.summary()
        img = Input(shape=(self.img_shape))
        labels = model(img)
        return Model(img, labels)

    
        
    def get_results(self, pred, truth, name, intype):
        score = accuracy_score(truth, pred)
        # save the output
        outstring = ""
        outstring += "*********** "+name+": "+intype+" ***********\n"
        outstring += name+" - acc: "+str(100*score)+"\n"
        outstring += str(classification_report(truth, pred))+"\n"
        outstring += str(confusion_matrix(truth, pred))+"\n"

        # print some stuff to terminal before writing to file
        print("*********** "+name+": "+intype+" ***********")
        print(name+" - acc: "+str(100*score))
        print(str(classification_report(truth, pred)))
        print(str(confusion_matrix(truth, pred)))
        return

    
    def plot_history(self, H, l2_val, exp, latent_size, x_or_i):
        # grab the history object dictionary
        H = H.history        
        # plot the training loss and accuracy
        N = np.arange(0, len(H["loss"]))
        plt.style.use("ggplot")
        plt.figure()
        plt.plot(N, H["loss"], label="train_loss")
        plt.plot(N, H["val_loss"], label="val_loss")
        plt.plot(N, H["acc"], label="train_acc")
        plt.plot(N, H["val_acc"], label="val_acc")
        plt.title(exp+" Style Prediction")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Error")
        plt.legend()
        # save the figure
        l2_val = str(l2_val).replace(".", "_")
        plt.savefig("plot."+exp+"."+str(latent_size)+"."+x_or_i+"."+l2_val+".png")
        plt.close()
        
        

def run_main(latent_size, x_or_i, dataset, l2_val):
    exp_folder = dataset+"_pca"


    # format changed slightly between orig and reconstructed from AE
    X_file = dataset+"/"+x_or_i+"_utts_X"
    y_file = dataset+"/"+x_or_i+"_utts_y"


    # load original data
    (X_train_orig, X_valid_orig, X_test_orig, y_train, y_valid, y_test) = load_numpy(X_file, y_file)
    X_train, X_valid, X_test, meani, stdi, maxi, mini = special_scaling(X_train_orig, X_valid_orig, X_test_orig)

    # run PCA and save it
    n_components = latent_size
    pca = decomposition.PCA(n_components)
    pca.fit(X_train)
    X_train = pca.transform(X_train)
    X_valid = pca.transform(X_valid)
    X_test = pca.transform(X_test)

    spk_train = load_spk_data(exp_folder, x_or_i, "train")
    spk_valid = load_spk_data(exp_folder, x_or_i, "valid")
    spk_test = load_spk_data(exp_folder, x_or_i, "test")
    train_enc, valid_enc, test_enc = scale_back(X_train, X_valid, X_test, meani, stdi, maxi, mini)
    save_train_data(exp_folder, x_or_i, latent_size, train_enc, valid_enc, test_enc, y_train, y_valid, y_test, "pca", spk_train, spk_valid, spk_test)
    save_ydata(exp_folder, x_or_i, y_train, y_valid, y_test)


    X_train = np.expand_dims(X_train, axis=3)
    X_valid = np.expand_dims(X_valid, axis=3)
    X_test = np.expand_dims(X_test, axis=3)

    classes = list(set(list(y_train)))
    num_classes = len(classes)
    print("Num classes: ", num_classes)
    print(classes)
    cats = {x:list(y_train).count(x) for x in list(y_train)}
    print(cats)
    cats = {x:list(y_valid).count(x) for x in list(y_valid)}
    print(cats)
    cats = {x:list(y_test).count(x) for x in list(y_test)}
    print(cats)

    # turn the y categories into 1-hot vectors for keras
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)

#    val_method = "val_loss"
#    val_mode = "min"
    val_method = "val_acc"
    val_mode = "max"
    batch_size = 32

    print(X_train.shape)
    print(X_valid.shape)
    print(X_test.shape)
    print(y_train.shape)
    print(y_valid.shape)
    print(y_test.shape)

    h = np.histogram(X_train)
    mean = np.mean(X_train)
    std = np.std(X_train)
    maximum = np.max(X_train)
    minimum = np.min(X_train)
    print(maximum, minimum, mean, std)

    h = np.histogram(X_valid)
    mean = np.mean(X_valid)
    std = np.std(X_valid)
    maximum = np.max(X_valid)
    minimum = np.min(X_valid)
    print(maximum, minimum, mean, std)

    early_stopping = EarlyStopping(monitor=val_method,
                                   min_delta=0,
                                   patience=2,
                                   verbose=1, mode=val_mode)
    callbacks_list = [early_stopping]

    latent_size = (X_train.shape[1], X_train.shape[2])
    dnn = DNN(num_classes, latent_size)
    DNN_style_history = dnn.style_classifier_DNN.fit(X_train, y_train,
                                batch_size=batch_size,
                                epochs=100,shuffle=True,
                                validation_data=[X_valid, y_valid],
                                callbacks=callbacks_list)

#    dnn.plot_history(DNN_style_history, l2_val, exp_folder, latent_size, x_or_i)
    eval_result = dnn.style_classifier_DNN.evaluate(X_test, y_test)
    print("eval_result: ", eval_result)

    preds = dnn.style_classifier_DNN.predict(X_test)
    preds = np.argmax(preds, axis=1)
    test = np.argmax(y_test, axis=1)
    dnn.get_results(preds, test, "DNN-style", x_or_i+"-vector")

    return eval_result[0], eval_result[1]


    
if x_or_i == "x":
    Z = [512, 400, 300, 200, 100, 50, 20, 10, 5]
    recon_size = 512
if x_or_i == "i":
    Z = [400, 300, 200, 100, 50, 20, 10, 5]
    recon_size = 400
RES = []
for z in Z:    
    loss, acc = run_main(z, x_or_i, dataset, l2_val)
    enc_string =  str(loss)+","+str(acc)
    RES.append(enc_string)

print(dataset, "sweep", exp, x_or_i)
for res in RES:
    print(res)
