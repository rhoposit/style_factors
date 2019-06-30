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

from DNN_AE import DNN_AE
import keras
from keras.utils import plot_model
from keras import regularizers
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers import MaxPooling2D, merge, SimpleRNN
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
import keras.backend as K
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, mean_absolute_error, mean_squared_error
from scipy.stats import entropy
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import numpy as np

#########################################################################################
# To Run:
# python 04_autoencoder.py [x|i] [iemocap|ivie] [orig|dnn-aev|dnn-ae1|dnn-ae2|dnn-ae3|dnn-aec]
# Examples:
# python 04_autoencoder.py x ivie dnn-aev
# python 04_autoencoder.py i ivie orig
#########################################################################################

x_or_i = sys.argv[1]   # select x or i vectors to work with
dataset = sys.argv[2]  # ivie, basic4, iemocap_large, mosei, etc
exp = sys.argv[3]      # dnn-ae1, dn-ae2, dnn-ae3, dnn-ae4, dnn-ae5, etc

def load_numpy(X_file, y_file):
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
    

def special_scaling(train, valid, test):
    maxi = train.max()
    mini = train.min()
    meani = train.mean()
    stdi = train.std()
    X_train = (train - meani) / stdi
    X_valid = (valid - meani) / stdi
    X_test = (test - meani) / stdi
#    X_train = [(x_i - mini) / (maxi - mini) for x_i in X_train]
#    X_valid = [(x_i - mini) / (maxi - mini) for x_i in X_valid]
#    X_test = [(x_i - mini) / (maxi - mini) for x_i in X_test]
    return np.array(X_train), np.array(X_valid), np.array(X_test), meani, stdi, maxi, mini


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


def get_results(pred, truth, name, intype):
    score = accuracy_score(truth, pred)
    # print some stuff to terminal before writing to file
    print("*********** Style: "+intype+" ***********")
    print("\n"+name+" - acc: "+str(100*score))    
    return


def plot_history(H, exp, latent_size, x_or_i, plot_dict, p, lr, l2, noise):
    # grab the history object dictionary
    H = H.history        
    # plot the training loss and accuracy
    N = np.arange(0, len(H["loss"]))
    plt.style.use("ggplot")
    plt.figure()
    for k,v in plot_dict.items():
        plt.plot(N, H[k], label=v)    
    plt.title(exp+" AE Training")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/Error")
    plt.legend()
    # save the figure
    l2 = str(l2).replace(".", "_")
    lr = str(lr).replace(".", "_")
    noise = str(noise).replace(".", "_")
    plt.savefig("plots/plot."+exp+"."+str(latent_size)+"."+str(p)+"."+lr+"."+l2+"."+noise+"."+x_or_i+".png")
    plt.close()


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


def get_plot_dict(exp):
    plot_dict = defaultdict(str)
    plot_dict["loss"] = "train_loss"
    plot_dict["val_loss"] = "val_loss"
    if exp == "dnn-aev":
        plot_dict["mean_absolute_error"] = "train_mae"
        plot_dict["val_mean_absolute_error"] = "val_mae"
        return plot_dict
    if exp == "dnn-ae1" or exp == "dnn-ae2" or exp == "dnn-ae3"
        plot_dict["DEC_loss"] = "train_mae"
        plot_dict["val_DEC_loss"] = "val_mae"
        plot_dict["S1_acc"] = "train_acc_enc"
        plot_dict["val_S1_acc"] = "val_acc_enc"
        plot_dict["S2_acc"] = "train_acc_adv"
        plot_dict["val_S2_acc"] = "val_acc_adv"
    return plot_dict


def AE_evaluate(preds, truth, meani, stdi, maxi, mini):
    preds = preds[:,:,0]
    truth = truth[:,:,0]
    hist = np.histogram(preds)
    print("Histogram of preds:\n", hist)
    preds_s = scale_back_test(preds, meani, stdi, maxi, mini)
    hist = np.histogram(preds_s)
    print("Histogram of preds (rescaled):\n", hist)
    hist = np.histogram(truth)
    print("Histogram of original:\n", hist)
    truth_s = scale_back_test(truth, meani, stdi, maxi, mini)
    hist = np.histogram(truth_s)
    print("Histogram of original (rescaled):\n", hist)
    mae = mean_absolute_error(preds, truth)
    return [str(mae)]

def MSE_MAE_evaluate_mean(preds, truth, meani, stdi, maxi, mini):
    preds = preds[:,:]
    truth = truth[:,:]
    hist = np.histogram(preds)
    print("Histogram of preds:\n", hist)
    preds_s = scale_back_test(preds, meani, stdi, maxi, mini)
    hist = np.histogram(preds_s)
    print("Histogram of preds (rescaled):\n", hist)
    hist = np.histogram(truth)
    print("Histogram of original:\n", hist)
    truth_s = scale_back_test(truth, meani, stdi, maxi, mini)
    hist = np.histogram(truth_s)
    print("Histogram of original (rescaled):\n", hist)
    mse = mean_squared_error(preds, truth)
    mae = mean_absolute_error(preds, truth)
    return [str(mse), str(mae)]


def cross_entropy(predictions, targets, epsilon=1e-12):
    predictions = np.clip(predictions, epsilon, 1. - epsilon)
    N = predictions.shape[0]
    ce = -np.sum(targets*np.log(predictions+1e-9))/N
    return ce

def STY_evaluate(preds, truth):
    cross_ent = cross_entropy(preds,truth)
    preds = np.argmax(preds, axis=1)
    truth = np.argmax(truth, axis=1)
    acc = accuracy_score(preds, truth)
    return [str(cross_ent), str(acc)]

    
def run_main(exp, latent_size, x_or_i, dataset, p, lr, l2, noise):
    exp_folder = dataset+"_"+exp
    X_file = dataset+"/"+x_or_i+"_utts_X"
    y_file = dataset+"/"+x_or_i+"_utts_y"
    (X_train_orig, X_valid_orig, X_test_orig, y_train, y_valid, y_test) = load_numpy(X_file, y_file)
    
    classes = list(set(list(y_train)))
    num_classes = len(classes)
    # turn the y categories into 1-hot vectors for keras
    y_train = keras.utils.to_categorical(y_train, num_classes=num_classes)
    y_valid = keras.utils.to_categorical(y_valid, num_classes=num_classes)
    y_test = keras.utils.to_categorical(y_test, num_classes=num_classes)


    h = np.histogram(X_train_orig)
    mean = np.mean(X_train_orig)
    std = np.std(X_train_orig)
    maximum = np.max(X_train_orig)
    minimum = np.min(X_train_orig)
    print("Original Values")
    print(h[0])
    print(h[1])
    print(maximum, minimum, mean, std)
    
    X_train, X_valid, X_test, meani, stdi, maxi, mini = special_scaling(X_train_orig, X_valid_orig, X_test_orig)

    h = np.histogram(X_train)
    mean = np.mean(X_train)
    std = np.std(X_train)
    maximum = np.max(X_train)
    minimum = np.min(X_train)
    print("Rescaled range")
    print(h[0])
    print(h[1])
    print(maximum, minimum, mean, std)

    X_train_mean = np.mean(X_train, axis=0)
    X_train_mean = np.tile(X_train_mean, (X_train.shape[0],1))
    h = np.histogram(X_train_mean)
    mean = np.mean(X_train_mean)
    std = np.std(X_train_mean)
    maximum = np.max(X_train_mean)
    minimum = np.min(X_train_mean)
    print("Mean of Rescaled")
    print(h[0])
    print(h[1])
    print(maximum, minimum, mean, std)
    
    X_train_mean = np.mean(X_train, axis=0)
    X_train_mean = np.tile(X_train_mean, (X_train.shape[0],1))
    print(X_train_mean.shape)
    print(X_train.shape)
    baseline = MSE_MAE_evaluate_mean(X_train_mean, X_train, meani, stdi, maxi, mini)
    print("baseline MSE: ", baseline[0])
    print("baseline MAE: ", baseline[1])

    X_train = np.expand_dims(X_train, axis=3)
    X_valid = np.expand_dims(X_valid, axis=3)
    X_test = np.expand_dims(X_test, axis=3)
    print(str(X_train.shape)+"\n"+str(X_valid.shape)+"\n"+str(X_test.shape))
    print(str(y_train.shape)+"\n"+str(y_valid.shape)+"\n"+str(y_test.shape))

    if x_or_i == "x":
        rows = 512
    if x_or_i == "i":
        rows = 400
    
    val_method = "val_loss"
    val_mode = "min"
    batch_size = 32
    dnn = DNN_AE(rows, num_classes, latent_size, exp, lr, l2, noise)
    early_stopping = EarlyStopping(monitor=val_method,
            min_delta=0,
            patience=p,
            verbose=1, mode=val_mode)
    callbacks_list = [early_stopping]
    if exp == "dnn-aev":
        train_out = [X_train]
        valid_out = [X_valid]
    else:
        train_out = [X_train, y_train, y_train]
        valid_out = [X_valid, y_valid, y_valid]

    DNN_AE_style_history = dnn.autoencoder.fit(X_train, train_out,
            batch_size=batch_size,
            epochs=100,shuffle=True,
            validation_data=[X_valid, valid_out],
            callbacks=callbacks_list)
    plot_history(DNN_AE_style_history, exp_folder, latent_size, x_or_i, get_plot_dict(exp),p, lr, l2, noise)


    if exp == "dnn-aec":
        train_enc1 = dnn.encoder1.predict(X_train)
        valid_enc1 = dnn.encoder1.predict(X_valid)
        test_enc1 = dnn.encoder1.predict(X_test)        
        train_enc2 = dnn.encoder2.predict(X_train)
        valid_enc2 = dnn.encoder2.predict(X_valid)
        test_enc2 = dnn.encoder2.predict(X_test)

        # degrade the enc2 portion
        print(train_enc2.shape)
        train_enc2_mean = np.mean(train_enc2, axis=0)
        print(train_enc2_mean.shape)
        train_enc2 = np.tile(train_enc2_mean, (train_enc2.shape[0],1))
        print(train_enc2.shape)

        print(valid_enc2.shape)
        valid_enc2_mean = np.mean(valid_enc2, axis=0)
        print(valid_enc2_mean.shape)
        valid_enc2 = np.tile(valid_enc2_mean, (valid_enc2.shape[0],1))
        print(valid_enc2.shape)

        print(test_enc2.shape)
        test_enc2_mean = np.mean(test_enc2, axis=0)
        print(test_enc2_mean.shape)
        test_enc2 = np.tile(test_enc2_mean, (test_enc2.shape[0],1))
        print(test_enc2.shape)

                
        train_in = [train_enc1, train_enc2]
        valid_in = [valid_enc1, valid_enc2]
        dnn.decoder.fit(train_in, X_train,
                        batch_size=batch_size,
                        epochs=100,shuffle=True,
                        validation_data=[valid_in, X_valid],
                        callbacks=callbacks_list)

        # save stuff down below, get some MAE values, do other stuff here?

    
    print("Compiling performance results.. ")
    test = dnn.autoencoder.predict(X_test)
    if exp == "dnn-aev":
        res = AE_evaluate(test, X_test, meani, stdi, maxi, mini)
    elif exp == "dnn-aec":
        print("# items returned from autoencoder", len(test))
        test2 = dnn.decoder.predict([test_enc1, test_enc2])
        res1 = AE_evaluate(test[0], X_test, meani, stdi, maxi, mini)
        res2 = AE_evaluate(test2, X_test, meani, stdi, maxi, mini)
        res3 = STY_evaluate(test[1], y_test)
        res4 = STY_evaluate(test[2], y_test)
        res = res1 + res2 + res3 + res4
    else:
        print("# items returned from autoencoder", len(test))
        res1 = AE_evaluate(test[0], X_test, meani, stdi, maxi, mini)
        res2 = STY_evaluate(test[1], y_test)
        res3 = STY_evaluate(test[2], y_test)
        res = res1 + res2 + res3
    print("Complete.")
#    return res

    
    print("Compiling new dataset.. ")
    spk_train = load_spk_data(exp_folder, x_or_i, "train")
    spk_valid = load_spk_data(exp_folder, x_or_i, "valid")
    spk_test = load_spk_data(exp_folder, x_or_i, "test")

    if exp == "dnn-ae1":
        train = dnn.autoencoder.predict(X_train).reshape(X_train.shape[0], X_train.shape[1])
        valid = dnn.autoencoder.predict(X_valid).reshape(X_valid.shape[0], X_valid.shape[1])
        test = dnn.autoencoder.predict(X_test).reshape(X_test.shape[0], X_test.shape[1])
        train_ae, valid_ae, test_ae = scale_back(train, valid, test, meani, stdi, maxi, mini)
        train = dnn.encoder.predict(X_train)
        valid = dnn.encoder.predict(X_valid)
        test = dnn.encoder.predict(X_test)
        train_enc, valid_enc, test_enc = scale_back(train, valid, test, meani, stdi, maxi, mini)
        print(str(train_ae.shape)+"\n"+str(valid_ae.shape)+"\n"+str(test_ae.shape))
        print(str(train_enc.shape)+"\n"+str(valid_enc.shape)+"\n"+str(test_enc.shape))
        print("Saving dataset.. ")
        save_train_data(exp_folder, x_or_i, latent_size, train_ae, valid_ae, test_ae, y_train, y_valid, y_test, "recon", spk_train, spk_valid, spk_test)
        save_train_data(exp_folder, x_or_i, latent_size, train_enc, valid_enc, test_enc, y_train, y_valid, y_test, "enc", spk_train, spk_valid, spk_test)
        save_ydata(exp_folder, x_or_i, y_train, y_valid, y_test)
        print("Complete.")
        
    elif exp == "dnn-ae1" or exp == "dnn-ae2"  or exp == "dnn-ae3":
        train = dnn.encoder1.predict(X_train)
        valid = dnn.encoder1.predict(X_valid)
        test = dnn.encoder1.predict(X_test)
        train_enc1, valid_enc1, test_enc1 = scale_back(train, valid, test, meani, stdi, maxi, mini)
        train = dnn.encoder2.predict(X_train)
        valid = dnn.encoder2.predict(X_valid)
        test = dnn.encoder2.predict(X_test)
        train_enc2, valid_enc2, test_enc2 = scale_back(train, valid, test, meani, stdi, maxi, mini)
        print(str(train_enc1.shape)+"\n"+str(valid_enc1.shape)+"\n"+str(test_enc1.shape))
        print(str(train_enc2.shape)+"\n"+str(valid_enc2.shape)+"\n"+str(test_enc2.shape))
        print("Saving dataset.. ")
        save_train_data(exp_folder, x_or_i, latent_size, train_enc1, valid_enc1, test_enc1, y_train, y_valid, y_test, "enc1", spk_train, spk_valid, spk_test)
        save_train_data(exp_folder, x_or_i, latent_size, train_enc2, valid_enc2, test_enc2, y_train, y_valid, y_test, "enc2", spk_train, spk_valid, spk_test)
        save_ydata(exp_folder, x_or_i, y_train, y_valid, y_test)
        print("Complete.")

    elif exp == "dnn-aec":
        train = dnn.encoder1.predict(X_train)
        valid = dnn.encoder1.predict(X_valid)
        test = dnn.encoder1.predict(X_test)
        train_enc1, valid_enc1, test_enc1 = scale_back(train, valid, test, meani, stdi, maxi, mini)
        train = dnn.encoder2.predict(X_train)
        valid = dnn.encoder2.predict(X_valid)
        test = dnn.encoder2.predict(X_test)
        train_enc2, valid_enc2, test_enc2 = scale_back(train, valid, test, meani, stdi, maxi, mini)
        print(str(train_enc1.shape)+"\n"+str(valid_enc1.shape)+"\n"+str(test_enc1.shape))
        print(str(train_enc2.shape)+"\n"+str(valid_enc2.shape)+"\n"+str(test_enc2.shape))
        print("Saving dataset.. ")
        save_train_data(exp_folder, x_or_i, latent_size, train_enc1, valid_enc1, test_enc1, y_train, y_valid, y_test, "enc1", spk_train, spk_valid, spk_test)
        save_train_data(exp_folder, x_or_i, latent_size, train_enc2, valid_enc2, test_enc2, y_train, y_valid, y_test, "enc2", spk_train, spk_valid, spk_test)
        save_ydata(exp_folder, x_or_i, y_train, y_valid, y_test)
        print("Complete.")

        
    return res



if x_or_i == "x":
    Z = [512, 400, 300, 200, 100, 50, 20, 10, 5]
if x_or_i == "i":
    Z = [400, 300, 200, 100, 50, 20, 10, 5]
if exp == "dnn-ae1":
    Z = Z
else:
    Z = Z[:-1] # can't do an odd integer layer size when splitting

FULL = defaultdict(str)

patience = [2]
learning_rate = [0.0002]
l2_val = [0.0001]
noise_val = [0.5]


for p in patience:
    for lr in learning_rate:
        for l2 in l2_val:
            noise = 0.0
            RES = []
            for z in Z:
                print("Running main for latent size: ", z)
                res = run_main(exp, z, x_or_i, dataset, p, lr, l2, noise)
                RES.append(",".join(res))
            idstring = ",".join([str(z), str(p), str(lr), str(l2), str(noise)])
            print(dataset, x_or_i, exp, idstring)
            print("\n".join(RES))
            FULL[idstring] = RES
                
# In the case of doing a full-sweep
#patience = [2]
#learning_rate = [0.0001, 0.0002, 0.0003]
#l2_val = [0.0001, 0.0002]
#noise_val = [0.1, 0.2]

