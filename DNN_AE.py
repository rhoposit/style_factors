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
import tensorflow
from flipGradientTF import GradientReversal
from keras import backend as K
from keras.layers import Lambda
#from tensorflow.python.keras.layers import Lambda;
K.set_image_dim_ordering('tf')
from keras.utils import plot_model
from keras import regularizers
from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, GaussianNoise
from keras.layers import BatchNormalization, Activation, Embedding
from keras.layers import MaxPooling2D, MaxPooling1D, merge, SimpleRNN
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling1D, Conv1D, ZeroPadding1D, Conv2DTranspose, Cropping1D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras import losses
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys, os
import numpy as np


def min_categorical_crossentropy(y_true, y_pred):
    return -K.categorical_crossentropy(y_true, y_pred)



def get_ae_nodes(latent_dim, cols):
    # x-vectors
    if latent_dim == cols and cols == 512:
        ae_nodes = [512, 512, 512, 512]
    # i-vectors    
    elif latent_dim == cols and cols == 400:
        ae_nodes = [400, 400, 400, 400]
    elif latent_dim == 400 and cols == 512:
        ae_nodes = [512, 512, 400, 400]
    elif latent_dim == 300:
        ae_nodes = [cols, 400, 350, 300]
    elif latent_dim == 200:
        ae_nodes = [cols, 400, 300, 200]
    elif latent_dim == 100:
        ae_nodes = [cols, 300, 200, 100]
    elif latent_dim == 50:
        ae_nodes = [cols, 200, 100, 50]
    elif latent_dim == 20:
        ae_nodes = [cols, 200, 100, 20]
    elif latent_dim == 10:
        ae_nodes = [cols, 200, 50, 10]
    elif latent_dim == 5:
        ae_nodes = [cols, 200, 50, 5]
    return ae_nodes


def get_dual_nodes(latent_dim, cols):
    # x-vectors
    if latent_dim == cols and cols == 512:
        enc_nodes = [512, 512, 256, 256]
        dec_nodes = [512, 512, 512, 512]        
    # i-vectors    
    elif latent_dim == cols and cols == 400:
        enc_nodes = [400, 400, 200, 200]
        dec_nodes = [400, 400, 400, 400]
    elif latent_dim == 400 and cols == 512:
        enc_nodes = [512, 400, 400, 200]
        dec_nodes = [400, 400, 512, 512]
    elif latent_dim == 300:
        enc_nodes = [cols, 300, 200, 150]
        dec_nodes = [300, 350, 400, cols]
    elif latent_dim == 200:
        enc_nodes = [cols, 300, 200, 100]
        dec_nodes = [200, 300, 400, cols]
    elif latent_dim == 100:
        enc_nodes = [cols, 250, 150, 50]
        dec_nodes = [100, 150, 250, cols]
    elif latent_dim == 50:
        enc_nodes = [cols, 200, 100, 25]
        dec_nodes = [50, 100, 200, cols]
    elif latent_dim == 20:
        enc_nodes = [cols, 200, 50, 10]
        dec_nodes = [20, 50, 200, cols]
    else:
        enc_nodes = [cols, 200, 50, 5]
        dec_nodes = [10, 50, 200, cols]
    return enc_nodes, dec_nodes


def output_of_lambda(input_shape):
    print(input_shape)
    return (input_shape[0], input_shape[1])





# define this DNN
class DNN_AE():
    def __init__(self,rows, num_classes, latent_size, exp, lr, l2, noise):
        self.img_rows = 1
        self.img_cols = rows
        self.channels = 1
        self.num_styles = num_classes
        self.img_shape = (self.img_cols, self.img_rows)
        self.latent_dim = latent_size
        half_dim = int(self.latent_dim/2)
        self.split_latent_dim = half_dim
        self.l2_val = l2
        self.noise = noise
        self.hp_lambda = -1

        optimizer = Adam(lr)
        img = Input(shape=self.img_shape)
        enc1 = Input(shape=(self.split_latent_dim,))
        enc2 = Input(shape=(self.split_latent_dim,))
        

        if exp == "dnn-aev":
            ae_nodes = get_ae_nodes(self.latent_dim, self.img_cols)
            print(ae_nodes)
            self.encoder = self.build_encoder(ae_nodes)
            self.decoder = self.build_decoder(ae_nodes)        
            encoded_repr = self.encoder(img)
            reconstructed_img = self.decoder(encoded_repr)
            self.enc_shape = (self.latent_dim,1,)
            print("encoded_shape", self.enc_shape)
            print("reconstructed_img", reconstructed_img.shape)
            self.enc_style_classifier = self.build_enc_style_classifier("STY")
            self.final_style_classifier = self.build_final_style_classifier()
            label_style = self.final_style_classifier(reconstructed_img)
            label_style_enc = self.enc_style_classifier(encoded_repr)
            self.enc_style_classifier.compile(loss='categorical_crossentropy',
                                              optimizer=optimizer,
                                              metrics=['accuracy'])
            self.final_style_classifier.compile(loss='categorical_crossentropy',
                                              optimizer=optimizer,
                                              metrics=['accuracy'])

        if exp == "dnn-ae1" or exp == "dnn-ae2": #2 encoders, 1 decoder
            enc_nodes, dec_nodes = get_dual_nodes(self.latent_dim, self.img_cols)
            print(enc_nodes, dec_nodes)
            self.encoder1 = self.build_dual_encoder(enc_nodes, "ENC1")
            self.encoder2 = self.build_dual_encoder(enc_nodes, "ENC2")
            self.merge_decoder = self.build_merge_dual_decoder(dec_nodes)
            # split encoding space between style and speaker
            encoded_repr1 = self.encoder1(img)
            encoded_repr2 = self.encoder2(img)
            print("repr1", encoded_repr1.shape)
            print("repr2", encoded_repr2.shape)
            # reconstruct and warp mixed space for decoder
            reconstructed_img = self.merge_decoder([encoded_repr1, encoded_repr2])
            self.enc_shape = (self.split_latent_dim,1,)
            print("encoded_shape", self.enc_shape)
            print("reconstructed_img", reconstructed_img.shape)
            # two auxiliary classifiers to guide the split encoding space
            self.enc_style_classifier1 = self.build_enc_style_classifier("S1")
            self.enc_style_classifier2 = self.build_enc_style_classifier("S2")
            label_style = self.enc_style_classifier1(encoded_repr1)
            not_label_style = self.enc_style_classifier2(encoded_repr2)
            self.enc_style_classifier1.compile(loss='categorical_crossentropy',
                                              optimizer=optimizer,
                                              metrics=['accuracy'])
            self.enc_style_classifier2.compile(loss='categorical_crossentropy',
                                              optimizer=optimizer,
                                              metrics=['accuracy'])

            
        if exp == "dnn-ae3": #2 encoders, 1 decoder
            # with the shit encoding, you cannot get good reconstruction
            enc_nodes, dec_nodes = get_dual_nodes(self.latent_dim, self.img_cols)
            print(enc_nodes, dec_nodes)
            self.encoder1 = self.build_dual_encoder(enc_nodes, "ENC1")
            self.encoder2 = self.build_mean_encoder(enc_nodes, "ENC2")
            self.merge_decoder = self.build_merge_dual_decoder(dec_nodes)
            # split encoding space between style and speaker
            encoded_repr1 = self.encoder1(img)
            encoded_repr2 = self.encoder2(img)
            print("repr1", encoded_repr1.shape)
            print("repr2", encoded_repr2.shape)
            # reconstruct and warp mixed space for decoder
            reconstructed_img = self.merge_decoder([encoded_repr1, encoded_repr2])
            self.enc_shape = (self.split_latent_dim,1,)
            print("encoded_shape", self.enc_shape)
            print("reconstructed_img", reconstructed_img.shape)
            # two auxiliary classifiers to guide the split encoding space
            self.enc_style_classifier1 = self.build_enc_style_classifier("S1")
            self.enc_style_classifier2 = self.build_enc_style_classifier("S2")
            label_style = self.enc_style_classifier1(encoded_repr1)
            not_label_style = self.enc_style_classifier2(encoded_repr2)
            self.enc_style_classifier1.compile(loss='categorical_crossentropy',
                                              optimizer=optimizer,
                                              metrics=['accuracy'])
            self.enc_style_classifier2.compile(loss='categorical_crossentropy',
                                              optimizer=optimizer,
                                              metrics=['accuracy'])

        if exp == "dnn-aec": #2 encoders, 1 decoder, freeze and retrain
            # with the shit encoding, you cannot get good reconstruction
            enc_nodes, dec_nodes = get_dual_nodes(self.latent_dim, self.img_cols)
            print(enc_nodes, dec_nodes)
            self.encoder1 = self.build_dual_encoder(enc_nodes, "ENC1")
            self.encoder2 = self.build_mean_encoder(enc_nodes, "ENC2")
            self.merge_decoder = self.build_merge_dual_decoder(dec_nodes)
            self.merge_decoder2 = self.build_merge_dual_decoder(dec_nodes)
            # split encoding space between style and speaker
            encoded_repr1 = self.encoder1(img)
            encoded_repr2 = self.encoder2(img)
            print("repr1", encoded_repr1.shape)
            print("repr2", encoded_repr2.shape)
            reconstructed_img = self.merge_decoder([encoded_repr1, encoded_repr2])
            reconstructed_img2 = self.merge_decoder2([enc1, enc2])
            self.enc_shape = (self.split_latent_dim,1,)
            print("encoded_shape", self.enc_shape)
            print("reconstructed_img", reconstructed_img.shape)
            # two auxiliary classifiers to guide the split encoding space
            self.enc_style_classifier1 = self.build_enc_style_classifier("S1")
            self.enc_style_classifier2 = self.build_enc_style_classifier("S2")
            label_style = self.enc_style_classifier1(encoded_repr1)
            not_label_style = self.enc_style_classifier2(encoded_repr2)
            self.enc_style_classifier1.compile(loss='categorical_crossentropy',
                                              optimizer=optimizer,
                                              metrics=['accuracy'])
            self.enc_style_classifier2.compile(loss='categorical_crossentropy',
                                              optimizer=optimizer,
                                              metrics=['accuracy'])

            
            
        if exp == "dnn-aev":
            self.autoencoder = Model(img, [reconstructed_img])
            self.autoencoder.compile(
                loss=['mse'],
                optimizer=optimizer,
                metrics=['mae'] )
        elif exp == "dnn-ae1":
            self.autoencoder = Model(img, [reconstructed_img, label_style, not_label_style])
            self.autoencoder.compile(
                loss=['mse', 'categorical_crossentropy','categorical_crossentropy'],
                optimizer=optimizer,
                metrics=['mae', 'accuracy', 'accuracy'] )
        elif exp == "dnn-ae2":
            self.autoencoder = Model(img, [reconstructed_img, label_style, not_label_style])
            self.autoencoder.compile(
                loss=['mse', 'categorical_crossentropy',min_categorical_crossentropy],
                loss_weights=[1.0, 1.0, 0.05],
                optimizer=optimizer,
                metrics=['mae', 'accuracy', 'accuracy'] )
        elif exp == "dnn-ae3":
            self.autoencoder = Model(img, [reconstructed_img, label_style, not_label_style])
            self.autoencoder.compile(
                loss=['mse', 'categorical_crossentropy','categorical_crossentropy'],
                optimizer=optimizer,
                metrics=['mae', 'accuracy', 'accuracy'] )
        elif exp == "dnn-aec":
            self.autoencoder = Model(img, [reconstructed_img, label_style, not_label_style])
            self.autoencoder.compile(
                loss=['mse', 'categorical_crossentropy',min_categorical_crossentropy],
                loss_weights=[1.0, 1.0, 0.05],
                optimizer=optimizer,
                metrics=['mae', 'accuracy', 'accuracy'] )
            self.decoder = Model([enc1, enc2], reconstructed_img2)
            self.decoder.compile(
                loss=['mse'],
                optimizer=optimizer,
                metrics=['mae'] )
            
    # Fully-Connected ENcoder/DEcoder
    def build_encoder(self, ae_nodes):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        img = Input(shape=self.img_shape)
        h = Flatten()(img)
        for n in ae_nodes:
#            h = BatchNormalization()(h)
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
#            h = BatchNormalization()(h)
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
#            h = BatchNormalization()(h)
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
        latent_repr = Dense(self.latent_dim)(h)
        return Model(img, latent_repr, name="ENC")

    
    def build_decoder(self, ae_nodes):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        ae_nodes = ae_nodes[::-1]
        model = Sequential()
        for n in ae_nodes:
#            model.add(BatchNormalization())
            model.add(Dense(n, activation=fx, input_dim=self.latent_dim, kernel_regularizer=regularizers.l2(l2_val)))
            model.add(Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val)))
#            model.add(BatchNormalization())
            model.add(Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val)))
#            model.add(BatchNormalization())
        model.add(Dense(np.prod(self.img_shape)))
        model.add(Reshape(self.img_shape))
        model.summary()
        z = Input(shape=(self.latent_dim,))
        img = model(z)
        return Model(z, img, name="DEC")

    
    def build_conv_encoder(self, invec, ae_nodes):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        h1 = Conv1D(filters=ae_nodes[0], kernel_size=k, padding='same',activation=fx,kernel_regularizer=regularizers.l2(l2_val))(invec)
        print("h1", h1.shape)
        p1 = MaxPooling1D(pool_size=m)(h1)
        print("p1", p1.shape)
        for n in ae_nodes[1:-1]:
            h1 = Conv1D(filters=n, kernel_size=k, padding='same',activation=fx,kernel_regularizer=regularizers.l2(l2_val))(p1)
            print("h1", n, h1.shape)
            p1 = MaxPooling1D(pool_size=m)(h1)
            print("p1", n, p1.shape)
        encoded = Conv1D(filters=ae_nodes[-1], kernel_size=k, padding='same',activation=fx,kernel_regularizer=regularizers.l2(l2_val))(p1)
        print("encoded", encoded.shape)
        return encoded

    def build_conv_decoder(self, invec, ae_nodes):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        ae_nodes = ae_nodes[::-1]
        h4 = Conv1D(filters=ae_nodes[0], kernel_size=k, padding='same',activation=fx,kernel_regularizer=regularizers.l2(l2_val))(invec)
        print("h4", h4.shape)
        u4 = UpSampling1D(m)(h4)
        print("u4", u4.shape)
        for n in ae_nodes[1:-1]:
            h4 = Conv1D(filters=n, kernel_size=k, padding='same',activation=fx,kernel_regularizer=regularizers.l2(l2_val))(u4)
            print("h4", n, h4.shape)
            u4 = UpSampling1D(m)(h4)
            print("u4", n, u4.shape)
        # normalize the values between -1 and 1 ???
        decoded = Conv1D(1, kernel_size=k, padding='same')(u4)
        print("decoded", decoded.shape)
        return decoded



    def build_split_conv_encoder(self, invec, ae_nodes):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        h1 = Conv1D(filters=ae_nodes[0], kernel_size=k, padding='same',activation=fx,kernel_regularizer=regularizers.l2(l2_val))(invec)
        print("h1", h1.shape)
        p1 = MaxPooling1D(pool_size=m)(h1)
        print("p1", p1.shape)
        for n in ae_nodes[1:-1]:
            h1 = Conv1D(filters=n, kernel_size=k, padding='same',activation=fx,kernel_regularizer=regularizers.l2(l2_val))(p1)
            print("h1", n, h1.shape)
            p1 = MaxPooling1D(pool_size=m)(h1)
            print("p1", n, p1.shape)
        latent_repr = Conv1D(filters=ae_nodes[-1], kernel_size=k, padding='same',activation=fx,kernel_regularizer=regularizers.l2(l2_val))(p1)
        def slice1(x):
            half_dim = int(self.latent_dim/2)
            return x[:, :, :half_dim]
        def slice2(x):
            half_dim = int(self.latent_dim/2)
            return x[:, :, half_dim:]
        latent_repr1 = Lambda(slice1)(latent_repr)
        latent_repr2 = Lambda(slice2)(latent_repr)
        print("latent_repr", latent_repr.shape)
        print("l1", latent_repr1.shape)
        print("l2", latent_repr2.shape)
        return latent_repr1, latent_repr2

    
    def build_split_conv_decoder(self, invec1, invec2, ae_nodes):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        ae_nodes = ae_nodes[::-1]
        z = keras.layers.concatenate([invec1,invec2], axis=-1)
        h4 = Conv1D(filters=ae_nodes[0], kernel_size=k, padding='same',activation=fx,kernel_regularizer=regularizers.l2(l2_val))(z)
        print("h4", h4.shape)
        u4 = UpSampling1D(m)(h4)
        print("u4", u4.shape)
        for n in ae_nodes[1:-1]:
            h4 = Conv1D(filters=n, kernel_size=k, padding='same',activation=fx,kernel_regularizer=regularizers.l2(l2_val))(u4)
            print("h4", n, h4.shape)
            u4 = UpSampling1D(m)(h4)
            print("u4", n, u4.shape)
        # normalize the values between -1 and 1 ???
        decoded = Conv1D(1, kernel_size=k, padding='same')(u4)
        print("decoded", decoded.shape)
        return decoded

    


    def build_final_style_classifier(self):
        fx, l2_val = 'relu', self.l2_val
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx))
        model.add(Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx))
        model.add(Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx))
        model.add(Dense(self.num_styles, activation='softmax'))
        reconstructed = Input(shape=(self.img_shape))
        label_style = model(reconstructed)
        return Model(reconstructed, label_style, name="FIN")



    def build_enc_style_classifier(self, name):
        fx, l2_val = 'relu', self.l2_val
        model = Sequential()
        model.add(Flatten(input_shape=self.enc_shape))
        model.add(Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx))
        model.add(Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx))
        model.add(Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx))
        model.add(Dense(self.num_styles, activation='softmax'))
        encoded = Input(shape=(self.enc_shape))
        label_style = model(encoded)
        return Model(encoded, label_style, name=name)


    def build_adv_style_classifier(self):
        fx, l2_val = 'relu', self.l2_val
        hp_lambda = self.hp_lambda
        Flip = GradientReversal(hp_lambda)
        encoded = Input(shape=(self.enc_shape))
        layer = Flip(encoded)
        h = Flatten(input_shape=self.enc_shape)(layer)
        h1 = Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx)(h)
        h2 = Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx)(h1)
        h3 = Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx)(h2)
        label_style = Dense(self.num_styles, activation='softmax')(h3)
        # reverse gradients during back-prop
        return Model(encoded, label_style, name="ADV")


    def build_noisy_adv_style_classifier(self):
        fx, l2_val, noise_val = 'relu', self.l2_val, self.noise
        hp_lambda = self.hp_lambda
        Flip = GradientReversal(hp_lambda)
        encoded = Input(shape=(self.enc_shape))
        layer = Flip(encoded)
        # add a gaussian noise layer with some noise value
        noisy = GaussianNoise(noise_val)(layer)
        h = Flatten(input_shape=self.enc_shape)(noisy)
        h1 = Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx)(h)
        h2 = Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx)(h1)
        h3 = Dense(256,kernel_regularizer=regularizers.l2(l2_val),activation=fx)(h2)
        label_style = Dense(self.num_styles, activation='softmax')(h3)
        # reverse gradients during back-prop
        return Model(encoded, label_style, name="ADV")

    

    # Fully-Connected ENcoder/DEcoder
    def build_split_encoder(self, ae_nodes):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        img = Input(shape=self.img_shape)
        h = Flatten()(img)
        for n in ae_nodes:
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
        latent_repr = Dense(self.latent_dim)(h)
        # split latent representation here
        def slice1(x):
            half_dim = int(self.latent_dim/2)
            return x[:, :half_dim]
        def slice2(x):
            half_dim = int(self.latent_dim/2)
            return x[:, half_dim:]
        latent_repr1 = Lambda(slice1)(latent_repr)
        latent_repr2 = Lambda(slice2)(latent_repr)
        print("l1", latent_repr1.shape)
        print("l2", latent_repr2.shape)
        return Model(img, [latent_repr1, latent_repr2], name="ENC")

    
    def build_merge_decoder(self, ae_nodes):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        ae_nodes = ae_nodes[::-1]
        model = Sequential()
        for n in ae_nodes:
            model.add(Dense(n, activation=fx, input_dim=self.latent_dim, kernel_regularizer=regularizers.l2(l2_val)))
            model.add(Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val)))
            model.add(Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val)))
        model.add(Dense(np.prod(self.img_shape)))
        model.add(Reshape(self.img_shape))
        model.summary()
        z1 = Input(shape=(self.split_latent_dim,))
        z2 = Input(shape=(self.split_latent_dim,))
        # merge z1 and z2 here
        z = keras.layers.concatenate([z1,z2], axis=-1)
        img = model(z)
        return Model([z1, z2], img, name="DEC")


    def build_dual_encoder(self, ae_nodes, name):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        img = Input(shape=self.img_shape)
        h = Flatten()(img)
        for n in ae_nodes:
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
        latent_repr = Dense(self.split_latent_dim)(h)
        return Model(img, latent_repr, name=name)
    

    def build_mean_encoder(self, ae_nodes, name):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        img = Input(shape=self.img_shape)        
        h = Flatten()(img)
        for n in ae_nodes:
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
        print(h.shape)
        latent_repr = Dense(output_dim=self.split_latent_dim)(h)
        def mean(x):
            print("x", x.shape)
            m = K.mean(x, axis=-1, keepdims=True)
            print("m", m.shape, m, m[0])
            r = K.repeat_elements(m, x.shape[1], 1)
            print("r", r.shape)
            return r
        latent_repr = Lambda(mean, output_shape=(self.split_latent_dim,))(latent_repr)
        return Model(img, latent_repr, name=name)
    


    def build_random_encoder(self, ae_nodes, name):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        img = Input(shape=self.img_shape)        
        h = Flatten()(img)
        for n in ae_nodes:
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
            h = Dense(n, activation=fx, kernel_regularizer=regularizers.l2(l2_val))(h)
        print(h.shape)
        latent_repr = Dense(output_dim=self.split_latent_dim)(h)
        def random(x):
            print("x", x.shape)
            r = K.random_uniform(x.shape, minval=-1, maxval=1)
            print("r", r.shape)
            return r
        latent_repr = Lambda(random, output_shape=(self.split_latent_dim,))(latent_repr)
        return Model(img, latent_repr, name=name)

    
    
    def build_merge_dual_decoder(self, ae_nodes):
        fx, m, k, l2_val = 'relu', 2, 3, self.l2_val
        model = Sequential()
        for n in ae_nodes:
            model.add(Dense(n, activation=fx, input_dim=self.latent_dim))
            model.add(Dense(n, activation=fx))
            model.add(Dense(n, activation=fx))
        model.add(Dense(np.prod(self.img_shape)))
        model.add(Reshape(self.img_shape))
        model.summary()
        z1 = Input(shape=(self.split_latent_dim,))
        z2 = Input(shape=(self.split_latent_dim,))
        # merge z1 and z2 here
        z = keras.layers.concatenate([z1,z2], axis=-1)
        img = model(z)
        return Model([z1, z2], img, name="DEC")
    
