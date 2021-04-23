from scipy.io.wavfile import read
#import matplotlib.pyplot as plt
from os import walk
import os
from os import mkdir,makedirs,listdir
import pandas as pd
import numpy as np
from os.path import isfile,isdir,getsize
import shutil
# from keras.preprocessing.image import ImageDataGenerator
# import keras
# from keras.models import Sequential
# from keras.layers import Input, Dense
# from keras.applications import ResNet50
# from keras.applications.resnet50 import preprocess_input
# from keras.optimizers import SGD,Adam
# print(keras.__version__)

df1=pd.read_csv('mffc_processed.csv')
print(df1.head(2))
#
# f_path='split/'
# train_folder=f_path+'/mfcc_train'
# test_folder=f_path+'/mfcc_test'


len_data=df1['file_path'].count()
train_example=50000
test_example=len_data-train_example
permutation=np.random.permutation(len_data)

train_set=[]
train_label=[]
test_set=[]
test_label=[]
print(permutation[:train_example])
for i,j in zip(permutation[:train_example],df1.index):
    train_set.append(df1['file_path'][i])
    train_label.append(df1['label'][i])
    print('Train : ')
    print(i)
#     print(train_set)
#     print(train_label)
print(permutation[-test_example:])
for i,j in zip(permutation[-test_example:],df1.index):
    test_set.append(df1['file_path'][i])
    test_label.append(df1['label'][i])
    print('Test:')
    print(i)
#     print(test_set)
#     print(test_label)
# ######################################################
# train_folder='split/mfcc_train'
# try:
#     for f,i in zip(train_set,train_label):
#         if i=='one':
#             shutil.copy2(f, train_folder+'/one/')
#         elif i=='two':
#             shutil.copy2(f, train_folder+'/two/')
#         elif i=='three':
#             shutil.copy2(f, train_folder+'/three/')
#         elif i=='four':
#             shutil.copy2(f, train_folder+'/four/')
#         elif i=='five':
#             shutil.copy2(f, train_folder+'/five/')
#         elif i=='six':
#             shutil.copy2(f, train_folder+'/six/')
#         elif i=='seven':
#             shutil.copy2(f, train_folder+'/seven/')
#         elif i=='eight':
#             shutil.copy2(f, train_folder+'/eight/')
#         elif i=='nine':
#             shutil.copy2(f, train_folder+'/nine/')
#         elif i=='zero':
#             shutil.copy2(f, train_folder+'/zero/')
#         elif i=='left':
#             shutil.copy2(f, train_folder+'/left/')
#         elif i=='right':
#             shutil.copy2(f, train_folder+'/right/')
#         elif i=='go':
#             shutil.copy2(f, train_folder+'/go/')
#         elif i=='stop':
#             shutil.copy2(f, train_folder+'/stop/')
#         elif i=='yes':
#             shutil.copy2(f, train_folder+'/yes/')
#         elif i=='no':
#             shutil.copy2(f, train_folder+'/no/')
#         else:
#             print('can not match folder')
#     print('All are splitted successfully...')
# except:
#     print('Something creates Problem :(')
# ##########################################
test_folder='split/mfcc_test'
try:
    for f,i in zip(test_set,test_label):
        if i=='one':
            shutil.copy2(f, test_folder+'/one/')
        elif i=='two':
            shutil.copy2(f, test_folder+'/two/')
        elif i=='three':
            shutil.copy2(f, test_folder+'/three/')
        elif i=='four':
            shutil.copy2(f, test_folder+'/four/')
        elif i=='five':
            shutil.copy2(f, test_folder+'/five/')
        elif i=='six':
            shutil.copy2(f, test_folder+'/six/')
        elif i=='seven':
            shutil.copy2(f, test_folder+'/seven/')
        elif i=='eight':
            shutil.copy2(f, test_folder+'/eight/')
        elif i=='nine':
            shutil.copy2(f, test_folder+'/nine/')
        elif i=='zero':
            shutil.copy2(f, test_folder+'/zero/')
        elif i=='left':
            shutil.copy2(f, test_folder+'/left/')
        elif i=='right':
            shutil.copy2(f, test_folder+'/right/')
        elif i=='go':
            shutil.copy2(f, test_folder+'/go/')
        elif i=='stop':
            shutil.copy2(f, test_folder+'/stop/')
        elif i=='yes':
            shutil.copy2(f, test_folder+'/yes/')
        elif i=='no':
            shutil.copy2(f, test_folder+'/no/')
        else:
            print('can not match folder')
    print('All are splitted test successfully...')
except:
    print('Something creates Problem :(')