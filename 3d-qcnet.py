import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
import keras 
import nibabel as nib
import densenet
import os
import random
from sklearn.metrics import confusion_matrix

from keras.layers import Dense
from keras import Model
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from shutil import copy
import argparse

## Saved Model Path = 


parser = argparse.ArgumentParser(description='3DQCNet Implementaiton')

parser.add_argument("--mode",help ="To run Test model select 'test'; To run in Prediction mode select 'pred'",required=True)
parser.add_argument("--thresh",help ="Value for probabaility threshold",default=0.5,type=int)
parser.add_argument("--model_path",help ="The path for the model which will be loaded in 3D-QCNet",default='./models/3dqcnet_base-model_ep9.hdf5)

args = parser.parse_args()


mode = args.mode
prob_thresh = args.thresh
model_path = args.model_path

print(mode,prob_thresh,model_path)

PRED_FP = './input/'+mode 
DESTN_FP = './output/' 

def creport (yt, y_pred):
    cm1 = confusion_matrix(yt, y_pred)
    print(cm1)
    total1 = sum(sum(cm1))
    accuracy1 = float(cm1[0, 0] + cm1[1, 1]) / total1
    recall1 = float(cm1[0, 0]) / (cm1[0, 0] + cm1[0, 1])
    precision1 = float(cm1[0, 0]) / (cm1[0, 0] + cm1[1, 0])
    return accuracy1,recall1,precision1


### Get File paths (depends on train and test)

pred_file_path = PRED_FP
pred_file_names = []
gt_cls=[]
gt_cls_name=[]

if mode == 'test':
    for x in os.listdir(pred_file_path):
        for y in os.listdir(pred_file_path+'/'+x):
            pred_file_names.append(pred_file_path+'/'+x+'/'+y)
            gt_cls_name.append(x)
            gt_cls.append(0 if x =='bad' else 1)
elif mode == 'pred':
    for x in os.listdir(pred_file_path):
            pred_file_names.append(pred_file_path+'/'+x)
            gt_cls_name.append('NA')

else:
    print("ERROR: Select Mode as Test or Pred")
    exit()


print(gt_cls[-5:])
print(gt_cls_name[-5:])        
# print(pred_file_names)

### Load Model

dense_model = densenet.DenseNet3D(input_shape=(96,96,70,1),include_top=False, pooling = 'avg', depth =10)
preds = Dense(2,activation='softmax')(dense_model.output)
model = Model(dense_model.input,preds)
model.compile(optimizer=Adam(lr=1e-4),loss='binary_crossentropy',metrics=['acc'])


model.load_weights(model_path)
print("Model Loaded")


### load files
X_preds = []

for x in pred_file_names:               #### To test on subset of data, change this to pred_file_names[:10] for eg.
    
    img = nib.load(x).get_data()
    
    new_img = np.zeros((96,96,70))
    depth = np.minimum(img.shape[-1],new_img.shape[-1])
    for i in range(depth):
        t = img[:,:,i]
        tt = cv2.resize(t,(96,96))
        new_img[:,:,i] = tt

    img = np.array(new_img)
    
    print(img.shape)

    img = np.reshape(img,(96,96,70,1))

    X_preds.append(np.array(img))
X_preds = np.array(X_preds)
X_preds.shape


### Run Model on Volumes

preds = []

for j in range(len(X_preds)):
    print(X_preds[j:j+1].shape)
    p = model.predict(X_preds[j:j+1], verbose = 1)[0]
    print(j,p)
    preds.append(p)
    
preds = np.array(preds)
print("Preds.shape - ",preds.shape)


### Apply probability thresholds, transfer files into destination folder and generate results.csv

## Transfer Volumes into bad and good folders
# copyfile(src,dist)
pred_cls = []
# count_good = 0
# count_bad = 0
for p in preds[:,0]:
    if p>prob_thresh:
        pred_cls.append(0)
    else:
        pred_cls.append(1)

pred_cls = np.array(pred_cls)
pred_cls


### Only for Test ###
if mode == 'test':
    print("Metrics :")

    print(gt_cls[:10])
    print(pred_cls)
    ac,re,pr = creport(gt_cls,pred_cls)
    print("Accuracy - ", ac)
    print("Recall - ", re)
    print("Precision - ", pr)

#dest_folder = DESTN_FP
#
#if not os.path.exists(dest_folder+'bad/'):
#    os.makedirs(dest_folder+'bad/')
#if not os.path.exists(dest_folder+'good'):
#    os.makedirs(dest_folder+'good/')

results = []

for idx,i in enumerate(pred_cls):
    fname = pred_file_names[idx]
    img_name = fname.split('/')[-1]
    if (i==0):
        copy(fname,DESTN_FP+'bad/'+img_name)
        results.append([fname,'bad',gt_cls_name[idx]]) 
    else:
        results.append([fname,'good',gt_cls_name[idx]])
        copy(fname,DESTN_FP+'good/'+img_name)

print(results)

df = pd.DataFrame(results,columns=['FilePath','Predicted Class','GT Class'])

df.to_csv('./results-test.csv')