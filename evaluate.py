import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

####################################################################

import numpy as np
import h5py
import scipy.io
import librosa
import soundfile as sound
import keras
import tensorflow
import librosa
import soundfile as sound
import keras
import tensorflow
import pandas as pd
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from plots import plot_confusion_matrix

print("Librosa version = ",librosa.__version__)
print("Pysoundfile version = ",sound.__version__)
print("keras version = ",keras.__version__)
print("tensorflow version = ",tensorflow.__version__)

####################################################################

data_train_path = '../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/'
train_file = data_train_path + 'evaluation_setup/fold1_train.csv'
#data_path =  '../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/' #remark
#test_file = data_path + 'evaluation_setup/fold1_evaluate.csv' #remark
data_test_path =  '../datasets/TAU-urban-acoustic-scenes-2020-mobile-evaluation/'
test_file = data_test_path + 'evaluation_setup/fold1_test.csv' 

feature_train = 'features/LM_train_raw'
feature_test = 'features/LM_test_raw' #name
model_path = 'models/DCASE_1a_Task_development_5ch_2.h5' #name
result_path = 'results/test.csv' #name
channels = '5ch' #name

sr = 44100
SampleDuration = 10
NumFreqBins = 128
NumFFTPoints = 2048
HopLength = int(NumFFTPoints/2)
NumTimeBins = int(np.ceil(SampleDuration*sr/HopLength))
num_audio_channels = 1

####################################################################

train_df = pd.read_csv(train_file,sep='\t', encoding='ASCII')
test_df = pd.read_csv(test_file,sep='\t', encoding='ASCII')
wavpaths_train = train_df['filename'].tolist()
wavpaths_test = test_df['filename'].tolist()

#ClassNames = np.unique(test_df['scene_label']) #remark
#y_test_labels = test_df['scene_label'].astype('category').cat.codes.values #remark
#y_test_labels.setflags(write=1) #remark

####################################################################

if os.path.exists(feature_train+'.npy'):
    LM_train_raw = np.load(feature_train+'.npy')
else:
    LM_train = np.zeros((len(wavpaths_train),NumFreqBins,NumTimeBins,num_audio_channels),'float32')
    print("number of train files : {}".format(len(wavpaths_train)))
    for i in range(len(wavpaths_train)):
        mono,fs = sound.read(data_train_path + wavpaths_train[i],stop=SampleDuration*sr)
        for channel in range(num_audio_channels):
            if len(mono.shape)==1:
                mono = np.expand_dims(mono,-1)
            LM_train[i,:,:,channel]= librosa.feature.melspectrogram(mono[:,channel], 
                                           sr=sr,
                                           n_fft=NumFFTPoints,
                                           hop_length=HopLength,
                                           n_mels=NumFreqBins,
                                           fmin=0.0,
                                           fmax=sr/2,
                                           htk=True,
                                           norm=None)
    LM_train_raw = np.log(LM_train+1e-8)
    np.save(feature_train,LM_train_raw)

if os.path.exists(feature_test+'.npy'):
    LM_test_raw = np.load(feature_test+'.npy')
else:
    LM_test = np.zeros((len(wavpaths_test),NumFreqBins,NumTimeBins,num_audio_channels),'float32')
    print("number of eval files : {}".format(len(wavpaths_test)))
    for i in range(len(wavpaths_test)):
        mono,fs = sound.read(data_test_path + wavpaths_test[i],stop=SampleDuration*sr)
        for channel in range(num_audio_channels):
            if len(mono.shape)==1:
                mono = np.expand_dims(mono,-1)
            LM_test[i,:,:,channel]= librosa.feature.melspectrogram(mono[:,channel], 
                                           sr=sr,
                                           n_fft=NumFFTPoints,
                                           hop_length=HopLength,
                                           n_mels=NumFreqBins,
                                           fmin=0.0,
                                           fmax=sr/2,
                                           htk=True,
                                           norm=None)
    LM_test_raw = np.log(LM_test+1e-8)
    np.save(feature_test,LM_test_raw)

####################################################################
    
classes = ['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']
class_list = []
class_overall_mean_list = []

for inds_class in classes:
    class_tmp = np.where(train_df['filename'].str.contains(inds_class+'-')==True)[0]
    class_list.append(list(class_tmp))

for i in range(len(classes)):
    mean_tmp = np.zeros((len(class_list[i]),NumFreqBins,1),'float32')
    overall_mean_tmp = np.zeros((NumFreqBins,1),'float32')
    for j in range(len(class_list[i])):
        for FreqBin in range(NumFreqBins):
            if i == 0:
                mean_tmp[j,FreqBin] = np.mean(LM_train_raw[j,FreqBin,:,0])
            else:
                mean_tmp[j,FreqBin] = np.mean(LM_train_raw[int(class_list[i][0]) + j,FreqBin,:,0])
    #print (mean_tmp.shape)
    for FreqBin in range(NumFreqBins):
        overall_mean_tmp[FreqBin] = np.mean(mean_tmp[:,FreqBin,0])
    #print (overall_mean_tmp.shape)
    class_overall_mean_list.append(list(overall_mean_tmp))
          
class_overall_mean = class_overall_mean_list[0]
for i in range(1,len(classes)):
    class_overall_mean = np.concatenate((class_overall_mean,class_overall_mean_list[i]),axis=-1)
#print (len(class_overall_mean), len(class_overall_mean[1]))

####################################################################

devices = ['a','b','c','s1','s2','s3']
device_list = []
device_overall_mean_list = []
for inds_device in devices:
    device_tmp = np.where(train_df['filename'].str.contains('-'+inds_device)==True)[0]
    device_list.append(list(device_tmp))
    
for i in range(len(devices)):
    mean_tmp = np.zeros((len(device_list[i]),NumFreqBins,1),'float32')
    overall_mean_tmp = np.zeros((NumFreqBins,1),'float32')
    for j in range(len(device_list[i])):
        for FreqBin in range(NumFreqBins):
            mean_tmp[j,FreqBin] = np.mean(LM_train_raw[device_list[i][0],FreqBin,:,0])
    #print (mean_tmp.shape)
    for FreqBin in range(NumFreqBins):
        overall_mean_tmp[FreqBin] = np.mean(mean_tmp[:,FreqBin,0])
    #print (overall_mean_tmp.shape)
    device_overall_mean_list.append(list(overall_mean_tmp))

device_overall_mean = device_overall_mean_list[0]
for i in range(1,len(devices)):
    device_overall_mean = np.concatenate((device_overall_mean,device_overall_mean_list[i]),axis=-1)
#print (len(device_overall_mean), len(device_overall_mean[1]))

####################################################################

class_var = np.var(class_overall_mean,axis=1)
class_var = class_var[:,np.newaxis]
class_std = np.std(class_overall_mean,axis=1)
class_std = class_std[:,np.newaxis]

device_var = np.var(device_overall_mean,axis=1)
device_var = device_var[:,np.newaxis]
device_std = np.std(device_overall_mean,axis=1)
device_std = device_std[:,np.newaxis]

####################################################################

def deltas(X_in):
    X_out = (X_in[:,:,2:,:]-X_in[:,:,:-2,:])/10.0
    X_out = X_out[:,:,1:-1,:]+(X_in[:,:,4:,:]-X_in[:,:,:-4,:])/5.0
    return X_out

if os.path.exists(feature_test.replace('raw',channels)+'.npy'):
    LM_test = np.load(feature_test.replace('raw',channels)+'.npy')
else:
    LM_test_class_std = LM_test_raw[:,:,:,0] * class_std
    LM_test_class_std = LM_test_class_std[:,:,:,np.newaxis]
    LM_test_device_std = LM_test_raw[:,:,:,0] / device_std
    LM_test_device_std = LM_test_device_std[:,:,:,np.newaxis]   
    
    LM_deltas_test = deltas(LM_test_raw)
    #LM_deltas_test_class_std = LM_deltas_test[:,:,:,0] * class_std
    #LM_deltas_test_class_std = LM_deltas_test_class_std[:,:,:,np.newaxis]
    #LM_deltas_test_device_std = LM_deltas_test[:,:,:,0] / device_std
    #LM_deltas_test_device_std = LM_deltas_test_device_std[:,:,:,np.newaxis]
    
    LM_deltas_deltas_test = deltas(LM_deltas_test)
    #LM_deltas_deltas_test_class_std = LM_deltas_deltas_test[:,:,:,0] * class_std
    #LM_deltas_deltas_test_class_std = LM_deltas_deltas_test_class_std[:,:,:,np.newaxis]
    #LM_deltas_deltas_test_device_std = LM_deltas_deltas_test[:,:,:,0] / device_std
    #LM_deltas_deltas_test_device_std = LM_deltas_deltas_test_device_std[:,:,:,np.newaxis]
    
    LM_test = np.concatenate((LM_test_raw[:,:,4:-4,:],
                              LM_deltas_test[:,:,2:-2,:],
                              LM_deltas_deltas_test,
                              LM_test_class_std[:,:,4:-4,:],
                              LM_test_device_std[:,:,4:-4,:],
                              #LM_deltas_test_class_std[:,:,2:-2,:],
                              #LM_deltas_test_device_std[:,:,2:-2,:],
                              #LM_deltas_deltas_test_class_std,
                              #LM_deltas_deltas_test_device_std
                             ),axis=-1)
    np.save(feature_test.replace('raw',channels),LM_test)
    
print(LM_test.shape)

####################################################################

best_model = keras.models.load_model(model_path)
output_prob_test = best_model.predict(LM_test)
y_pred_test = np.argmax(output_prob_test,axis=1)

print (len(wavpaths_test))
print (len(output_prob_test))
print (len(y_pred_test))

with open(result_path, 'w') as f:
    for i in range(len(wavpaths_test)):
        if i == 0:
            f.write('filename'+'\t')
            f.write('scene_label'+'\t')
            for cs in classes:
                if cs == 'tram':
                    f.write(str(cs))
                else:
                    f.write(str(cs)+'\t')
            f.write('\n') 
        f.write(str(wavpaths_test[i].replace('audio/',''))+'\t')
        f.write(str(classes[int(y_pred_test[i])])+'\t')
        n=0
        for prob in output_prob_test[i]:
            if n == 9:
                f.write(str(prob))
            else:
                f.write(str(prob)+'\t')
            n=n+1
        f.write('\n')
#exit ()

####################################################################

print("overall:")
overall_accuracy = np.sum(y_pred_test==y_test_labels)/LM_test.shape[0]
print("accuracy: ", '%.1f' % (overall_accuracy*100))
logloss_overall = log_loss(y_test_labels, output_prob_test)
print("logloss: ", '%.3f' % logloss_overall)
print()
conf_matrix = confusion_matrix(y_test_labels,y_pred_test)
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
precision_by_class = np.diagonal(conf_mat_norm_precision)
mean_recall = np.mean(recall_by_class)
mean_precision = np.mean(precision_by_class)
#print("per-class accuracy (recall): ",recall_by_class*100)
#print("per-class precision: ",precision_by_class*100)
#print("mean per-class recall: ",mean_recall*100)
#print("mean per-class precision: ",mean_precision*100)
print("class_wise:")
classes = ['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']
class_list = []
for inds_class in classes:
    class_tmp = np.where(test_df['filename'].str.contains(inds_class+'-')==True)[0]
    class_list.append(list(class_tmp))
i=0
for file_list in class_list:
    print(classes[i])
    class_accuracy = np.sum(y_pred_test[file_list]==y_test_labels[file_list])/len(file_list)
    class_logloss = log_loss(y_test_labels[file_list], output_prob_test[file_list], labels=list(range(len(classes))))
    print("accuracy: ", '%.1f' % (class_accuracy*100))
    print("logloss:  ", '%.3f' % class_logloss)
    print()
    i=i+1
print("devices_wise:")
devices = ['a','b','c','s1','s2','s3','s4','s5','s6']
device_list = []
for inds_device in devices:
    device_tmp = np.where(test_df['filename'].str.contains('-'+inds_device)==True)[0]
    device_list.append(list(device_tmp))
i=0
for file_list in device_list:
    print(devices[i])
    device_accuracy = np.sum(y_pred_test[file_list]==y_test_labels[file_list])/len(file_list)
    device_logloss = log_loss(y_test_labels[file_list], output_prob_test[file_list],labels=list(range(len(classes))))
    print("accuracy: ", '%.1f' % (device_accuracy*100))
    print("logloss:  ", '%.3f' % device_logloss)
    print()
    i=i+1

####################################################################
    
overall_accuracy = np.sum(y_pred_test==y_test_labels)/LM_test.shape[0]
print("overall accuracy: ", overall_accuracy)

conf_matrix = confusion_matrix(y_test_labels,y_pred_test)
conf_mat_norm_recall = conf_matrix.astype('float32')/conf_matrix.sum(axis=1)[:,np.newaxis]
conf_mat_norm_precision = conf_matrix.astype('float32')/conf_matrix.sum(axis=0)[:,np.newaxis]
recall_by_class = np.diagonal(conf_mat_norm_recall)
precision_by_class = np.diagonal(conf_mat_norm_precision)
mean_recall = np.mean(recall_by_class)
mean_precision = np.mean(precision_by_class)

print("per-class accuracy (recall): ",recall_by_class)
print("per-class precision: ",precision_by_class)
print("mean per-class recall: ",mean_recall)
print("mean per-class precision: ",mean_precision)