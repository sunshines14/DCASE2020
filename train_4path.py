import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

####################################################################

import numpy as np
import matplotlib.pyplot as plt
import h5py
import pandas as pd
import librosa
import soundfile as sound
import keras
import tensorflow
from keras.optimizers import SGD
from network_4path import model_resnet
from utils import LR_WarmRestart, MixupGenerator
from keras.utils import multi_gpu_model
from keras.callbacks import ModelCheckpoint

print("Librosa version = ",librosa.__version__)
print("Pysoundfile version = ",sound.__version__)
print("keras version = ",keras.__version__)
print("tensorflow version = ",tensorflow.__version__)

####################################################################

data_path = '../datasets/TAU-urban-acoustic-scenes-2020-mobile-development/'
train_file = data_path + 'evaluation_setup/fold1_train.csv'
val_file = data_path + 'evaluation_setup/fold1_evaluate.csv'

feature_train = 'features/LM_train_raw'
feature_val = 'features/LM_val_raw'
model_path = 'models/DCASE_1a_Task_development_21.h5'
channels = '9ch'
num_feature_channels = 9

sr = 44100
num_audio_channels = 1 
SampleDuration = 10

NumFreqBins = 128
NumFFTPoints = 2048
HopLength = int(NumFFTPoints/2)
NumTimeBins = int(np.ceil(SampleDuration*sr/HopLength))

max_lr = 0.1
batch_size = 32
#num_epochs = 510
num_epochs = 1022
mixup_alpha = 0.4
crop_length = 400

####################################################################

dev_train_df = pd.read_csv(train_file,sep='\t', encoding='ASCII')
dev_val_df = pd.read_csv(val_file,sep='\t', encoding='ASCII')
wavpaths_train = dev_train_df['filename'].tolist()
wavpaths_val = dev_val_df['filename'].tolist()
y_train_labels =  dev_train_df['scene_label'].astype('category').cat.codes.values
y_val_labels =  dev_val_df['scene_label'].astype('category').cat.codes.values

ClassNames = np.unique(dev_train_df['scene_label'])
NumClasses = len(ClassNames)

y_train = keras.utils.to_categorical(y_train_labels, NumClasses)
y_val = keras.utils.to_categorical(y_val_labels, NumClasses)

####################################################################

if os.path.exists(feature_train+'.npy'):
    LM_train_raw = np.load(feature_train+'.npy')
else:
    LM_train = np.zeros((len(wavpaths_train),NumFreqBins,NumTimeBins,num_audio_channels),'float32')
    print("number of train files : {}".format(len(wavpaths_train)))
    for i in range(len(wavpaths_train)):
        mono,fs = sound.read(data_path + wavpaths_train[i],stop=SampleDuration*sr)
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

if os.path.exists(feature_val+'.npy'):
    LM_val_raw = np.load(feature_val+'.npy')
else:
    LM_val = np.zeros((len(wavpaths_val),NumFreqBins,NumTimeBins,num_audio_channels),'float32')
    print("number of eval files : {}".format(len(wavpaths_val)))
    for i in range(len(wavpaths_val)):
        mono,fs = sound.read(data_path + wavpaths_val[i],stop=SampleDuration*sr)
        for channel in range(num_audio_channels):
            if len(mono.shape)==1:
                mono = np.expand_dims(mono,-1)
            LM_val[i,:,:,channel]= librosa.feature.melspectrogram(mono[:,channel], 
                                           sr=sr,
                                           n_fft=NumFFTPoints,
                                           hop_length=HopLength,
                                           n_mels=NumFreqBins,
                                           fmin=0.0,
                                           fmax=sr/2,
                                           htk=True,
                                           norm=None)
    LM_val_raw = np.log(LM_val+1e-8)
    np.save(feature_val,LM_val_raw)

####################################################################

classes = ['airport','bus','metro','metro_station','park','public_square','shopping_mall','street_pedestrian','street_traffic','tram']
class_list = []
class_overall_mean_list = []

for inds_class in classes:
    class_tmp = np.where(dev_train_df['filename'].str.contains(inds_class+'-')==True)[0]
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
    device_tmp = np.where(dev_train_df['filename'].str.contains('-'+inds_device)==True)[0]
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

if os.path.exists(feature_train.replace('raw',channels)+'.npy'):
    LM_train = np.load(feature_train.replace('raw',channels)+'.npy')
else:
    LM_train_class_std = LM_train_raw[:,:,:,0] * class_std
    LM_train_class_std = LM_train_class_std[:,:,:,np.newaxis]
    M_train_device_std = LM_train_raw[:,:,:,0] / device_std
    LM_train_device_std = LM_train_device_std[:,:,:,np.newaxis]
    
    LM_deltas_train = deltas(LM_train_raw)
    LM_deltas_train_class_std = LM_deltas_train[:,:,:,0] * class_std
    LM_deltas_train_class_std = LM_deltas_train_class_std[:,:,:,np.newaxis]
    LM_deltas_train_device_std = LM_deltas_train[:,:,:,0] / device_std
    LM_deltas_train_device_std = LM_deltas_train_device_std[:,:,:,np.newaxis]
    
    LM_deltas_deltas_train = deltas(LM_deltas_train)
    LM_deltas_deltas_train_class_std = LM_deltas_deltas_train[:,:,:,0] * class_std
    LM_deltas_deltas_train_class_std = LM_deltas_deltas_train_class_std[:,:,:,np.newaxis]
    LM_deltas_deltas_train_device_std = LM_deltas_deltas_train[:,:,:,0] / device_std
    LM_deltas_deltas_train_device_std = LM_deltas_deltas_train_device_std[:,:,:,np.newaxis]
    
    LM_train = np.concatenate((LM_train_raw[:,:,4:-4,:],
                               LM_deltas_train[:,:,2:-2,:],
                               LM_deltas_deltas_train,
                               LM_train_class_std[:,:,4:-4,:],
                               LM_train_device_std[:,:,4:-4,:],
                               LM_deltas_train_class_std[:,:,2:-2,:],
                               LM_deltas_train_device_std[:,:,2:-2,:],
                               LM_deltas_deltas_train_class_std,
                               LM_deltas_deltas_train_device_std
                              ),axis=-1)
    np.save(feature_train.replace('raw',channels),LM_train)

if os.path.exists(feature_val.replace('raw',channels)+'.npy'):
    LM_val = np.load(feature_val.replace('raw',channels)+'.npy')
else:
    LM_val_class_std = LM_val_raw[:,:,:,0] * class_std
    LM_val_class_std = LM_val_class_std[:,:,:,np.newaxis]
    LM_val_device_std = LM_val_raw[:,:,:,0] / device_std
    LM_val_device_std = LM_val_device_std[:,:,:,np.newaxis] 
    
    LM_deltas_val = deltas(LM_val_raw)
    LM_deltas_val_class_std = LM_deltas_val[:,:,:,0] * class_std
    LM_deltas_val_class_std = LM_deltas_val_class_std[:,:,:,np.newaxis]
    LM_deltas_val_device_std = LM_deltas_val[:,:,:,0] / device_std
    LM_deltas_val_device_std = LM_deltas_val_device_std[:,:,:,np.newaxis]
       
    LM_deltas_deltas_val = deltas(LM_deltas_val)
    LM_deltas_deltas_val_class_std = LM_deltas_deltas_val[:,:,:,0] * class_std
    LM_deltas_deltas_val_class_std = LM_deltas_deltas_val_class_std[:,:,:,np.newaxis]
    LM_deltas_deltas_val_device_std = LM_deltas_deltas_val[:,:,:,0] / device_std
    LM_deltas_deltas_val_device_std = LM_deltas_deltas_val_device_std[:,:,:,np.newaxis]
    
    LM_val = np.concatenate((LM_val_raw[:,:,4:-4,:],
                             LM_deltas_val[:,:,2:-2,:],
                             LM_deltas_deltas_val,
                             LM_val_class_std[:,:,4:-4,:],
                             LM_val_device_std[:,:,4:-4,:],
                             LM_deltas_val_class_std[:,:,2:-2,:],
                             LM_deltas_val_device_std[:,:,2:-2,:],
                             LM_deltas_deltas_val_class_std,
                             LM_deltas_deltas_val_device_std
                            ),axis=-1)
    np.save(feature_val.replace('raw',channels),LM_val)
    
print(LM_train.shape)
print(LM_val.shape)
####################################################################

model = model_resnet(NumClasses,
                     input_shape=[NumFreqBins,None,num_feature_channels*num_audio_channels], 
                     num_filters=24,
                     wd=1e-3)
model = multi_gpu_model(model, gpus=4)
model.compile(loss='categorical_crossentropy',
              optimizer =SGD(lr=max_lr,decay=0, momentum=0.9, nesterov=False),
              metrics=['accuracy'])

model.summary()

####################################################################

lr_scheduler = LR_WarmRestart(nbatch=np.ceil(LM_train.shape[0]/batch_size), Tmult=2,
                              initial_lr=max_lr, min_lr=max_lr*1e-4,
                              epochs_restart=[11.0, 31.0, 71.0, 151.0, 311.0, 631.0])
#epochs_restart=[3.0, 7.0, 15.0, 31.0, 63.0,127.0,255.0,511.0]

checkpoint = ModelCheckpoint(filepath=model_path, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')

TrainDataGen = MixupGenerator(LM_train, 
                              y_train, 
                              batch_size=batch_size,
                              alpha=mixup_alpha,
                              crop_length=crop_length)()

history = model.fit_generator(TrainDataGen,
                              validation_data=(LM_val, y_val),
                              epochs=num_epochs, 
                              verbose=1, 
                              workers=4,
                              max_queue_size=100,
                              callbacks=[lr_scheduler, checkpoint],
                              steps_per_epoch=np.ceil(LM_train.shape[0]/batch_size)) 