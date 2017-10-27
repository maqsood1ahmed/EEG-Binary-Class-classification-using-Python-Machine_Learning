###
#Developed by Muhammad Salman and Maqsood Ahmed students of NED University of Engineering and Technology karachi, Pakistan
###

#Load username.pkl model to predict live eeg data output will be A or B then we
#can send that action to any application through socket

import os
import ctypes
import sys
from ctypes import *
from numpy import *
import time
from ctypes.util import find_library
print ctypes.util.find_library('edk.dll')  
print os.path.exists('.\\edk.dll')
libEDK = cdll.LoadLibrary(".\\edk.dll")
import numpy as np
from sklearn.externals import joblib

class LivePrediction:
    def __init__(self,self_kivy,profilepath,profilename):  
        
        self.profile_name = profilename
        self.profile_path = profilepath

        self.self_kivy = self_kivy
        self.action_type = 'no_action'
        self.is_reference_data_taken = True
        self.time = 0
        self.reference_data = np.zeros((14,2560))
        self.reference_data_features = np.zeros((1,140))
        self.live_data =np.zeros((14,640))

        self.clf = joblib.load((self.profile_path+self.profile_name+'_model.pkl'))
        print(self.profile_path+self.profile_name+'_model.pkl')
        self.ED_COUNTER = 0
        self.ED_INTERPOLATED=1
        self.ED_RAW_CQ=2
        self.ED_AF3=3
        self.ED_F7=4
        self.ED_F3=5
        self.ED_FC5=6
        self.ED_T7=7
        self.ED_P7=8
        self.ED_O1=9
        self.ED_O2=10
        self.ED_P8=11
        self.ED_T8=12
        self.ED_FC6=13
        self.ED_F4=14
        self.ED_F8=15
        self.ED_AF4=16
        self.ED_GYROX=17
        self.ED_GYROY=18
        self.ED_TIMESTAMP=19
        self.ED_ES_TIMESTAMP=20
        self.ED_FUNC_ID=21
        self.ED_FUNC_VALUE=22
        self.ED_MARKER=23
        self.ED_SYNC_SIGNAL=24

        self.targetChannelList = [self.ED_RAW_CQ,self.ED_AF3, self.ED_F7, self.ED_F3, self.ED_FC5, self.ED_T7,self.ED_P7, self.ED_O1, self.ED_O2, self.ED_P8, self.ED_T8,self.ED_FC6, self.ED_F4, self.ED_F8, self.ED_AF4, self.ED_GYROX, self.ED_GYROY, self.ED_TIMESTAMP, self.ED_FUNC_ID, self.ED_FUNC_VALUE, self.ED_MARKER, self.ED_SYNC_SIGNAL]
        self.eEvent      = libEDK.EE_EmoEngineEventCreate()
        self.eState      = libEDK.EE_EmoStateCreate()
        self.userID            = c_uint(0)
        self.nSamples   = c_uint(0)
        self.nSam       = c_uint(0)
        self.nSamplesTaken  = pointer(self.nSamples)
        self.data     = pointer(c_double(0))
        self.user     = pointer(self.userID)
        self.composerPort          = c_uint(1726)
        self.secs      = c_float(1)
        self.datarate    = c_uint(0)
        self.readytocollect    = False
        self.option      = c_int(0)
        self.state     = c_int(0)
        
        print libEDK.EE_EngineConnect("Emotiv Systems-5")
        if libEDK.EE_EngineConnect("Emotiv Systems-5") != 0:
            print "Emotiv Engine start up failed."

        print "Start receiving EEG Data! Press any key to stop logging...\n"

        self.hData = libEDK.EE_DataCreate()
        libEDK.EE_DataSetBufferSizeInSec(self.secs)

        self.j=0

        print "Buffer size in secs:"
        

    def do_fft(self,all_channel_data): 
                """
                Do fft in each channel for all channels.
                Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
                Output: FFT result with dimension N x M. N denotes number of channel and M denotes number of FFT data from each channel.
                """
                data_fft = map(lambda x: np.fft.fft(x),all_channel_data)

                return data_fft

    def get_frequency(self,all_channel_data): 
            """
            Get frequency from computed fft for all channels. 
            Input: Channel data with dimension N x M. N denotes number of channel and M denotes number of EEG data from each channel.
            Output: Frequency band from each channel: Delta, Theta, Alpha, Beta, and Gamma.
            """
            #Length data channel
            L = len(all_channel_data[0])

            #Sampling frequency
            Fs = 128

            #Get fft data
            data_fft = self.do_fft(all_channel_data)

            #Compute frequency
            frequency = map(lambda x: abs(x/L),data_fft)
            frequency = map(lambda x: x[: L/2+1]*2,frequency)

            #List frequency
            delta = map(lambda x: x[L*1/Fs-1: L*4/Fs],frequency)
            theta = map(lambda x: x[L*4/Fs-1: L*8/Fs],frequency)
            alpha = map(lambda x: x[L*5/Fs-1: L*13/Fs],frequency)
            beta = map(lambda x: x[L*13/Fs-1: L*30/Fs],frequency)
            gamma = map(lambda x: x[L*30/Fs-1: L*50/Fs],frequency)

            return delta,theta,alpha,beta,gamma


    def get_feature(self,all_channel_data): 
            #Get frequency data
            (delta,theta,alpha,beta,gamma) = self.get_frequency(all_channel_data)

            #Compute feature std
            delta_std = np.std(delta, axis=1)
            theta_std = np.std(theta, axis=1)
            alpha_std = np.std(alpha, axis=1)
            beta_std = np.std(beta, axis=1)
            gamma_std = np.std(gamma, axis=1)

            #Compute feature mean
            delta_m = np.mean(delta, axis=1)
            theta_m = np.mean(theta, axis=1)
            alpha_m = np.mean(alpha, axis=1)
            beta_m = np.mean(beta, axis=1)
            gamma_m = np.mean(gamma, axis=1)

            #Concate feature
            feature = np.array([delta_std,delta_m,theta_std,theta_m,alpha_std,alpha_m,beta_std,beta_m,gamma_std,gamma_m])
            feature = feature.T
            feature = feature.ravel()

            return feature

    def data_acq(self):
        while (1):
            state = libEDK.EE_EngineGetNextEvent(self.eEvent)
            if state == 0:
                eventType = libEDK.EE_EmoEngineEventGetType(self.eEvent)
                libEDK.EE_EmoEngineEventGetUserId(self.eEvent, self.user)
                if eventType == 16:
                    libEDK.EE_DataAcquisitionEnable(self.userID,True)
                    self.readytocollect = True
            
            if self.readytocollect==True:
                libEDK.EE_DataUpdateHandle(0, self.hData)
                libEDK.EE_DataGetNumberOfSample(self.hData,self.nSamplesTaken)
                if self.nSamplesTaken[0] == 128:
                    self.nSam=self.nSamplesTaken[0]
                    arr=(ctypes.c_double*self.nSamplesTaken[0])()
                    ctypes.cast(arr, ctypes.POINTER(ctypes.c_double))                         
                    data = array('d')
                    y = np.zeros((128,14))
                    for sampleIdx in range(self.nSamplesTaken[0]):
                        x = np.zeros(14)
                        for i in range(1,15):
                            libEDK.EE_DataGet(self.hData,self.targetChannelList[i],byref(arr), self.nSam)
                            x[i-1] = arr[sampleIdx]
                            
                        y[sampleIdx] = x
                    y = np.transpose(y)
                    
                    if self.is_reference_data_taken == False:
                        t = self.time
                        self.self_kivy.lbl4.text = 'Please First provide Neutral Thought Data for 20 seconds'
                        if t !=-1:
                            self.reference_data[:,(t*128):((t*128)+127)] = y[:,0:127]
                        self.time = self.time + 1

                        if self.time>19:
                            self.is_reference_data_taken = True
                            self.reference_data_features[0,:] = self.get_feature(self.reference_data)
                            self.time = 0

                    if self.is_reference_data_taken == True:
                        t = self.time

                        if t>=0 and t<=4:
                            self.live_data[:,(t*128):((t*128)+127)] = y[:,0:127]
                            self.time = self.time + 1
                        if t>4:
                            
                            original_data_features = self.get_feature(self.live_data)
                            X=np.zeros((1,140))
                            
                            x = original_data_features
                            X=np.zeros((1,140))
                            X[0]= x
                            y = []
                            
                            y = self.clf.predict(X)
            
                            if y[0]==1:
                                print('hy')
                                self.self_kivy.lbl4.text=''
                                self.self_kivy.text = os.getcwd()+'\\A.png' 
                                print('score is ',self.clf.score(X,y))

                            elif y[0]==2:
                                self.self_kivy.lbl4.text=''
                                print(y[0])
                                print(y[0],'for B')
                                self.self_kivy.text = os.getcwd()+'\\B.png'
                                print('score is ',self.clf.score(X,y))

                            self.time = 0   
                else:
                    print('No')
                    self.x = 'no_action'
                

            time.sleep(1.1)
            self.self_kivy.text = ''
            if self.j>=5:
                print(' ')
                self.disconnect_engine()
                print("Engine Disconnected")
                break

    def disconnect_engine(self):
        libEDK.EE_DataFree(self.hData)
        libEDK.EE_EngineDisconnect()
        libEDK.EE_EmoStateFree(self.eState)
        libEDK.EE_EmoEngineEventFree(self.eEvent)


