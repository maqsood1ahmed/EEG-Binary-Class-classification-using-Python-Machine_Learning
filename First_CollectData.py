###
#Developed by Muhammad Salman and Maqsood Ahmed students of NED University of Engineering and Technology
###

#Data Collection program to collect data from Emotiv Epoc device
#using edk.dll, edk_utils.dll, edk.lib and edk_utils.lib libraries which comes Emotiv SDK Premium Edition

#Data Taken for two thoughts e.g A and B and for each A and B there are number of trials and each trial is 13 seconds long for 14 sensors. 5sec for rest, 3sec for ready state
#and 5sec for thinking state

#All Trials are stored in pandas DataFrame format and then save as username.csv format for processing and model creation


import os
import ctypes
import sys
from ctypes import *
from numpy import *
import numpy as np
import time
from time import gmtime, strftime
import random
import pandas as pd
from ctypes.util import find_library
print ctypes.util.find_library('edk.dll')  
print os.path.exists('.\\edk.dll')
libEDK = cdll.LoadLibrary(".\\edk.dll")

class CollectData:
    def __init__(self,self_kivy,profileloc,profilename):  
        
        
        self.choice = ''
        self.time = 1
        self.profile_loc = profileloc
        self.profile_name = profilename
        
        self.self_kivy = self_kivy
        self.exit = 0

        if os.path.exists(self.profile_loc+self.profile_name+'.csv'): #if exist then load
            self.original_df = pd.read_csv(self.profile_loc+self.profile_name+'.csv')
            self.curr_index = (self.original_df['Index_No'][-1:].values)[0]+1
            self.trial_no = (self.original_df['Trial_No'].max())
            print(self.original_df.head())
        else:
            self.curr_index = 0
            self.trial_no = 0  
            self.original_df = pd.DataFrame(columns = ['System_Time',
                  'Index_No',
                  'Trial_No',
                  'Label',
                  'Time_Of_Trial(sec)',
                  'Sensor1',
                  'Sensor2',
                  'Sensor3',
                  'Sensor4',
                  'Sensor5',
                  'Sensor6',
                  'Sensor7',
                  'Sensor8',
                  'Sensor9',
                  'Sensor10',
                  'Sensor11',
                  'Sensor12',
                  'Sensor13',
                  'Sensor14'])
            
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

        print "Buffer size in secs:"

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
                print(self.nSamplesTaken[0])
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
                    
                    if (self.choice == 'A' or self.choice == 'B') and self.time<=13:
                        temp_data_dict = {'System_Time':[strftime("%Y-%m-%d %H:%M:%S", gmtime()) for i in range(128)],
                                          'Index_No':[(self.curr_index+ii) for ii in range(128)],
                                          'Trial_No':[self.trial_no for i in range(128)],
                                          'Label':[self.choice for i in range(128)],
                                          'Time_Of_Trial(sec)':[(self.time) for i in range(128)],
                                          'Sensor1':y[0],
                                          'Sensor2':y[1],
                                          'Sensor3':y[2],
                                          'Sensor4':y[3],
                                          'Sensor5':y[4],
                                          'Sensor6':y[5],
                                          'Sensor7':y[6],
                                          'Sensor8':y[7],
                                          'Sensor9':y[8],
                                          'Sensor10':y[9],
                                          'Sensor11':y[10],
                                          'Sensor12':y[11],
                                          'Sensor13':y[12],
                                          'Sensor14':y[13]}
                        self.curr_index = self.curr_index + 128
                        
                        temp_df = pd.DataFrame(temp_data_dict ,columns = ['System_Time','Index_No','Trial_No','Label','Time_Of_Trial(sec)','Sensor1','Sensor2','Sensor3','Sensor4','Sensor5','Sensor6','Sensor7','Sensor8','Sensor9','Sensor10','Sensor11','Sensor12','Sensor13','Sensor14'])
                        self.time = self.time + 1
                        self.original_df = self.original_df.append(temp_df)
                        
                        if self.time==14:                            
                            self.original_df.to_csv(self.profile_loc+self.profile_name+'.csv',index=False)
                            print(self.trial_no+1)
                            
                    if self.time==1 or self.time==14:
                        self.choice = random.choice(['A','B'])
                        self.trial_no = self.trial_no +1
                        self.time = 1

                    if self.time>=1 and self.time<=5:
                        self.self_kivy.lbl3.text = 'Please Take a Rest'
                        self.self_kivy.text = ''
                        
                    elif self.time>=6 and self.time<=8:
                        self.self_kivy.lbl3.text = 'Please Ready for Action'
                        
                    elif self.time>=9 and self.time<=13:
                        self.self_kivy.lbl3.text = 'Think about action ' + self.choice
                        self.self_kivy.text = ''
                        if self.time==13:
                            self.self_kivy.text = os.path.join(os.getcwd(),(self.choice+'.png'))
                        
                else:
                    print('No')
            
            time.sleep(1.02)

            if self.exit==1:
                self.disconnect_engine()
                self.self_kivy.lbl3.text = 'Disconnected'
                print("Engine Disconnected")
                break

    def disconnect_engine(self):
        libEDK.EE_DataFree(self.hData)
        libEDK.EE_EngineDisconnect()
        libEDK.EE_EmoStateFree(self.eState)
        libEDK.EE_EmoEngineEventFree(self.eEvent)

    def disconnect(self):
        self.exit=1

    
