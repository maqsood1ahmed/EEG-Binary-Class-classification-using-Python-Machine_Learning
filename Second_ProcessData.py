###
#Developed by Muhammad Salman and Maqsood Ahmed students of NED University of Engineering and Technology karachi, Pakistan
###

#This program load data of a user and convert time-series data into frequency domain
#and then mean and standard deviation of delta, theta, alpha, beta and gamma is calculated and provide these features of each trial to svm grid-search cv to
#make a model with best C and gamma parameters then the .pkl model is stored for a user which will be used for real time data prediction


import csv
import numpy as np
import os
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.externals import joblib
import pandas as pd 
from sklearn.metrics import accuracy_score,classification_report
import time
from sklearn import cross_validation,preprocessing,neighbors
from sklearn.pipeline import Pipeline, FeatureUnion
import matplotlib.pyplot as plt
class ProcessData:
        
        def __init__(self,profilepath,profilename):
                self.profile_name = profilename
                self.profile_path = profilepath
                self.features = []
                self.labels = []
                self.feature_names = ['Delta_Mean','Theta_Mean','Alpha_Mean','Beta_Mean','Gamma_Mean','Delta_STD','Theta_STD','Alpha_STD','Beta_STD','Gamma_STD']
                self.column_names = ['Features','Label','Sensor1','Sensor2','Sensor3','Sensor4','Sensor5','Sensor6','Sensor7','Sensor8','Sensor9','Sensor10','Sensor11','Sensor12','Sensor13','Sensor14']
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
                alpha = map(lambda x: x[L*8/Fs-1: L*13/Fs],frequency) #from 8 instead of 5
                beta = map(lambda x: x[L*13/Fs-1: L*30/Fs],frequency)
                gamma = map(lambda x: x[L*30/Fs-1: L*50/Fs],frequency)

                avg_frequency = np.mean(frequency)
                
                return delta,theta,alpha,beta,gamma,avg_frequency


        def get_feature(self,all_channel_data): 
                #Get frequency data
                (delta,theta,alpha,beta,gamma,avg_frequency) = self.get_frequency(all_channel_data)

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
                
                return feature,avg_frequency
        
        def grid_searchcv(self,feature_with_labels,y):
                #print(y)
                ###For Equal Weiht


                df = pd.DataFrame(feature_with_labels)
                df = df.sort_values([140], ascending=[True])

                X = np.array(df.drop([140], 1),dtype=np.float32)
                y = np.array(df[140],dtype=np.uint8)

                #scaler = preprocessing.StandardScaler()
                #X = scaler.fit_transform(X)
                #print(X)
                
                X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.1)
                
                
                min_c = -5
                max_c = 15
                C_range = [2**i for i in range(min_c,max_c+1)]

                min_gamma = -10
                max_gamma = 5
                gamma_range = [2**i for i in range(min_gamma,max_gamma+1)]
                
                print("# Tuning hyper-parameters")

                param_grid = {'C' : C_range, 'gamma' : gamma_range}
                cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
                clf = GridSearchCV(SVC(kernel='rbf'), param_grid=param_grid, cv=cv)  #####see page 1195 for SVM rbf gridsearch cv from http://scikit-learn.org/dev/_downloads/scikit-learn-docs.pdf
                clf.fit(X_train, y_train)

                print('Best score for X:', clf.best_score_)
                print('Best C:',clf.best_estimator_.C) 
                #print('Best Kernel:',clf.best_estimator_.kernel)
                print('Best Gamma:',clf.best_estimator_.gamma)
                
                print('svm result ==> ',clf.score(X_test, y_test))

                '''
                print('#############################################')
                k_range = range(1,31)
                k_scores = []
                for k in k_range:
                        knn = neighbors.KNeighborsClassifier(n_neighbors = k, weights='uniform')
                        scores = cross_validation.cross_val_score(knn, X_train, y_train, cv = 10, scoring='accuracy')
                        k_scores.append(scores.mean())
                print(k_scores)

                plt.plot(k_range, k_scores)
                plt.xlabel('value of k')
                plt.ylabel('cross validated accuracy')
                plt.show()
                #trained_model = clf_knn.fit(X_train,y_train)
                #print('knn result ==> ',trained_model.score(X_test,y_test))
                '''
                joblib.dump(clf, os.path.join(self.profile_path,self.profile_name+'_model.pkl'))
                
        def main_process(self):
                self.original_df = pd.read_csv(os.path.join(self.profile_path,self.profile_name+'.csv'))

                Trials = len(self.original_df['Trial_No'])/1664
                
                selected = 0
                rejected = 0
                
                for trial in range(1,Trials+1):
                        
                        loc = self.original_df['Trial_No']==trial
                        trial_df = self.original_df[loc]
                        
                        rest_df = trial_df[:640]  #first 640
                        self.action_df = trial_df[-640:] #last 640

                        rest_all_channels_data = np.zeros((14,640),dtype=np.double)
                        action_all_channels_data = np.zeros((14,640),dtype=np.double)

                        for i in range(0,14):
                               
                            sensor_data = np.array((rest_df['Sensor'+str(i+1)]),dtype=np.double)
                            rest_all_channels_data[i] = sensor_data
                            
                            sensor_data = np.array((self.action_df['Sensor'+str(i+1)]),dtype=np.double)
                            action_all_channels_data[i] = sensor_data                            
                        
                        rest_features = np.zeros((1,140),dtype=np.double)
                        action_features = np.zeros((1,140),dtype=np.double)

                        rest_features[0],rest_avg_frequency     = self.get_feature(rest_all_channels_data)
                        action_features[0],action_avg_frequency = self.get_feature(action_all_channels_data)
                        
                        if action_avg_frequency < rest_avg_frequency:
                                print('Trial==>'+str(trial)+'  ##accepted' + 'with frequency '+str(action_avg_frequency) + 'and rest frequency is=>'+str(rest_avg_frequency))
                                if (self.action_df['Label'].values)[0]=='A':
                                        self.labels.append(1)
                                        self.features.append(np.insert(action_features[0],140,1))
                                        selected = selected +1
                                        
                                elif (self.action_df['Label'].values)[0]=='B':
                                        self.labels.append(2)
                                        self.features.append(np.insert(action_features[0],140,2))
                                        selected = selected + 1
                        else:
                                print('Trial==>'+str(trial)+'  ##rejected'+ 'with frequency '+str(action_avg_frequency) + 'and rest frequency is=>'+str(rest_avg_frequency))
                                rejected = rejected + 1
                                        
                print('selected = ',selected, ' and rejected =',rejected)                        
                Store_Features = pd.DataFrame(columns = self.column_names)
                
                lab = 0
                for f in self.features:
                        features_dict = {
                            'Features' : self.feature_names,
                            'Label'    : [self.labels[lab] for i in range(0,10)],
                            'Sensor1'  : f[0:10],
                            'Sensor2'  : f[10:20],
                            'Sensor3'  : f[20:30],
                            'Sensor4'  : f[30:40],
                            'Sensor5'  : f[40:50],
                            'Sensor6'  : f[50:60],
                            'Sensor7'  : f[60:70],
                            'Sensor8'  : f[70:80],
                            'Sensor9'  : f[80:90],
                            'Sensor10' : f[90:100],
                            'Sensor11' : f[100:110],
                            'Sensor12' : f[110:120],
                            'Sensor13' : f[120:130],
                            'Sensor14' : f[130:140],  
                                        }
                        df_features = pd.DataFrame(features_dict,columns = self.column_names)
                        Store_Features = Store_Features.append(df_features)
                        lab = lab+1

                Store_Features.to_csv(os.path.join(self.profile_path,self.profile_name+'_Features'+'.csv'))
                
                                
                self.grid_searchcv(np.array(self.features),np.array(self.labels))

#prd = ProcessData('C:\Users\maqsood ahmed\Desktop\project\Project_Source_Code\UsersData','username')
#prd.main_process()
