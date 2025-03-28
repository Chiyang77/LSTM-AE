
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import plotly.graph_objects as go
import pickle
import math
import seaborn as sns
Dataset='KKK13'
tstep='0001' #for KKK13, tstep=0.01, KKK14,15,16, tstep=0.008, KKK17, tstep=0.005

os.chdir('C:/Users/aljcl/Desktop/SDS/Final_Project/Data')


names=['energy','std','maxabs','skw','kurt','rms','clear','crest','shape','fe','p2p','spectrms','label'] #the columns names for LDV2
names_r=['energy_r','std_r','maxabs_r','skw_r','kurt_r','rms_r','clear_r','crest_r','shape_r','fe_r','p2p_r','spectrms_r','label']  #the columns names for LDV ratio


#concatnate the data from LDV2 with the LDVRatio
data_2= pd.read_excel('LDV_train_classification_signalfeatures_'+Dataset+'_tstep'+tstep+'.xls',header=None,names=names, sheet_name='ldv2')
data_2=data_2.drop(['label'],axis=1)
data_r= pd.read_excel('LDV_train_classification_signalfeatures_'+Dataset+'_tstep'+tstep+'.xls',header=None,names=names_r, sheet_name='ldv_r')
data=pd.concat([data_2,data_r],axis=1)
data_label=pd.DataFrame(data['label'])


#drop some columns that we are not gonna use (find out through correlation analysis)
data_2=data_2.drop(['clear','p2p','maxabs','crest','spectrms'],axis=1)
data_r=data_r.drop(['kurt_r','skw_r','clear_r','shape_r','energy_r','std_r','p2p_r','crest_r','fe_r','rms_r'],axis=1) #spectrms_r, rms_r
data=pd.concat([data_2,data_r],axis=1)

parameters_tune={'mov_win':[50,70,100,150,200,250,300,350],'TIME_STEPS':[10,20,30,40]}

#%%

latent_dim = 2

class AutoencoderModel:
    def __init__(self, X, latent_dim):
        self.inputs = Input(shape=(X.shape[1], X.shape[2]))
        self.encoded = LSTM(128, activation='relu', return_sequences=True)(self.inputs)
        self.encoded = Dropout(rate=0.2)(self.encoded)
        self.encoded = LSTM(64, activation='relu', return_sequences=True)(self.encoded)
        self.encoded = Dropout(rate=0.2)(self.encoded)
        self.encoded = LSTM(16, activation='relu', return_sequences=True)(self.encoded)
        self.encoded = Dropout(rate=0.2)(self.encoded)
        self.encoded = LSTM(latent_dim)(self.encoded)
        self.decoded = RepeatVector(X.shape[1])(self.encoded)
        self.decoded = LSTM(16, activation='relu', return_sequences=True)(self.decoded)
        self.decoded = Dropout(rate=0.2)(self.decoded)
        self.decoded = LSTM(64, activation='relu', return_sequences=True)(self.decoded)
        self.decoded = Dropout(rate=0.2)(self.decoded)
        self.decoded = LSTM(128, activation='relu', return_sequences=True)(self.decoded)
        self.output = TimeDistributed(Dense(X.shape[2]))(self.decoded)
        self.model = Model(inputs=self.inputs, outputs=self.output)
        self.encoder = Model(self.inputs, self.encoded)

    def get_model(self):
        return self.model

    def get_encoder(self):
        return self.encoder

TIME_STEPS=20

class SequenceGenerator:
    def __init__(self, time_steps=TIME_STEPS):
        self.time_steps = time_steps
    
    def create_sequences(self, X, y):
        Xs, ys = [], []
        for i in range(len(X)-self.time_steps):
            Xs.append(X.iloc[i:(i+self.time_steps)].values)
            ys.append(y.iloc[i+self.time_steps])
        
        return np.array(Xs), np.array(ys)
    

#%%
#------------------------------------------------------------------
#|Do Log transform for skew data, make it more normal distributed  |
#-------------------------------------------------------------------
float_columns=[x for x in data.columns != 'label']
float_columns=data.columns[float_columns]
skew_columns=(data[float_columns].skew().sort_values(ascending=False))
skew_columns = skew_columns.loc[skew_columns>0.75]

for col in skew_columns.index.tolist():
    data[col]=np.log1p(data[col])

data_temp=data

data_label_temp=data_label
velocity_ldv2=pd.read_excel('LDV2_downsample_'+Dataset+'_tstep'+tstep+'.xls',header=None)
#---------------------
#|Do moving average  |
#---------------------

#%%
for win in parameters_tune['mov_win']:
    
    for tsteps in parameters_tune['TIME_STEPS']:
        data=data_temp.copy()

        data_label=data_label_temp
        for col in float_columns:
            data[col]=data[col].rolling(window=win,center=False,min_periods=1).mean() #350 for KKK13 70 for 14 15 16 , 50for 17    

    
        X_train1, X_test1, y_train1, y_test1 = train_test_split(data[float_columns], data['label'], test_size = 0.3, shuffle = False) #train test split
        
    
    
    #--------------------------------------------------------------------------
    #|Standardize the data, and save the prarameters used for standardization  |
    #---------------------------------------------------------------------------
    
    

        ss = StandardScaler()
        ssmean=pd.DataFrame()
        ssscale= pd.DataFrame()
        ssmean.rename(columns=data.columns)
        ssscale.rename(columns=data.columns)
        
        for col in float_columns:
            X_train1[col] = ss.fit_transform(X_train1[[col]]).squeeze()
            ssmean[col]=ss.mean_
            ssscale[col]=ss.scale_
            X_test1[col] = ss.transform(X_test1[[col]]).squeeze()
            data[col]=ss.transform(data[[col]]).squeeze()
        
        
        
        #drop some columns that we are not gonna use (find out through PCA)
        pca_columns=pd.Series(['std','fe','energy','rms','kurt']) #for shape for 14,15,16
        X_train1=X_train1[pca_columns]
        X_test1=X_test1[pca_columns]
        data=data[pca_columns]
    
    
    
        print('2')
    
    
    
    #%
        X_train, y_train=[],[]
        X_test, y_test=[],[]
        
        seq_gen = SequenceGenerator(time_steps=tsteps)
        
        X_train, y_train = seq_gen.create_sequences(X_train1, y_train1)
        print('2.5')
        X_test, y_test = seq_gen.create_sequences(X_test1, y_test1)
        print('3')
        data_feature, data_label =seq_gen.create_sequences(data,data_label)
        print('3.5')
    
        autoencoder = AutoencoderModel(X_train, latent_dim)
        model = autoencoder.get_model()
        encoder = autoencoder.get_encoder()
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mae")
    
        transfer_es = EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
        history=model.fit(X_train, X_train, epochs=50, batch_size=72, validation_split=0.1, callbacks= transfer_es)
        
     
    
    
    
        hfont = {'fontname':'Times New Roman'}
        fig2, ax2 = plt.subplots()
        plt.grid(visible=None)
        plt.ylim(ymin=0,ymax=round(max(max(history.history["val_loss"]),max(history.history["loss"])),1)*1.2)
        plt.xlim(0,50)
        ax2.plot(history.history["val_loss"], label="Validation Loss",color='k',linestyle='-')
        ax2.plot(history.history["loss"], label="Training Loss",color='0.45',linestyle='--')
        plt.xlabel("Number of Epoch",fontsize=24) #plt.xlabel("Number of Epoch", **hfont,fontsize=24)
        plt.ylabel("Loss",fontsize=24)
        plt.yticks(np.linspace(0,round(max(max(history.history["val_loss"]),max(history.history["loss"])),1)*1.2,num=5),fontsize=18)
        plt.xticks(fontsize=18)
        plt.legend(fontsize=16)
        plt.title(str(win)+'_'+str(tsteps))
        plt.show()
    
#        save_path = './gridsearch/SDS_'+Dataset+'_'+str(win)+'.h5'
#        model.save(save_path)
    
    
        x_train=X_train
        x_test= X_test
        
        # Get train MAE loss.
        x_train_pred = model.predict(x_train)
        train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)
        
        
        train_mae_loss=np.sum(train_mae_loss, axis=1)
        train_mae_loss = train_mae_loss.reshape((-1))
        train_mae_loss = train_mae_loss[50:-1,]
        # Get reconstruction loss threshold.
        
        
        threshold = np.max(train_mae_loss)*1+3*np.std(train_mae_loss) 
        
        training_mean = x_train.mean()
        training_std = x_train.std()
        df_test_value = (X_test - training_mean) / training_std
        
        # Get test MAE loss.
        x_test_pred = model.predict(x_test)
        test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)
        
        
        test_mae_loss=np.sum(test_mae_loss, axis=1)
        test_mae_loss = test_mae_loss.reshape((-1))
        
        
        
        # Detect all the samples which are anomalies.
        anomalies = test_mae_loss > threshold
        
        #%
        # data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
        anomalous_data_indices = []
        
        for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
            if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
                anomalous_data_indices.append(data_idx)
        
        
        predicted_labels = (test_mae_loss > threshold).astype(int)
        
        #%
        ldv2_acc_org= pd.read_csv('LDV2_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
        fs=1250000
        time=np.linspace(0, len(ldv2_acc_org)/fs, num=len(ldv2_acc_org))
        mae_loss=np.concatenate([train_mae_loss,test_mae_loss],axis=0)
        time_mae=np.linspace(0,time[-1],num=len(mae_loss))
        
        fig1, ax1 = plt.subplots()
        ax1.plot(time_mae,mae_loss,c='k')
        plt.xlabel("Time(s)",fontsize=16,fontname='Times New Roman')
        plt.ylabel("MAE Loss",fontsize=16,fontname='Times New Roman')
        plt.xlim(xmin=min(time), xmax=max(time))
        plt.title(str(win)+'_'+str(tsteps))
        fig1.set_figwidth(12)
        fig1.set_figheight(4)
        ax1.axhline(threshold, xmin=0.0, xmax=1.0, color='k',linestyle='--',linewidth=3)
        timearray=np.append(np.arange(0, time[-1], step=0.5),round(time[-1],1))
        plt.xticks(timearray,fontsize=12) 
        
        
    
        
        indx=anomalous_data_indices+np.ones(len(anomalous_data_indices))*y_train.shape[0]
        print('4')
        fig3, ax3 = plt.subplots()
        ax3.plot(velocity_ldv2)
        
        ax3.plot(velocity_ldv2.iloc[indx],color='r')
        plt.xlabel("data points")
        plt.ylabel("velocity(m/s)")
    



#%%
  
# read text file into pandas DataFrame
ldv2_acc_org= pd.read_csv('LDV2_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
ldv1_acc_org= pd.read_csv('LDV1_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")

fs=1250000
# display DataFrame
print(ldv2_acc_org)


def separate_array(arr):
  sublists = []
  sublist = [arr[0]]
  for i in range(1, len(arr)):
    if arr[i] - arr[i-1] == 1:
      sublist.append(arr[i])
    else:
      sublists.append(sublist)
      sublist = [arr[i]]
  sublists.append(sublist)
  return sublists


indx_sep=separate_array(indx)
anom_start_end=[]
anom_org=[]
for i in range(0,len(indx_sep)):
    indx_org_s=math.floor((indx_sep[i][0]/len(velocity_ldv2))*len(ldv2_acc_org))
    indx_org_e=math.floor((indx_sep[i][-1]/len(velocity_ldv2))*len(ldv2_acc_org))
    anom_start_end.append([indx_org_s,indx_org_e])
    anom_org.append(np.arange(indx_org_s,indx_org_e+1))

# indx_org_s=math.floor((indx[0]/len(velocity_ldv2))*len(ldv2_acc_org))
# indx_org_e=math.floor((indx[-1]/len(velocity_ldv2))*len(ldv2_acc_org))
# indx_org=np.arange(indx_org_s,indx_org_e+1)



hfont = {'fontname':'Times New Roman'}
time=np.linspace(0, len(ldv2_acc_org)/fs, num=len(ldv2_acc_org))
fig2, ax2 = plt.subplots()
ax2.plot(time,ldv2_acc_org,color='k',linewidth=0.5)
ax2.plot(time,ldv1_acc_org,color='k',linestyle='--',linewidth=0.5)
for i in range(0,len(indx_sep)):
    ax2.plot(time[anom_org[i]],ldv2_acc_org.iloc[anom_org[i]],c='0.45',linewidth=0.5) #,marker="o",markersize=3,markerfacecolor='none'
    ax2.plot(time[anom_org[i]],ldv1_acc_org.iloc[anom_org[i]],c='0.45',linestyle='--',linewidth=0.5) #,marker="o",markersize=3,markerfacecolor='none'

#ax2.plot(time[indx_org],ldv2_acc_org.iloc[indx_org],c='0.45',linewidth=0.1,marker="o",markersize=3,markerfacecolor='none')

plt.xlabel("Time(s)", **hfont,fontsize=16)
plt.ylabel("Velocity(m/s)", **hfont,fontsize=16)
plt.xlim(xmin=min(time), xmax=max(time))
timearray=np.append(np.arange(0, time[-1], step=0.5),round(time[-1],1))
plt.xticks(timearray,fontsize=12) 
#plt.grid(visible=None)
fig2.set_figwidth(12)
fig2.set_figheight(4)

ldv2_acc_org = ldv2_acc_org.to_numpy()
lim_array=max(ldv2_acc_org)*1.5
lim=round(lim_array[0],2)
#lim=0.06
ax2.set_ylim([-1*lim,lim])
ax2.set_yticks(np.linspace(-1*lim,lim,num=5))
plt.yticks(fontsize=12)
ax2.grid(False)
ax2.legend(["LDV2","LDV1","LDV2_anomaly","LDV1_anomaly"],loc='upper right', fancybox=True, framealpha=0.1,ncol=4)

#%%
#plot the time series features
#names=['energy','std','maxabs','kurt','rms','fe','p2p','spectrms','ssd','label']

feature="fe"
#data=pd.concat([data_train,data_test],axis=0,ignore_index=True)
time_d=np.linspace(0, len(ldv2_acc_org)/fs,num= len(data))
plt.plot(time_d,data[feature],'k')
plt.xlabel("Time(s)", **hfont)
plt.xlim(xmin=min(time), xmax=max(time))

lim_feat=round(abs(max(data[feature])),1)
ax2.set_yticks(np.linspace(-1*lim_feat,lim_feat,num=3))
plt.ylabel("Normalized Entropy", **hfont)
#ax2.legend()
#%%

#convert the predicted features back to the same dimension of the orginal time series

def extract_original_data(Xs, time_steps):
    X_original = pd.DataFrame(columns=[f'A{i}' for i in range(Xs.shape[2])])
    for i in range(0, Xs.shape[0], time_steps):
        X_original = X_original.append(pd.DataFrame(Xs[i], columns=[f'A{i}' for i in range(Xs.shape[2])]))
    X_original.reset_index(drop=True, inplace=True)
    return X_original

x_test_pred_orginalsize=[]
x_train_pred_orginalsize=[]
x_test_pred_orginalsize=extract_original_data(X_test,TIME_STEPS)
x_train_pred_orginalsize=extract_original_data(X_train,TIME_STEPS)
test_sizediff=len(x_test)-len(x_test_pred_orginalsize)
train_sizediff=len(x_train)-len(x_train_pred_orginalsize)
x_test_pred_orginalsize=pd.concat([x_test_pred_orginalsize,x_test_pred_orginalsize.tail(test_sizediff)],axis=0,ignore_index=True)
x_train_pred_orginalsize=pd.concat([x_train_pred_orginalsize,x_train_pred_orginalsize.tail(train_sizediff)],axis=0,ignore_index=True)
#%%
plot_latent='True'
latent_option='y_test'
#either anomalies or y_test or ( data_feature or data_label)

if latent_option == 'data_label':
    latent_c=data_label
    latent_c=data_label.reshape(-1)
    encoderinput=data_feature
else:
    latent_c=y_test
    encoderinput=x_test
            
def plot_lat_predict(latent_dim,latent_c,encoderinput):

            
    latent_representation = encoder.predict(encoderinput)
    latent_representation = np.array(latent_representation)
    colors = {True:'grey',False:'black'} 
    latent_c_color=np.vectorize(colors.get)(latent_c)

    
    if latent_dim == 2:
        plt.grid(b=None)
        ax = plt.axes
        plt.scatter(latent_representation[:, 0], latent_representation[:, 1],c=latent_c_color,alpha=0.8)
#        plt.xticks([-min(latent_representation[:, 0]), 0, max(latent_representation[:, 0])],fontsize=16)
 #       plt.yticks([-min(latent_representation[:, 1]), 0, max(latent_representation[:, 1])],fontsize=16)
        
        plt.locator_params(axis='both', nbins=4)
        plt.xticks(fontsize=16)
        plt.yticks(fontsize=16)
#        plt.colorbar()
        plt.show()
        
    else:
        
        ax = plt.axes(projection ="3d")
        plt.grid(b=None)
        ax.scatter3D(latent_representation[:, 0], latent_representation[:, 1], latent_representation[:, 2], c=latent_c_color,alpha=0.8)
        
        ax.view_init(0, 0)
        ax.xaxis._axinfo["grid"].update({"linewidth":0.2})
        ax.yaxis._axinfo["grid"].update({"linewidth":0.2})
        ax.zaxis._axinfo["grid"].update({"linewidth":0.2})
        ax.set_xticks([-max(abs(latent_representation[:, 0])), 0, max(abs(latent_representation[:, 0]))])
        ax.set_yticks([-max(abs(latent_representation[:, 1])), 0, max(abs(latent_representation[:, 1]))])
        ax.set_zticks([-max(abs(latent_representation[:, 2])), 0, max(abs(latent_representation[:, 2]))])
        ax.zaxis.set_tick_params(labelsize=8)
        
#        plt.colorbar()
        

if plot_latent=='True':        
    plot_lat_predict(latent_dim,latent_c,encoderinput)
    
