
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import os 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report

import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, Dense, LSTM, Dropout, RepeatVector, TimeDistributed
import plotly.graph_objects as go
import pickle
import math
#%%
Dataset='KKK13'
tstep='0001'

os.chdir('C:/Users/aljcl/Desktop/SDS/Final_Project/Data')
names=['energy','std','maxabs','skw','kurt','rms','clear','crest','shape','fe','p2p','spectrms','label']
names_r=['energy_r','std_r','maxabs_r','skw_r','kurt_r','rms_r','clear_r','crest_r','shape_r','fe_r','p2p_r','spectrms_r','label']



data_2= pd.read_excel('LDV_train_classification_signalfeatures_'+Dataset+'_tstep'+tstep+'.xls',header=None,names=names, sheet_name='ldv2')

data_2=data_2.drop(['skw','clear','crest','shape','label'],axis=1)
data_r= pd.read_excel('LDV_train_classification_signalfeatures_'+Dataset+'_tstep'+tstep+'.xls',header=None,names=names_r, sheet_name='ldv_r')
data_r=data_r.drop(['skw_r','clear_r','crest_r','shape_r'],axis=1)
data=pd.concat([data_2,data_r],axis=1)
data_label=pd.DataFrame(data['label'])
#data=data_2


#data=data.drop(['spectrms_r','p2p_r','maxabs_r','std_r','energy_r','spectrms','p2p','fe','rms'],axis=1)

float_columns=[x for x in data.columns != 'label']
float_columns=data.columns[float_columns]

#%%
skew_columns=(data[float_columns].skew().sort_values(ascending=False))
skew_columns = skew_columns.loc[skew_columns>0.75]
skew_columns

for col in skew_columns.index.tolist():
    data[col]=np.log1p(data[col])

#data.head(4)

for col in float_columns:
    data[col]=data[col].rolling(window=300,center=False,min_periods=1).mean() #300 for KKK13 %70 for others
#%%
X_train1, X_test1, y_train1, y_test1 = train_test_split(data[float_columns], data['label'], test_size = 0.3, shuffle = False)



from sklearn.preprocessing import MinMaxScaler, StandardScaler

mms = MinMaxScaler()
ss = StandardScaler()

for col in float_columns:
    X_train1[col] = ss.fit_transform(X_train1[[col]]).squeeze()
    X_test1[col] = ss.transform(X_test1[[col]]).squeeze()


#%%
names=['energy','std','maxabs','kurt','rms','fe','p2p','spectrms']
#data=pd.concat([data_train,data_test],axis=0,ignore_index=True)
plt.plot(X_test1['p2p'])

#%%
from pca import pca
pca_model = pca()
pca_model = pca(n_components=0.98) # n_components larger than 1, a feature reduction will be performed to exactly that number of components. By setting n_components smaller than 1, it describes the percentage of explained variance that needs to be covered at least. Or in other words, by setting n_components=0.95
out = pca_model.fit_transform(X_train1)

# Print the top features. The results show that f1 is best, followed by f2 etc
print(out['topfeat'])
pca_model.plot()
pca_model.plot(visible=True)
#ax = pca_model.biplot(n_feat=10, legend=False)
#ax = pca_model.biplot3d(n_feat=10, legend=False,label=None)
#%%

#pca_columns=out['topfeat']['feature'][:-9]

#data=data[pca_columns]
#pca_columns=pd.Series(['spectrms','kurt','rms_r','rms','fe','energy'])#for KKK17
#pca_columns=pd.Series(['spectrms','rms','kurt','fe','std','energy']) #for KKK15
#pca_columns=pd.Series(['spectrms','rms','kurt','fe','maxabs_r','energy']) #for KKK14
#pca_columns=pd.Series(['kurt','fe_r','maxabs_r','energy','kurt_r','std']) #for KKK16
pca_columns=pd.Series(['energy','fe_r','maxabs','kurt_r','rms_r','fe'])# for KKK13
X_train1=X_train1[pca_columns]
X_test1=X_test1[pca_columns]
data=data[pca_columns]
#%%
TIME_STEPS=20

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

X_train, y_train=[],[]
X_test, y_test=[],[]

X_train, y_train = create_sequences(X_train1, y_train1)
X_test, y_test = create_sequences(X_test1, y_test1)
data_feature, data_label =create_sequences(data,data_label)


print(f'Training shape: {X_train.shape}')
print(f'Testing shape: {X_test.shape}')


#%%

latent_dim = 2
plot_latent='True'

def autoencoder_model(X,latent_dim):
    
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    encoded = LSTM(128,activation='elu',return_sequences=True)(inputs)
    encoded=Dropout(rate=0.2)(encoded)
    encoded = LSTM(64,activation='elu',return_sequences=True)(encoded)
    encoded=Dropout(rate=0.2)(encoded)
    encoded = LSTM(16,activation='elu',return_sequences=True)(encoded)
    encoded=Dropout(rate=0.2)(encoded)
    encoded = LSTM(latent_dim)(encoded)
    decoded = RepeatVector(X.shape[1])(encoded)
    decoded = LSTM(16, activation='relu',return_sequences=True)(decoded)
    decoded=Dropout(rate=0.2)(decoded)
    decoded = LSTM(64, activation='relu',return_sequences=True)(decoded)
    decoded=Dropout(rate=0.2)(decoded)
    decoded = LSTM(128, activation='relu',return_sequences=True)(decoded)
#    decoded=Dropout(rate=0.2)(decoded)
    #decoded = LSTM(input_dim, return_sequences=True)(decoded)
    output=TimeDistributed(Dense(X.shape[2]))(decoded)
    model=Model(inputs=inputs, outputs=output)
    encoder = Model(inputs, encoded)
    return model, encoder

model,encoder = autoencoder_model(X_train,latent_dim)
#%%
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mae")
model.summary()
history = model.fit(X_train, X_train, epochs=50, batch_size=72, validation_split=0.1)



#%%
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d


def plot_lat(latent_dim):
    latent_representation = encoder.predict(X_test)
    latent_representation = np.array(latent_representation)
    if latent_dim == 2:
        plt.scatter(latent_representation[:, 0], latent_representation[:, 1],c=y_test)
        plt.colorbar()
        plt.show()
    else:
        ax = plt.axes(projection ="3d")
        ax.scatter3D(latent_representation[:, 0], latent_representation[:, 1], latent_representation[:, 2], c=y_test)
        ax.view_init(100, 70)

if plot_latent=='True':        
    plot_lat(latent_dim)

#%%
hfont = {'fontname':'Times New Roman'}
plt.grid(visible=None)
plt.ylim(ymin=0,ymax=round(max(max(history.history["val_loss"]),max(history.history["loss"])),1)*1.2)
plt.xlim(0,50)
plt.plot(history.history["val_loss"], label="Validation Loss",color='k',linestyle='-')
plt.plot(history.history["loss"], label="Training Loss",color='0.45',linestyle='--')
plt.xlabel("Number of Epoch",fontsize=24) #plt.xlabel("Number of Epoch", **hfont,fontsize=24)
plt.ylabel("Loss",fontsize=24)
plt.yticks(np.linspace(0,round(max(max(history.history["val_loss"]),max(history.history["loss"])),1)*1.2,num=5),fontsize=18)
plt.xticks(fontsize=18)
plt.legend(fontsize=16)
plt.show()


#%%
save_path = './SDS_KKK13.h5'
model.save(save_path)
#%%
save_path = './SDS_KKK13.h5'
model = keras.models.load_model(save_path)

#%%
x_train=X_train
x_test= X_test

# Get train MAE loss.
x_train_pred = model.predict(x_train)
train_mae_loss = np.mean(np.abs(x_train_pred - x_train), axis=1)

plt.hist(train_mae_loss, bins=50)
plt.xlabel("Train MAE loss")
plt.ylabel("No. of samples")
plt.show()

# Get reconstruction loss threshold.
#threshold = np.max(train_mae_loss[:,0]+train_mae_loss[:,1]+train_mae_loss[:,2]+train_mae_loss[:,3]+train_mae_loss[:,4])
#threshold = np.max(train_mae_loss)+30*np.std(np.abs(x_train_pred - x_train))
threshold = np.max(train_mae_loss)*2 #1.5 for KKK13

#threshold = np.mean(np.abs(x_train_pred - x_train)) + 30*np.std(np.abs(x_train_pred - x_train))
print("Reconstruction error threshold: ", threshold)


training_mean = x_train.mean()
training_std = x_train.std()

df_test_value = (X_test - training_mean) / training_std


print("Test input shape: ", x_test.shape)

# Get test MAE loss.
x_test_pred = model.predict(x_test)
test_mae_loss = np.mean(np.abs(x_test_pred - x_test), axis=1)

#test_mae_loss= test_mae_loss[:,1]+test_mae_loss[:,0]+test_mae_loss[:,2]+test_mae_loss[:,3]++test_mae_loss[:,4]
test_mae_loss=np.sum(test_mae_loss, axis=1)
test_mae_loss = test_mae_loss.reshape((-1))

plt.hist(test_mae_loss, bins=50)
plt.xlabel("test MAE loss")
plt.ylabel("No of samples")
plt.show()

# Detect all the samples which are anomalies.
anomalies = test_mae_loss > threshold
print("Number of anomaly samples: ", np.sum(anomalies))
print("Indices of anomaly samples: ", np.where(anomalies))

# data i is an anomaly if samples [(i - timesteps + 1) to (i)] are anomalies
anomalous_data_indices = []

for data_idx in range(TIME_STEPS - 1, len(df_test_value) - TIME_STEPS + 1):
    if np.all(anomalies[data_idx - TIME_STEPS + 1 : data_idx]):
        anomalous_data_indices.append(data_idx)
#%%
#df_subset=np.zeros(y_test.shape)
#df_subset[anomalous_data_indices,0] = y_test[anomalous_data_indices]
#
#fig, ax = plt.subplots()
#ax.plot(data_test['energy1'])
#ax.plot(data_test.iloc[anomalous_data_indices]['energy1'],color='r')


velocity_ldv2=pd.read_excel('LDV2_downsample_'+Dataset+'_tstep'+tstep+'.xls',header=None)
#velocity_ldv2=pd.read_excel('LDV2_downsample_KKK14_tstep00008.xls',header=None)


indx=anomalous_data_indices+np.ones(len(anomalous_data_indices))*y_train.shape[0]

fig1, ax1 = plt.subplots()
ax1.plot(velocity_ldv2)
ax1.plot(velocity_ldv2.iloc[indx],color='r')
plt.xlabel("data points")
plt.ylabel("velocity(m/s)")


#fig1, ax1 = plt.subplots()
#ax1.plot(velocity_ldv2)
#plt.xlabel("data points")
#plt.ylabel("velocity(m/s)")

#%%
  
# read text file into pandas DataFrame
ldv2_acc_org= pd.read_csv('LDV2_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
ldv1_acc_org= pd.read_csv('LDV1_vel_original_'+Dataset+'_tstep'+tstep+'.txt', sep=" ")
#ldv2_acc_org= pd.read_csv("LDV2_acc_original_KKK17_tstep00005.txt", sep=" ")
#plt.plot(ldv2_acc_org)

fs=1250000
# display DataFrame
print(ldv2_acc_org)

#indx=[1,2,3,4,5,10,11,12,13,16,17,18,19]
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
    ax2.plot(time[anom_org[i]],ldv2_acc_org.iloc[anom_org[i]],c='0.45',linewidth=0.5,marker="o",markersize=3,markerfacecolor='none')
    ax2.plot(time[anom_org[i]],ldv1_acc_org.iloc[anom_org[i]],c='0.45',linestyle='--',linewidth=0.5,marker="o",markersize=3,markerfacecolor='none')

#ax2.plot(time[indx_org],ldv2_acc_org.iloc[indx_org],c='0.45',linewidth=0.1,marker="o",markersize=3,markerfacecolor='none')

plt.xlabel("Time(s)", **hfont,fontsize=16)
plt.ylabel("Velocity(m/s)", **hfont,fontsize=16)
plt.xlim(xmin=min(time), xmax=max(time))
plt.xticks(np.arange(0, time[-1], step=0.5),fontsize=12) 

fig2.set_figwidth(12)
fig2.set_figheight(4)

ldv2_acc_org = ldv2_acc_org.to_numpy()
lim_array=max(ldv2_acc_org)*1.5
lim=round(lim_array[0],2)
#lim=0.06
ax2.set_ylim([-1*lim,lim])
ax2.set_yticks(np.linspace(-1*lim,lim,num=5))
plt.yticks(fontsize=12)
plt.grid(visible=None)
ax2.legend(["LDV2","LDV1","LDV2_anomaly","LDV1_anomaly"],loc='upper right', fancybox=True, framealpha=0.1,ncol=4)

#%%
#plot the time series features
#names=['energy','std','maxabs','kurt','rms','fe','p2p','spectrms','ssd','label']

feature="energy"
#data=pd.concat([data_train,data_test],axis=0,ignore_index=True)
time_d=np.linspace(0, len(ldv2_acc_org)/fs,num= len(data))
plt.plot(time_d,data[feature],'k')
plt.xlabel("Time(s)", **hfont)
plt.xlim(xmin=min(time), xmax=max(time))
plt.ylim(ymin=-abs(max(data[feature]))*1.2, ymax=abs(max(data[feature]))*1.2)
lim_feat=round(abs(max(data[feature])),1)
ax2.set_yticks(np.linspace(-1*lim_feat,lim_feat,num=3))
plt.ylabel("Normalized Entropy", **hfont)
ax2.legend()
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


latent_option='data_label'
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
#%%

#
#feature A5 for entropy
feature_d='A1'
data_predict=pd.concat([x_train_pred_orginalsize,x_test_pred_orginalsize],axis=0,ignore_index=True)
time_d=pd.DataFrame(np.linspace(0, len(ldv2_acc_org)/fs,num= len(data_predict)))
plt.plot(time_d,data_predict[feature_d],'k')
#plt.plot(time_d.iloc[indx],data_predict[feature_d].iloc[indx],color='r')
plt.scatter(time_d.iloc[indx],data_predict[feature_d].iloc[indx],color='r')
plt.xlabel("Time(s)", **hfont)
plt.ylabel("Normalized Entropy", **hfont)

plt.xlim(xmin=min(time_d), xmax=max(time))
plt.ylim(ymin=-abs(max(data_predict[feature_d]))*1.1, ymax=abs(max(data_predict[feature_d]))*1.1)
lim_feat_d=round(abs(max(data_predict[feature_d])),1)
plt.yticks(np.linspace(-1*lim_feat_d,lim_feat_d,num=5))

#%%
time_dd=time_d.reshape(-1,1)
time_dd=time_dd[:-10]
#%%
plt.plot(time_dd,data_label)
plt.xlim(xmin=min(time_dd), xmax=max(time_dd))
plt.grid()
#%%
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
anomalies_01=anomalies*1
fpr, tpr, _ = metrics.roc_curve(y_test,  anomalies_01)
#%%

auc = metrics.roc_auc_score(y_test, anomalies_01)

#create ROC curve
plt.plot(fpr,tpr,label="AUC="+str(auc))
print(f'model 1 AUC score: {roc_auc_score(y_test, anomalies_01)}')
#%%
#Other possible models
#%%
#model 3
def autoencoder_model(X):
    inputs=Input(shape=(X.shape[1], X.shape[2]))
    L1=LSTM(128,activation='relu', return_sequences=True)(inputs)
    L2=Dropout(rate=0.2)(L1)
    L3=LSTM(64,activation='relu',return_sequences=True)(L2)
    L4=Dropout(rate=0.2)(L3)   
    L5=LSTM(32,activation='relu',return_sequences=False)(L4)
    L6=Dropout(rate=0.2)(L5) 
    L7=RepeatVector(X.shape[1])(L6)
    L8=LSTM(32,activation='relu',return_sequences=True)(L7)
    L9=Dropout(rate=0.2)(L8)          
    L10=LSTM(64,activation='relu',return_sequences=True)(L9)
    L11=Dropout(rate=0.2)(L10)      
    L12=LSTM(128,activation='relu',return_sequences=True)(L11)
    L13=Dropout(rate=0.2)(L12)        
    output=TimeDistributed(Dense(X.shape[2]))(L13)
    model=Model(inputs=inputs, outputs=output)
    return model
#%%

def autoencoder_model(X):
    inputs=Input(shape=(X.shape[1], X.shape[2]))
    L1=LSTM(16,activation='relu', return_sequences=True)(inputs)
    L2=Dropout(rate=0.2)(L1)
    L3=LSTM(4,activation='relu',return_sequences=False)(L2)
    L4=Dropout(rate=0.2)(L3)   
    L5=RepeatVector(X.shape[1])(L4)
    L6=LSTM(4,activation='relu',return_sequences=True)(L5)
    L7=Dropout(rate=0.2)(L6)      
    L8=LSTM(16,activation='relu',return_sequences=True)(L7)
    L9=Dropout(rate=0.2)(L8)        
    output=TimeDistributed(Dense(X.shape[2]))(L9)
    model=Model(inputs=inputs, outputs=output)
    return model

#%%
latent_dim = 2
def autoencoder_model(X,latent_dim):
    
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    encoded = LSTM(128,activation='relu',return_sequences=True)(inputs)
    encoded = Dropout(rate=0.2)(encoded)    
    encoded = LSTM(64,activation='relu',return_sequences=True)(encoded)
    encoded = LSTM(latent_dim)(encoded)
    decoded = RepeatVector(X.shape[1])(encoded)
    decoded = LSTM(64, activation='relu',return_sequences=True)(decoded)
    decoded = LSTM(128, activation='relu',return_sequences=True)(decoded)
    #decoded = LSTM(input_dim, return_sequences=True)(decoded)
    output=TimeDistributed(Dense(X.shape[2]))(decoded)
    model=Model(inputs=inputs, outputs=output)
    encoder = Model(inputs, encoded)
    return model, encoder

model,encoder = autoencoder_model(X_train,latent_dim)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()
history = model.fit(X_train, X_train, epochs=100, batch_size=72, validation_split=0.1)


#%%
model = autoencoder_model(X_train)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse")
model.summary()
history = model.fit(X_train, X_train, epochs=80, batch_size=72, validation_split=0.1)