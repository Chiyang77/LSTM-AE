
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

Dataset='KKK13'
np.random.seed(1)
tf.random.set_seed(1)

os.chdir('C:/Users/aljcl/Desktop/SDS/Final_Project/Data')
names=['energy','std','maxabs','skw','kurt','rms','clear','crest','shape','fe','p2p','spectrms','ssd','label']
data_train= pd.read_excel('LDV_label_train_signalfeatures_'+Dataset+'_tstep0001.xls',header=None,names=names)
data_test= pd.read_excel('LDV_label_test_signalfeatures_'+Dataset+'_tstep0001.xls',header=None,names=names)


#need to do this for KKK15
data_train=data_train.drop(['skw','clear','crest','shape','ssd'],axis=1)
data_test=data_test.drop(['skw','clear','crest','shape','ssd'],axis=1)


#%
float_columns=[x for x in data_train.columns != 'label']
float_columns=data_train.columns[float_columns]


skew_columns=(data_train[float_columns].skew().sort_values(ascending=False))
skew_columns = skew_columns.loc[skew_columns>0.75]
skew_columns

for col in skew_columns.index.tolist():
    data_train[col]=np.log1p(data_train[col])
    data_test[col]=np.log1p(data_test[col])

data_train.head(4)


from sklearn.preprocessing import MinMaxScaler, StandardScaler

mms = MinMaxScaler()
ss = StandardScaler()
#%%
for col in float_columns:
    data_train[col] = ss.fit_transform(data_train[[col]]).squeeze()
    data_test[col] = ss.transform(data_test[[col]]).squeeze()  
    data_train[col]=data_train[col].rolling(window=50,center=False,min_periods=1).mean()
    data_test[col]=data_test[col].rolling(window=50,center=False,min_periods=1).mean()


data=pd.concat([data_train,data_test],axis=0,ignore_index=True)
plt.plot(data['energy'])
#%%
TIME_STEPS=10

def create_sequences(X, y, time_steps=TIME_STEPS):
    Xs, ys = [], []
    for i in range(len(X)-time_steps):
        Xs.append(X.iloc[i:(i+time_steps)].values)
        ys.append(y.iloc[i+time_steps])
    
    return np.array(Xs), np.array(ys)

X_train, y_train=[],[]
X_test, y_test=[],[]

X_train, y_train = create_sequences(data_train[float_columns], data_train['label'])
X_test, y_test = create_sequences(data_test[float_columns], data_test['label'])
data_feature, data_label =create_sequences(data[float_columns],data['label'])


print(f'Training shape: {X_train.shape}')
print(f'Testing shape: {X_test.shape}')


#%%

latent_dim = 3
plot_latent='True'

def autoencoder_model(X,latent_dim):
    
    inputs = Input(shape=(X.shape[1], X.shape[2]))
    encoded = LSTM(128,activation='relu',return_sequences=True)(inputs)
    encoded=Dropout(rate=0.2)(encoded)
    encoded = LSTM(64,activation='relu',return_sequences=True)(encoded)
    encoded=Dropout(rate=0.2)(encoded)
    encoded = LSTM(16,activation='relu',return_sequences=True)(encoded)
    encoded=Dropout(rate=0.2)(encoded)
    encoded = LSTM(latent_dim)(encoded)
    decoded = RepeatVector(X.shape[1])(encoded)
    decoded = LSTM(16, activation='relu',return_sequences=True)(decoded)
    decoded=Dropout(rate=0.2)(decoded)
    decoded = LSTM(64, activation='relu',return_sequences=True)(decoded)
    decoded=Dropout(rate=0.2)(decoded)
    decoded = LSTM(128, activation='relu',return_sequences=True)(decoded)
    decoded=Dropout(rate=0.2)(decoded)
    #decoded = LSTM(input_dim, return_sequences=True)(decoded)
    output=TimeDistributed(Dense(X.shape[2]))(decoded)
    model=Model(inputs=inputs, outputs=output)
    encoder = Model(inputs, encoded)
    return model, encoder

model,encoder = autoencoder_model(X_train,latent_dim)

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
plt.ylim(ymin=0,ymax=round(max(history.history["val_loss"]),1))
plt.xlim(0,50)
plt.plot(history.history["val_loss"], label="Validation Loss",color='k',linestyle='-')
plt.plot(history.history["loss"], label="Training Loss",color='0.45',linestyle='--')
plt.xlabel("Number of Epoch", **hfont,fontsize=15)
plt.ylabel("Loss", **hfont,fontsize=15)
plt.yticks(np.linspace(0,round(max(history.history["val_loss"]),1),num=5))
plt.legend()
plt.show()


#%%
save_path = './SDS_KKK13.h5'
model.save(save_path)
#%%
save_path = './SDS_KKK1.h5'
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
#threshold = np.max(train_mae_loss)+5*np.std(np.abs(x_train_pred - x_train))
threshold = np.max(train_mae_loss)*5
#threshold = np.mean(np.abs(x_train_pred - x_train)) + 30*np.std(np.abs(x_train_pred - x_train))
print("Reconstruction error threshold: ", threshold)


training_mean = x_train.mean()
training_std = x_train.std()

df_test_value = (X_test - training_mean) / training_std


print("Test input shape: ", x_test.shape)
#%%
# Get test MAE loss.
x_test_pred = model.predict(x_test)
x_test_pred_prob = model.predict_on_batch(x_test_pred)

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
#
df_subset=np.zeros(data_test.shape)
df_subset[anomalous_data_indices,0] = data_test.iloc[anomalous_data_indices,0]
#
#fig, ax = plt.subplots()
#ax.plot(data_test['energy1'])
#ax.plot(data_test.iloc[anomalous_data_indices]['energy1'],color='r')

#%%
velocity_ldv2=pd.read_excel('LDV2_downsample_'+Dataset+'_tstep0001.xls',header=None)

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
  
# read text file into pandas DataFrame
ldv2_acc_org= pd.read_csv('LDV2_acc_original_'+Dataset+'_tstep0001.txt', sep=" ")
#ldv2_acc_org= pd.read_csv("LDV2_acc_original_KKK14.txt", sep=" ")
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
#%%
# indx_org_s=math.floor((indx[0]/len(velocity_ldv2))*len(ldv2_acc_org))
# indx_org_e=math.floor((indx[-1]/len(velocity_ldv2))*len(ldv2_acc_org))
# indx_org=np.arange(indx_org_s,indx_org_e+1)



hfont = {'fontname':'Times New Roman'}
time=np.linspace(0, len(ldv2_acc_org)/fs, num=len(ldv2_acc_org))
fig2, ax2 = plt.subplots()
ax2.plot(time,ldv2_acc_org,color='k',linewidth=0.5)
for i in range(0,len(indx_sep)):
    ax2.plot(time[anom_org[i]],ldv2_acc_org.iloc[anom_org[i]],c='0.45',linewidth=0.1,marker="o",markersize=3,markerfacecolor='none')

#ax2.plot(time[indx_org],ldv2_acc_org.iloc[indx_org],c='0.45',linewidth=0.1,marker="o",markersize=3,markerfacecolor='none')

plt.xlabel("Time(s)", **hfont)
plt.ylabel("Velocity(m/s)", **hfont)
plt.xlim(xmin=min(time), xmax=max(time))
plt.xticks(np.arange(0, time[-1], step=0.5)) 

fig2.set_figwidth(8)
fig2.set_figheight(4)

ldv2_acc_org = ldv2_acc_org.to_numpy()
lim_array=max(ldv2_acc_org)*1.5
lim=round(lim_array[0],2)
#lim=0.06
ax2.set_ylim([-1*lim,lim])
ax2.set_yticks(np.linspace(-1*lim,lim,num=5))
#%%
#plot the time series features
#names=['energy','std','maxabs','skw','kurt','rms','clear','crest','shape','fe','p2p','spectrms','ssd','label']

feature="energy"
data=pd.concat([data_train,data_test],axis=0,ignore_index=True)
time_d=np.linspace(0, len(ldv2_acc_org)/fs,num= len(data))
plt.plot(time_d,data[feature],'k')
plt.xlabel("Time(s)", **hfont)
plt.xlim(xmin=min(time), xmax=max(time))
plt.ylim(ymin=-abs(max(data[feature]))*1.2, ymax=abs(max(data[feature]))*1.2)
lim_feat=round(abs(max(data[feature])),1)
ax2.set_yticks(np.linspace(-1*lim_feat,lim_feat,num=3))
plt.ylabel("Normalized Entropy", **hfont)
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
x_test_pred_orginalsize=extract_original_data(x_test,TIME_STEPS)
x_train_pred_orginalsize=extract_original_data(x_train,TIME_STEPS)
test_sizediff=len(data_test)-len(x_test_pred_orginalsize)
train_sizediff=len(data_train)-len(x_train_pred_orginalsize)
x_test_pred_orginalsize=pd.concat([x_test_pred_orginalsize,x_test_pred_orginalsize.tail(test_sizediff)],axis=0,ignore_index=True)
x_train_pred_orginalsize=pd.concat([x_train_pred_orginalsize,x_train_pred_orginalsize.tail(train_sizediff)],axis=0,ignore_index=True)

x_test_pred_prob_org=extract_original_data(x_test_pred_prob,TIME_STEPS)
#%%


latent_c=anomalies #either anomalies or y_test or data_feature, or data_label

def plot_lat_predict(latent_dim,latent_c):
    latent_representation = encoder.predict(X_test)
    latent_representation = np.array(latent_representation)
    colors = {True:'red',False:'black'} 
    latent_c_color=np.vectorize(colors.get)(latent_c)

    
    if latent_dim == 2:
        plt.scatter(latent_representation[:, 0], latent_representation[:, 1],c=latent_c_color,alpha=0.8)
#        plt.colorbar()
        plt.show()
    else:
        ax = plt.axes(projection ="3d")
        ax.scatter3D(latent_representation[:, 0], latent_representation[:, 1], latent_representation[:, 2], c=latent_c_color,alpha=0.8)
        
        ax.view_init(-150, 60)
#        plt.colorbar()
        

if plot_latent=='True':        
    plot_lat_predict(latent_dim,latent_c)
    
#%%
latent_c=y_test #either anomalies or y_test or ( data_feature or data_label)

def plot_lat_predict(latent_dim,latent_c):
    latent_representation = encoder.predict(x_test)
    latent_representation = np.array(latent_representation)
    colors = {True:'red',False:'black'} 
    latent_c_color=np.vectorize(colors.get)(latent_c)

    
    if latent_dim == 2:
        plt.scatter(latent_representation[:, 0], latent_representation[:, 1],c=latent_c_color,alpha=0.8)
#        plt.colorbar()
        plt.show()
    else:
        ax = plt.axes(projection ="3d")
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
    plot_lat_predict(latent_dim,latent_c)

#%%

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


plt.plot(time_dd,data_label)
plt.xlim(xmin=min(time_dd), xmax=max(time_dd))
plt.grid()
#%%
from sklearn import metrics
from sklearn.metrics import accuracy_score, confusion_matrix, roc_auc_score, roc_curve
anomalies_01=anomalies*1
fpr, tpr, _ = metrics.roc_curve(y_test,  anomalies_01)
auc = metrics.roc_auc_score(y_test, anomalies_01)
#%
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