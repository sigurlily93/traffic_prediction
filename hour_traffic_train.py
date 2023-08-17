import datetime
import tensorflow as tf
from tensorflow import keras
import pandas as pd
import numpy as ny
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout, Conv1D, MaxPooling1D,Flatten,RepeatVector,TimeDistributed,Reshape
from joblib import dump
from joblib import load

# ------------------------------------sub-functions---------------------------------------------
# --------------------------remove outlier data--------------------------------------------------
def remove_outlier(df,column_nodate):
    for i in column_nodate:
        Dmean = df[i].mean()
        Dstd = df[i].std()
        high = Dmean+4*Dstd
        low = Dmean - 4* Dstd
        outliers = [x for x in df[i] if x < low or x > high]
        df[i] = df[i].replace(outliers,Dmean)
    return df
# -----------------------transfer dataframe to dataset----------------------------------------
# x_train is a 3D-array:  (none, hops, no_col)
# y_train is a 2D-array: (none, no_col)
def df2ds(df_x,df_y,hops,no_train_data):
    x_train=[]
    y_train=[]
    for i in range(hops,no_train_data-3*hops):
        x_train.append(df_x[i-hops:i+2*hops])
        y_train.append(df_y[i+2*hops:i+3*hops])
    x_train,y_train = ny.array(x_train),ny.array(y_train)
    return x_train,y_train

#------------------------create a model with 2 convolution layers ---------------------------
# defined call back function (early stop the model training)
# defined call back function (early stop the model training)
# defined callback

def create_model_lstm(hops,no_col):
    model = Sequential()
    model.add(Conv1D(filters=no_col, kernel_size=int(3*hops/4), activation='linear',input_shape = (3*hops,no_col)))
    model.add(Conv1D(filters=no_col*2, kernel_size=int(3*hops*3/4), activation='linear'))
    model.add(LSTM(units=128,return_sequences=True))
    model.add(Dropout(0.05))
    model.add(LSTM(units=128))
    model.add(Dropout(0.05))
    model.add(Dense(128,activation='linear'))
    model.add(Dense(hops*(no_col-1),activation='linear'))
    model.add(Reshape((hops,no_col-1)))
    optimizer = keras.optimizers.Adam(learning_rate=0.0005/2)
    #model.compile(optimizer=optimizer,loss='mean_squared_logarithmic_error',metrics=['mse','mae','msle'])
    model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mse','mae','msle'])
    return model

def create_model_baseline(hops,no_col):
    model = Sequential()
    model.add(LSTM(units=128,input_shape = (hops,no_col),return_sequences=True,activation='selu'))
    model.add(Dropout(0.05))
    model.add(LSTM(units=128,activation='selu'))
    model.add(Dropout(0.05))
    model.add(Dense(128,activation='selu'))
    model.add(Dense(hops*(no_col-1),activation='linear'))
    model.add(Reshape((hops,no_col-1)))
    optimizer = keras.optimizers.Adam(learning_rate=0.0005/8)
    #model.compile(optimizer=optimizer,loss='mean_squared_logarithmic_error',metrics=['mse','mae','msle'])
    model.compile(optimizer=optimizer,loss='mean_squared_error',metrics=['mse','mae','msle'])
    return model

# --------------------------input parameters--------------------------------------------------------------------------------------
# read data
#读取数据
#df1 = pd.read_csv("/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/traffic_all2.csv")
#df1 = pd.read_csv("/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/kpi数据_山东_流量.csv")
#df1 = pd.read_csv("/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/lili_业务预测_河南_17地市.csv")
df1 = pd.read_csv("/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/2023年部分小区kpi数据.csv")
# model saved file path and name
#模型存储位置
file_path_model="/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/0531cell_model_batchsize1"
#存储归一化处理数据模型
file_path_sc="/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/sc.joblib"
file_path_sc2="/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/sc2.joblib"

#-----------------------data information and DNN architecture parameters ------------------------------
#这部分是模型框架相关参数
# replace blank cells with 0
df1.fillna(0, inplace=True)
# get all the column names
column_names = list(df1.columns)
# choose columns that used for machine learning
column_names = column_names[0:]
# time slot
# 时间间隔，time_slot为时间的字段名，这里是"天"，时间格式为 YYYYMMDD
time_slot = 'hour_id'
# set LSTM timestep length(if time slot is day, hops=14, if time slot is hour, hops= 48 )
# hops is also used for prediction test (to show predicted v.s. real )
hops = 24*7
# number of iterations (the more iterations, the longer time of training the model)
no_iteration = 100
# batch size (from 1 to number of training data, the more batch size, the shorter training time)
batchsize = 1
# predicted column name
#预测的地市名
pre_data_name = column_names[2]

#-----------------------------------------------------------------------------------------------------------
#---------------------------intermediate parameters --------------------------------------------------------
# get column names without date(date in the first column)
column_nodate = column_names[1:]
df2 = df1[column_names]
# get date information
df2_date = df1[[time_slot]]
# get dataset shape
df_shape = df2.shape
# number of training data samples
no_records = df_shape[0]
# number of dataset columns
no_col = df_shape[1]

#--------------------------------------------------------------------------------------------------------


# split data set
df2_nodate=df2[column_nodate]
df2_nodate.reset_index(inplace=True)
# remove outlier data from df2_no_date
df2_nodate= remove_outlier(df2_nodate,column_nodate)
# here no test dataset
df2_train = df2_nodate
#df2_test = df2_nodate.iloc[no_records-hops:,:]


#get number of train data items
train_data_shape = df2_train.shape
no_train_data = train_data_shape[0]


#data standarlization
sc =StandardScaler()
df2_train_scaled = sc.fit_transform(df2_train)
sc2 = StandardScaler()
df2_train_scaled_y = sc2.fit_transform(df2_train[column_nodate])
# save sc and sc2
#存储sc和sc2
dump(sc, file_path_sc)
dump(sc2, file_path_sc2)



# set input and output data format according to model tensor
x_train,y_train = df2ds(df2_train_scaled,df2_train_scaled_y,hops,no_train_data)
#------------------------------------create and train a model-------------------------------------------------------------------
#model=create_model(hops,no_col)

# defined call back function (early stop the model training)
# defined callback
es=tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=10,verbose=1,  restore_best_weights=True)
rlronp=tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=10,verbose=0)
callbacks=[es, rlronp]
best_epo = 0
while(best_epo<batchsize):
    model=create_model_lstm(hops,no_col)
    history = model.fit(x_train,y_train,epochs=no_iteration,batch_size=batchsize,validation_split=0.1,callbacks=callbacks)
    best_epo=callbacks[0].best_epoch

# history = model.fit(x_train,y_train,epochs=no_iteration,batch_size=batchsize,validation_split=0.1)

# 这两句代码用来显示训练过程中的loss
pd.DataFrame(history.history).plot(figsize=(8,5))
plt.show()

# save model
model.save(file_path_model)

