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
        high = Dmean+3*Dstd
        low = Dmean - 3* Dstd
        outliers = [x for x in df[i] if x < low or x > high or x==0]
        df[i] = df[i].replace(outliers,Dmean)
    return df
# -----------------------transfer dataframe to dataset----------------------------------------
# x_train is a 3D-array:  (none, hops, no_col)
# y_train is a 2D-array: (none, no_col)
def df2ds(df_x,df_y,hops,no_train_data):
    x_train=[]
    y_train=[]
    for i in range(hops,no_train_data-hops):
        x_train.append(df_x[i-hops:i])
        y_train.append(df_y[i:i+hops])
    x_train,y_train = ny.array(x_train),ny.array(y_train)
    return x_train,y_train

# --------------------------input parameters--------------------------------------------------------------------------------------

# read data
#读取数据
#df1 = pd.read_csv("/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/traffic_all2.csv")
#df1 = pd.read_csv("/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/kpi数据_山东_流量.csv")
#df1 = pd.read_csv("/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/lili_业务预测_河南_17地市.csv")
df1 = pd.read_csv("/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/hour_cell_forpredict_jy.csv")
# model saved file path and name
#从这个路径加载模型
file_path_model="/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/0531cell_model_batchsize1"
model = tf.keras.models.load_model(file_path_model)
#从这个路径加载数据适配模型sc和sc2
file_path_sc="/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/sc.joblib"
file_path_sc2="/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/sc2.joblib"
sc = load(file_path_sc)
sc2 = load(file_path_sc2)
#预测结果存储路径
file_path_csv="/Users/lily/Desktop/中讯/项目与资料/planform/KG_RAN/pred_data.csv"
# ---------------------------data information and basic process-------------------------------------------------------------------------
# replace blank cells with 0
df1.fillna(0, inplace=True)
# get all the column names
column_names = list(df1.columns)
# !!!!!! 这里加载的df1多了索引行，需要去掉，直接从数据库读取的数据不需要去除！！！！！
column_names = column_names[1:]
# time slot
# 时间间隔，time_slot为时间的字段名，这里是"小时"，时间格式为 YYYYMMDDHH
time_slot = 'hour_id'
# set LSTM timestep length(if time slot is day, hops=14, if time slot is hour, hops= 48 )
# hops is also used for prediction test (to show predicted v.s. real )
hops = 24*7

#预测的地市名
pre_data_name = column_names[2]
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
# --------------------------predict traffic data --------------------------------------------------
# prepare data set
df2_nodate=df2[column_nodate]
df2_nodate.reset_index(inplace=True)
# remove outlier data from df2_nodate
df2_nodate= remove_outlier(df2_nodate,column_nodate)

#get number of train data items
df_data_shape = df2_nodate.shape
no_train_data = df_data_shape[0]
#data standarlization
df2_train_scaled = sc.fit_transform(df2_nodate)
# transfer dataframe to dataset
x_pred=[]
# here we only predict 2 weeks, first week for compare to real traffic data, second is future prediction
for i in range(0,2):
    x_pred.append(df2_train_scaled[hops*i:hops*(i+1)])
x_pred = ny.array(x_pred)

# ---------------------------------------------------------------prediction and predicted data processing -----------------------
y_pred = model.predict(x_pred)   #用训练好的模型model，输入最后60天的数据，预测得到的y_pred是第61-90天的数据
y_pred_org1 = sc2.inverse_transform(y_pred[0,:,:])
y_pred_org2 = sc2.inverse_transform(y_pred[1,:,:])
pred_total = ny.concatenate((y_pred_org1,y_pred_org2),axis=0)

#predicted data and date process
last_date = str(df1[time_slot][hops])
last_date_obj = datetime.datetime.strptime(last_date, "%Y%m%d%H")
pre_date = []
for i in range(0,2*hops):
    y_date = last_date_obj + datetime.timedelta(hours=i)
    y_date_str = y_date.strftime("%Y%m%d%H")
    pre_date.append(y_date_str)


y_index = pd.Index(list(range(no_train_data-hops,no_train_data+hops)),name ='index')
predicted_total = pd.DataFrame(pred_total,index=y_index,columns = column_nodate)
predicted_total[time_slot] = pre_date

# ------------------------save predicted data ---------------------------------------------------------
predicted_total.to_csv(file_path_csv)
# plot figure
# python绘图语句，可删除
plt.plot(df1[pre_data_name],label=pre_data_name,color='red')
plt.plot(df2_nodate[pre_data_name],label=pre_data_name,color='yellow')
plt.plot(predicted_total[pre_data_name],label='predicted',color='blue')
plt.legend()

