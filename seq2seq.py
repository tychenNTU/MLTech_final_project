from model import Seq2seq
import pandas as pd
from pandas import to_datetime
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import util
import os
import csv
plot = False
# model = 'Transformer'
model = 'Seq2seq'
# model = 'Prophet'

rescale = 'MinMax'
# rescale = 'Standard'
# rescale options: MinMax, Standard ,None
# model option: Prophet, seq2seq

# load data
path = os.getcwd() + r'\processed_data\data.csv'
df = pd.read_csv(path, header = 0)

# plot origin data
if plot:
    print(df)    
    df.plot()
    plt.show()

df.columns = ['ds', 'y']

output_seq_len = 153

# fit the model
if rescale == 'Standard':
    scaler = StandardScaler()
else:
    scaler = MinMaxScaler(feature_range=(-1,1))
x = df['ds'].values
y = scaler.fit_transform(df['y'].values.reshape(-1,1))
y_true = df['y'].values[-output_seq_len:].copy()

if model == 'Prophet':
    df['ds'] = to_datetime(df['ds'])
    df['y'] = y.reshape(-1)
    model = Prophet()
    # model.fit(df)
    model.fit(df[:-output_seq_len])

    # pred = model.predict(df)
    pred = model.predict(df[-output_seq_len:])
    y_pred = scaler.inverse_transform(pred['yhat'].values.reshape(-1,1)).reshape(-1)
    y_forecast = y_pred
    
elif model == 'Seq2seq':
    input_seq_len = 365

    model = Seq2seq(input_seq_len, output_seq_len, layers=[256],bidirectional=False)
    model.train(y, batches=1, epochs=20, batch_size=5)

    input_seq_test = y[-input_seq_len - output_seq_len: -output_seq_len].reshape((1,input_seq_len,1))
    pred = model.predict(input_seq_test)
    y_pred = scaler.inverse_transform(pred.reshape(-1,1)).reshape(-1)
    
    input_seq_test = y[-input_seq_len:].reshape((1,input_seq_len,1))
    forecast = model.predict(input_seq_test)
    y_forecast = scaler.inverse_transform(forecast.reshape(-1,1)).reshape(-1)
    

else:
    print('請輸入正確的model名稱')

print('y_true:', y_true)
print('y_pred:', y_pred)
print('y_forecast:', y_forecast)
MAE = util.MAE(y_pred, y_true)
ME = util.ME(y_pred, y_true)
MSE = util.MSE(y_pred, y_true)
RMSE = util.RMSE(y_pred, y_true)
print('MAE:', MAE)
print('ME:', ME)
print('MSE:', MSE)
print('RMSE:', RMSE)

y = df['y'].values
with open(os.path.join(os.getcwd(), 'prediction_seq2seq.csv'), 'w',newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['date', 'revenue', 'revenue_prediction'])
    for i in range(len(x)):
        if i + output_seq_len < len(x):
            writer.writerow([x[i], y[i], 0])
        else:
            #print(i, len(x), len(y), len(y_pred), i + output_seq_len - len(x))
            writer.writerow([x[i], y[i], y_pred[i + output_seq_len - len(x)]])
csvfile.close()

# plt.plot(x, y_true, label='Actual')
# plt.plot(x, y_pred, label='Predicted')
# model.plot(forecast)
# plt.show()
