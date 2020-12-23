import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, LSTMCell, RNN, Bidirectional, concatenate
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler


class Seq2seq:
    def __init__(self, input_seq_len = 60, output_seq_len = 14, layers = [128], n_in_features = 1, n_out_features = 1, bidirectional=False, loss = 'mean_absolute_error'):
        self.input_seq_len = input_seq_len
        self.output_seq_len = output_seq_len
        self.n_in_features = n_in_features
        self.n_out_features = n_out_features
        n_layers = len(layers)
        
        ## Encoder
        encoder_inputs = Input(shape=(None, n_in_features))
        lstm_cells = [LSTMCell(hidden_dim) for hidden_dim in layers]
        # lstm_cells = LSTMCell(layers)
        if bidirectional:
            encoder = Bidirectional(RNN(lstm_cells, return_state=True))
            encoder_outputs_and_states = encoder(encoder_inputs)
            bi_encoder_states = encoder_outputs_and_states[1:]
            encoder_states = []
            for i in range(int(len(bi_encoder_states)/2)):
                temp = concatenate([bi_encoder_states[i],bi_encoder_states[2*n_layers + i]], axis=-1)
                encoder_states.append(temp)
        else:  
            encoder = RNN(lstm_cells, return_state=True)
            encoder_outputs_and_states = encoder(encoder_inputs)
            encoder_states = encoder_outputs_and_states[1:]
    
        ## Decoder
        decoder_inputs = Input(shape=(None, n_out_features))
        if bidirectional:
            decoder_cells = [LSTMCell(hidden_dim*2) for hidden_dim in layers]
        else:
            decoder_cells = [LSTMCell(hidden_dim) for hidden_dim in layers]
            
        decoder_lstm = RNN(decoder_cells, return_sequences=True, return_state=True)

        decoder_outputs_and_states = decoder_lstm(decoder_inputs,
                                                initial_state=encoder_states)
        decoder_outputs = decoder_outputs_and_states[0]

        decoder_dense = Dense(n_out_features) 
        decoder_outputs = decoder_dense(decoder_outputs)

        self.model = Model([encoder_inputs,decoder_inputs], decoder_outputs)
        self.model.compile(Adam(), loss = loss)

    def generate_train_sequences(self, x):
        input_seq_len = self.input_seq_len
        output_seq_len = self.output_seq_len
        total_start_points = len(x) - input_seq_len - output_seq_len
        start_x_idx = np.random.choice(range(total_start_points), total_start_points, replace = False)
        input_batch_idxs = [(range(i, i+input_seq_len)) for i in start_x_idx]
        input_seq = np.take(x, input_batch_idxs, axis = 0)
        
        output_batch_idxs = [(range(i+input_seq_len, i+input_seq_len+output_seq_len)) for i in start_x_idx]
        output_seq = np.take(x, output_batch_idxs, axis = 0)
        
        input_seq =(input_seq.reshape(input_seq.shape[0],input_seq.shape[1], self.n_in_features))
        output_seq=(output_seq.reshape(output_seq.shape[0],output_seq.shape[1], self.n_out_features))
        
        return input_seq, output_seq

    def train(self, x_train, batches = 1,epochs = 100,batch_size = 10):

        total_loss = []
        total_val_loss = []
        for _ in range(batches):
            input_seq, output_seq = self.generate_train_sequences(x_train)
            encoder_input_data = input_seq
            decoder_target_data = output_seq
            decoder_input_data = np.zeros(decoder_target_data.shape)

            history = self.model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
                                batch_size=batch_size,
                                epochs=epochs,
                                validation_split=0.1, 
                                shuffle=False)
                            
            total_loss.append(history.history['loss'])
            total_val_loss.append(history.history['val_loss'])
        return total_loss, total_val_loss
    
    def predict(self, input_seq_test):
        decoder_input_test = np.zeros((1,self.output_seq_len,1))
        return self.model.predict([input_seq_test,decoder_input_test])
"""
input_seq_len = 60
output_seq_len = 14
n_in_features = 1
n_out_features = 1
batch_size = 10

path = os.getcwd() + r'\data\data_prophet.csv'
df = pd.read_csv(path, header = 0)

df.columns = ['ds', 'y']

scaler = MinMaxScaler(feature_range=(-1,1))
"""
# 作弊偷看test
# x_train = scaler.fit_transform(df['y'].values.reshape(-1,1))

# 不偷看
"""
x_train = scaler.fit_transform(df['y'].values[:-input_seq_len-output_seq_len].reshape(-1,1))

model_1 = create_model(layers=[128],bidirectional=False)
total_loss = []
total_val_loss = []
model_1.compile(Adam(), loss = 'mean_squared_error')
run_model(model_1,batches=1, epochs=100, batch_size=batch_size)

total_loss = [j for i in total_loss for j in i]
total_val_loss = [j for i in total_val_loss for j in i]

input_seq_test = x_train[-input_seq_len - output_seq_len: -output_seq_len].reshape((1,input_seq_len,1))
output_seq_test = x_train[-output_seq_len:]
decoder_input_test = np.zeros((1,output_seq_len,1))

pred1 = model_1.predict([input_seq_test,decoder_input_test])

pred_values1 = scaler.inverse_transform(pred1.reshape(-1,1))
output_seq_test1 = scaler.inverse_transform(output_seq_test)

print(pred_values1)
"""
"""

plt.plot(pred_values1, label = "pred")
plt.plot(output_seq_test1, label = "actual")
plt.title("Prediction vs Actual")
plt.ylabel("Data", fontsize=12)
plt.xlabel("Date", fontsize=12)
plt.legend()
plt.savefig('uni_dir1.png')
plt.show()
"""