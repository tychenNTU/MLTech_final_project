import pickle as pkl

with open("preprocess/processed_train.pkl", "rb") as f:
    global train_x, train_y
    train_x, train_y = pkl.load(f)

print(train_x.shape)
print(train_x[:5])
# print('There are %d original features.' % np.array(x).shape)