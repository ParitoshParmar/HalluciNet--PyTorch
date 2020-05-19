# Author: Paritosh

randomseed = 0

# # splits directory
# splits_dir = '/home/paritosh/pytorch_tutos/model_3dcnn_small/splits/'

# input dims
model_2dcnn_C, model_2dcnn_H, model_2dcnn_W = 3, 227, 227
model_3dcnn_C, model_3dcnn_H, model_3dcnn_W = 3, 112, 112

train_batch_size = 25
test_batch_size = 50

num_classes = 101

model_2dcnn_size = 320,240
model_3dcnn_size = 171,128