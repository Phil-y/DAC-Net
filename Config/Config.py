
import os
import torch
import time


# PARAMETERS OF THE MODEL
save_model = True
tensorboard = True
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
use_cuda = torch.cuda.is_available()
seed = 666
os.environ['PYTHONHASHSEED'] = str(seed)

n_filts = 16
cosineLR = True # whether use cosineLR or not
n_channels = 3
n_labels = 1

learning_rate = 1e-3
batch_size = 4
img_size = 224

print_frequency = 1
save_frequency = 5000
vis_frequency = 10

epochs = 300
early_stopping_patience = 50

pretrain = False




task_name = 'DDTI'
# task_name = 'TN3K'




model_name = 'DAC_Net'







train_dataset = './datasets/'+ task_name+ '/Train_Folder/'
val_dataset = './datasets/'+ task_name+ '/Val_Folder/'
test_dataset = './datasets/'+ task_name+ '/Test_Folder/'

session_name       = 'Test_session' + '_' + time.strftime('%m.%d_%Hh%M')
save_path          ="./data_train_test_session/" + task_name +'/'+ model_name +'/' + session_name + '/'

model_path         = save_path + 'models/'
tensorboard_folder = save_path + 'tensorboard_logs/'
logger_path        = save_path + session_name + ".log"
visualize_path     = save_path + 'visualize_val/'

# used in testing phase, copy the session name in training phase
test_session = "Test_session_03.13_15h25"
# test_session = "Test_session"






