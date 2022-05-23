from inspect import Parameter
from statistics import mode
from braindecode.preprocessing import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.models import ShallowFBCSPNet, EEGNetv1, EEGNetv4, Deep4Net
from braindecode.augmentation import FrequencyShift, Mixup, FTSurrogate
from braindecode.preprocessing import create_windows_from_events
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import ShuffleSplit, train_test_split
from braindecode.util import set_random_seeds
from braindecode.datasets import MOABBDataset
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from braindecode import EEGClassifier
from ModelFamily import SCCNet, HSCNN, SCCTransformer, EEGNet, EEGNet_TCN, TCN
from torch.optim.lr_scheduler import StepLR
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
from torchsummary import summary
from torch import Tensor, device
import seaborn as sns
from torch import nn
import numpy as np
import argparse
import pickle
import torch
import math
from tqdm import tqdm
import mne

def read_data(subject_id, high_hz, low_hz, sample_rate):
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

    
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000
    ch_names = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    include = ['C3', 'C4', 'Cz'] #7 9 11
    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        #Preprocessor('pick_channels', ch_names=ch_names ,include=include),
        Preprocessor('resample',sfreq=sample_rate),
        Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_hz, h_freq=high_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                    factor_new=factor_new, init_block_size=init_block_size)
    ]

    # Transform the data
    preprocess(dataset, preprocessors)

    trial_start_offset_seconds = -0.5
    
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sample_rate)

    # Create windows using braindecode function for this. It needs parameters to define how
    # trials should be used.
    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
    )

    splitted = windows_dataset.split('session')
    train_set = splitted['session_T']
    valid_set = splitted['session_E']


    return train_set, valid_set
    

def run_braindecode_model(eeg_model, subject_id, n_iter, train_set, valid_set):

    #valid_set[trial_number][0] -> X (22,1125)
    #valid_set[trial_number][1] -> Y (1)

    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    # Set random seed to be able to roughly reproduce results
    # Note that with cudnn benchmark set to True, GPU indeterminism
    # may still make results substantially different between runs.
    # To obtain more consistent results at the cost of increased computation time,
    # you can set `cudnn_benchmark=False` in `set_random_seeds`
    # or remove `torch.backends.cudnn.benchmark = True`
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    n_classes = 4
    # Extract number of chans and time steps from dataset
    n_chans = train_set[0][0].shape[0]
    input_window_samples = train_set[0][0].shape[1]



    # Model instruction (Braindecode)
   
    model = eeg_model(
        n_chans,
        n_classes,
        input_window_samples=input_window_samples,
        n_filters_time=25,
        n_filters_spat=25,
        stride_before_pool=True,
        n_filters_2=int(n_chans * 2),
        n_filters_3=int(n_chans * (2 ** 2.0)),
        n_filters_4=int(n_chans * (2 ** 3.0)),
        final_conv_length = 200,
    )
    
    if cuda:
        model.cuda()

    lr = 0.0625 * 0.01
    weight_decay = 0
    batch_size = 64
    n_epochs = 200

    clf = EEGClassifier(
        model,
        criterion=torch.nn.NLLLoss,
        optimizer=torch.optim.AdamW,
        train_split=predefined_split(valid_set),  # using valid_set for validation
        optimizer__lr=lr,
        optimizer__weight_decay=weight_decay,
        batch_size=batch_size,
        callbacks=[
            "accuracy", ("lr_scheduler", LRScheduler('CosineAnnealingLR', T_max=n_epochs - 1)),
        ],
        device=device,
    )
    
    clf.fit(train_set, y=None, epochs=n_epochs)
    


    # save the model to disk
    filename = f'./{eeg_model.__name__}/{eeg_model.__name__}_S{subject_id}_Iter{n_iter}.sav'
    pickle.dump(clf, open(filename, 'wb'))
 
def to_dataloader_onde_subject(train_set, valid_set, batch_size):

    train_X = []
    for i in range(288):
        train_X.append(train_set[i][0])
    train_X = np.stack(train_X, axis=0)

    train_Y = []
    for i in range(288):
        train_Y.append(np.array(train_set[i][1]))
    train_Y = np.stack(train_Y, axis=0)

    valid_X = []
    for i in range(288):
        valid_X.append(valid_set[i][0])
    valid_X = np.stack(valid_X, axis=0)

    valid_Y = []
    for i in range(288):
        valid_Y.append(np.array(valid_set[i][1]))
    valid_Y = np.stack(valid_Y, axis=0)

    # train_x: 288 1 22 438 valid_x: 288 1 22 438
    # train_y: 288          valid_y: 288
    train_X = train_X.reshape(-1,1,22,438)
    valid_X = valid_X.reshape(-1,1,22,438) 

    # X = np.concatenate((train_X, valid_X))
    # y = np.concatenate((train_Y, valid_Y))

    # train_X, valid_X, train_Y, valid_Y = train_test_split(X, y, test_size=0.1, random_state=7)

    train_set_tensor = Tensor(train_X)
    train_label_tensor = Tensor(train_Y).type(torch.LongTensor)

    train_dataset = TensorDataset(train_set_tensor, train_label_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    test_set_tensor = Tensor(valid_X)
    test_label_tensor = Tensor(valid_Y).type(torch.LongTensor)

    test_dataset = TensorDataset(test_set_tensor, test_label_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def get_numpy_data(train_set, valid_set):

    train_X = []
    for i in range(288):
        train_X.append(train_set[i][0])
    train_X = np.stack(train_X, axis=0)

    train_Y = []
    for i in range(288):
        train_Y.append(np.array(train_set[i][1]))
    train_Y = np.stack(train_Y, axis=0)

    valid_X = []
    for i in range(288):
        valid_X.append(valid_set[i][0])
    valid_X = np.stack(valid_X, axis=0)

    valid_Y = []
    for i in range(288):
        valid_Y.append(np.array(valid_set[i][1]))
    valid_Y = np.stack(valid_Y, axis=0)

    # 438 or 875
    # 22 or 3
    # train_X = np.stack((train_X[:,6,:],train_X[:,8,:],train_X[:,10,:]),axis=1)
    # valid_X = np.stack((valid_X[:,6,:],valid_X[:,8,:],valid_X[:,10,:]),axis=1)
    print(train_X.shape,valid_X.shape)
    train_X = train_X.reshape(-1,1,22,1125)
    valid_X = valid_X.reshape(-1,1,22,1125) 
    

    return train_X, train_Y, valid_X, valid_Y

def numpy_data_to_dataloader(train_X, train_Y, valid_X, valid_Y, batch_size):

    train_set_tensor = Tensor(train_X)
    train_label_tensor = Tensor(train_Y).type(torch.LongTensor)

    train_dataset = TensorDataset(train_set_tensor, train_label_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    test_set_tensor = Tensor(valid_X)
    test_label_tensor = Tensor(valid_Y).type(torch.LongTensor)

    test_dataset = TensorDataset(test_set_tensor, test_label_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

# train function
def train(model, optimizer, train_loader, criterion, mode = 'train'):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if mode == 'train':
        model.train()
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        #loss = criterion(output, target) + 0.01 * torch.square(torch.norm(model.fc1[0].weight))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    return correct / total

# train function
def train_with_scheduler(model, optimizer, scheduler, train_loader, criterion):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    correct = 0
    total = 0
    
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target) + 0.01 * torch.square(torch.norm(model.fc1[0].weight))
        loss.backward()
        optimizer.step()
        
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

    scheduler.step()

    return correct / total
# test function
def test(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

def test_return_predict(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    pred_seq = []
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(data_loader)):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            pred_seq.append(predicted)


    return torch.concat(pred_seq)

def run_SCCNet_model(subject_id, n_iter, train_set, valid_set, try_aug):
    
    #input shape ()

    cuda = torch.cuda.is_available()  # check if GPU is available, if True chooses to use it
    device = 'cuda' if cuda else 'cpu'
    if cuda:
        torch.backends.cudnn.benchmark = True
    # Set random seed to be able to roughly reproduce results
    # Note that with cudnn benchmark set to True, GPU indeterminism
    # may still make results substantially different between runs.
    # To obtain more consistent results at the cost of increased computation time,
    # you can set `cudnn_benchmark=False` in `set_random_seeds`
    # or remove `torch.backends.cudnn.benchmark = True`
    seed = 20200220
    set_random_seeds(seed=seed, cuda=cuda)

    lr = 0.0625 * 0.01
    weight_decay = 1e-1
    batch_size = 64
    n_epochs = 200

    #--------------augmentation---------------------#
    train_X, train_Y, valid_X, valid_Y = get_numpy_data(train_set, valid_set)

    if try_aug == True:
        transform = FTSurrogate(probability=0.5)
        augmented_train_X, _ = transform.operation(torch.as_tensor(train_X).float(), None, 0.5)
        train_X = np.concatenate((augmented_train_X, train_X), axis=0)
        train_Y = np.concatenate((train_Y, train_Y), axis=0)

    train_loader, test_loader = numpy_data_to_dataloader(train_X, train_Y, valid_X, valid_Y, batch_size)

    model = SCCTransformer(num_classes=4).to(device=device)
    #model = SCCNet(num_classes=4).to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        train_acc = train(model, optimizer, train_loader, criterion)
        test_acc = test(model, test_loader)
        print(
            "SCCNet subject: {} iter: {} Epoch [{}/{}], Training Accuracy: {:.4f}%, Testing Accuracy: {:.4f}%".format(
                subject_id, n_iter, epoch + 1, n_epochs, train_acc * 100, test_acc * 100
            )
        )
    
    torch.save(model.state_dict(), f'SCCNet/SCCNet_S{subject_id}_Iteration_{n_iter}')
   

def run_TCN_Fusion_Model(subject_id, n_iter, train_set, valid_set):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 5e-4
    weight_decay = 9e-4
    batch_size = 64
    n_epochs = 500
    criterion = nn.CrossEntropyLoss()
    train_X, train_Y, valid_X, valid_Y = get_numpy_data(train_set, valid_set)
    train_loader, test_loader = numpy_data_to_dataloader(train_X, train_Y, valid_X, valid_Y, batch_size)

    model = EEGNet_TCN().to(device=device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        train_acc = train(model, optimizer, train_loader, criterion)
        test_acc = test(model, test_loader)
        print(
            "EEGNet_TCN subject: {} iter: {} Epoch [{}/{}], Training Accuracy: {:.4f}%, Testing Accuracy: {:.4f}%".format(
                subject_id, n_iter, epoch + 1, n_epochs, train_acc * 100, test_acc * 100
            )
        )
    
    torch.save(model.state_dict(), f'EEGNet_TCN/EEGNet_TCN_S{subject_id}_Iteration_{n_iter}')

def run_TCN_Fusion():
    # #load data
    subject_data =[]
    for subject_id in range(1,10): 
        subject_data.append(read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=250))

    #run model
    for i in range(9):
        train_set, valid_set = subject_data[i]
        for n_iter in range(10):
            run_TCN_Fusion_Model(subject_id= i+1,n_iter=n_iter ,train_set=train_set, valid_set=valid_set)



def run_SCCNet(try_aug):
    # #load data
    subject_data =[]
    for subject_id in range(1,10): 
        subject_data.append(read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=125))

    #run model
    for i in range(9):
        train_set, valid_set = subject_data[i]
        for n_iter in range(10):
            run_SCCNet_model(subject_id= i+1,n_iter=n_iter ,train_set=train_set, valid_set=valid_set, try_aug=try_aug)

def run_augment_playground(try_aug):
    # Read data
    train_set, valid_set = read_data(subject_id=2, high_hz=38, low_hz=0.5, sample_rate=125)
    train_X, train_Y, valid_X, valid_Y = get_numpy_data(train_set, valid_set)

    # Data Augment not transform
    if try_aug == True:
        transform = FTSurrogate(probability=0.5)
        augmented_train_X, _ = transform.operation(torch.as_tensor(train_X).float(), None, 0.5)
        train_X = np.concatenate((augmented_train_X, train_X), axis=0)
        train_Y = np.concatenate((train_Y, train_Y), axis=0)

    print(train_X.shape, train_Y.shape)

    # Hyperparameter
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.0625 * 0.01
    weight_decay = 1e-4
    batch_size = 64
    n_epochs = 200
    criterion = nn.CrossEntropyLoss()
    model = SCCNet(num_classes=4).to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    train_loader, test_loader = numpy_data_to_dataloader(train_X, train_Y, valid_X, valid_Y, batch_size)
    
    # Training
    for epoch in range(n_epochs):
        train_acc = train(model, optimizer, train_loader, criterion)
        test_acc = test(model, test_loader)
        print(
            "SCCNet aug: {} Epoch [{}/{}], Training Accuracy: {:.4f}%, Testing Accuracy: {:.4f}%".format(
                "YES" if try_aug else "NO" ,epoch + 1, n_epochs, train_acc * 100, test_acc * 100
            )
        )
    return test(model, test_loader)


def wrong_run_SCCNet_cross_validation():
        # load data
    subject_data =[]
    for subject_id in range(1,10): 
        subject_data.append(read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=125))

    total_train_X = []
    total_train_Y = []
    total_valid_X = []
    total_valid_Y = []

    for i in range(9):
        train_X, train_Y, valid_X, valid_Y = get_numpy_data(subject_data[i][0], subject_data[i][1])
        total_train_X.append(train_X)
        total_train_Y.append(train_Y)
        total_valid_X.append(valid_X)
        total_valid_Y.append(valid_Y)

    train_X = np.concatenate(total_train_X, axis=0)
    train_Y = np.concatenate(total_train_Y, axis=0)
    valid_X = np.concatenate(total_valid_X, axis=0)
    valid_Y = np.concatenate(total_valid_Y, axis=0)

    X = np.concatenate((train_X, valid_X), axis=0)
    Y = np.concatenate((train_Y, valid_Y), axis=0)

    rs = ShuffleSplit(n_splits=5, random_state=0)

    score = []
    for train_index, test_index in rs.split(X):
        train_X = X[train_index]
        train_Y = Y[train_index]
        valid_X = X[test_index]
        valid_Y = Y[test_index]
        #(4665, 1, 22, 438) (4665,) (519, 1, 22, 438) (519,)
        

        # Hyperparameter
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        lr = 0.0625 * 0.01
        weight_decay = 1e-4
        batch_size = 64
        n_epochs = 200
        criterion = nn.CrossEntropyLoss()
        model = SCCNet(num_classes=4).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        train_loader, test_loader = numpy_data_to_dataloader(train_X, train_Y, valid_X, valid_Y, batch_size)
        
        # Training
        for epoch in range(n_epochs):
            train_acc = train(model, optimizer, train_loader, criterion)
            test_acc = test(model, test_loader)
            print(
                "SCCNet Epoch [{}/{}], Training Accuracy: {:.4f}%, Testing Accuracy: {:.4f}%".format(
                    epoch + 1, n_epochs, train_acc * 100, test_acc * 100
                )
            ) 
        
        score.append(test(model, test_loader))
    
    subject_score = []
    for i in range(9):
        train_set,valid_set = subject_data[i]
        _, test_loader = to_dataloader_onde_subject(train_set,valid_set,64)
        accuracy = test(model, test_loader)
        subject_score.append(accuracy)
        print(f'S{i+1}_Score:{accuracy}')
    
    print(f"5-fold accuracy{sum(score)/len(score)} average accuracy:{sum(subject_score)/len(subject_score)}")   

def run_9010_HSCNN_model(subject_id, train_set_4, valid_set_4, train_set_8, valid_set_8, train_set_13, valid_set_13):

    # train_set_4, valid_set_4 = read_data(subject_id=subject_id, high_hz=9, low_hz=4, sample_rate=250)
    # train_set_8, valid_set_8 = read_data(subject_id=subject_id, high_hz=15, low_hz=8, sample_rate=250)
    # train_set_13, valid_set_13 = read_data(subject_id=subject_id, high_hz=31, low_hz=14, sample_rate=250)

    train_X_4, train_Y_4, valid_X_4, valid_Y_4 = get_numpy_data(train_set_4, valid_set_4)
    train_X_8, train_Y_8, valid_X_8, valid_Y_8 = get_numpy_data(train_set_8, valid_set_8)
    train_X_13, train_Y_13, valid_X_13, valid_Y_13 = get_numpy_data(train_set_13, valid_set_13)

    train_X = np.stack([train_X_4,train_X_8,train_X_13],axis=1)
    train_Y = train_Y_4
    valid_X = np.stack([valid_X_4,valid_X_8,valid_X_13],axis=1)
    valid_Y = valid_Y_4
    
    print(train_X.shape,train_Y.shape,valid_X.shape,valid_Y.shape)

    X = np.concatenate((train_X, valid_X), axis=0)
    Y = np.concatenate((train_Y, valid_Y), axis=0)

    rs = ShuffleSplit(n_splits=10, random_state=7)
    
    # parameter
    device = 'cuda' if  torch.cuda.is_available() else 'cpu'

    lr = 1e-2
    weight_decay = 1e-2
    batch_size = 64
    n_epochs = 400
    criterion = nn.CrossEntropyLoss()

    score = []
    n_fold = 0
    for train_index, test_index in rs.split(X):
        n_fold += 1
        train_X = X[train_index]
        train_Y = Y[train_index]
        valid_X = X[test_index]
        valid_Y = Y[test_index]
        model = HSCNN(num_classes=4).to(device=device)
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
        train_loader, test_loader = numpy_data_to_dataloader(train_X, train_Y, valid_X, valid_Y, batch_size)

        # Training
        for epoch in range(n_epochs):
            train_acc = train_with_scheduler(model, optimizer, scheduler, train_loader, criterion)
            test_acc = test(model, test_loader)
            print(
                "HSCNN Fold: {} Subject: {} Epoch [{}/{}], Training Accuracy: {:.4f}%, Testing Accuracy: {:.4f}%".format(
                    n_fold , subject_id, epoch + 1, n_epochs, train_acc * 100, test_acc * 100
                )
            ) 
        
        score.append(test(model, test_loader))

    return sum(score)/len(score)

def run_individual_HSCNN_model(subject_id, train_set_4, valid_set_4, train_set_8, valid_set_8, train_set_13, valid_set_13):

    # train_set_4, valid_set_4 = read_data(subject_id=subject_id, high_hz=9, low_hz=4, sample_rate=250)
    # train_set_8, valid_set_8 = read_data(subject_id=subject_id, high_hz=15, low_hz=8, sample_rate=250)
    # train_set_13, valid_set_13 = read_data(subject_id=subject_id, high_hz=31, low_hz=14, sample_rate=250)

    train_X_4, train_Y_4, valid_X_4, valid_Y_4 = get_numpy_data(train_set_4, valid_set_4)
    train_X_8, train_Y_8, valid_X_8, valid_Y_8 = get_numpy_data(train_set_8, valid_set_8)
    train_X_13, train_Y_13, valid_X_13, valid_Y_13 = get_numpy_data(train_set_13, valid_set_13)

    train_X = np.stack([train_X_4,train_X_8,train_X_13],axis=1) # train_X: b 3 1 22 875
    train_Y = train_Y_4
    valid_X = np.stack([valid_X_4,valid_X_8,valid_X_13],axis=1)
    valid_Y = valid_Y_4
    
    # parameter
    device = 'cuda' if  torch.cuda.is_available() else 'cpu'

    lr = 1e-3
    weight_decay = 1e-2
    batch_size = 64
    n_epochs = 200
    criterion = nn.CrossEntropyLoss()


    model = HSCNN(num_classes=4).to(device=device)
    #optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.9)
    train_loader, test_loader = numpy_data_to_dataloader(train_X, train_Y, valid_X, valid_Y, batch_size)

    # Training
    for epoch in range(n_epochs):
        #train_acc = train_with_scheduler(model, optimizer, scheduler, train_loader, criterion)
        train_acc = train(model, optimizer, train_loader, criterion)
        test_acc = test(model, test_loader)
        print(
            "HSCNN  Subject: {} Epoch [{}/{}], Training Accuracy: {:.4f}%, Testing Accuracy: {:.4f}%".format(
                subject_id, epoch + 1, n_epochs, train_acc * 100, test_acc * 100
            )
        ) 
    
    return test_acc


def run_9010_SCCNet(subject_id, train_set, valid_set ,try_aug):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 0.0625 * 0.01
    weight_decay = 1e-3
    batch_size = 64
    n_epochs = 200
    criterion = nn.CrossEntropyLoss()
    

    train_X, train_Y, valid_X, valid_Y = get_numpy_data(train_set, valid_set)

    if try_aug == True:
        transform = FTSurrogate(probability=0.5)
        augmented_train_X, _ = transform.operation(torch.as_tensor(train_X).float(), None, 0.5)
        train_X = np.concatenate((augmented_train_X, train_X), axis=0)
        train_Y = np.concatenate((train_Y, train_Y), axis=0)

    X = np.concatenate((train_X, valid_X), axis=0)
    Y = np.concatenate((train_Y, valid_Y), axis=0)

    rs = ShuffleSplit(n_splits=10, random_state=7)

    score = []
    n_fold = 0
    for train_index, test_index in rs.split(X):
        n_fold += 1
        train_X = X[train_index]
        train_Y = Y[train_index]
        valid_X = X[test_index]
        valid_Y = Y[test_index]
        model = SCCNet(num_classes=4).to(device=device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        train_loader, test_loader = numpy_data_to_dataloader(train_X, train_Y, valid_X, valid_Y, batch_size)

        # Training
        for epoch in range(n_epochs):
            train_acc = train(model, optimizer, train_loader, criterion)
            test_acc = test(model, test_loader)
            print(
                "SCCNet Fold: {} Subject: {} Epoch [{}/{}], Training Accuracy: {:.4f}%, Testing Accuracy: {:.4f}%".format(
                    n_fold , subject_id, epoch + 1, n_epochs, train_acc * 100, test_acc * 100
                )
            ) 
        
        score.append(test(model, test_loader))

    return sum(score)/len(score)
       
def run_SCCNet_cross_validation(try_aug):
    # #load data
    subject_data =[]
    for subject_id in range(1,10): 
        subject_data.append(read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=125))

    #run model
    score = []
    for i in range(9):
        train_set, valid_set = subject_data[i]
        test_acc = run_9010_SCCNet(subject_id= i+1, train_set=train_set, valid_set=valid_set,try_aug=try_aug)
        score.append(test_acc)

    print(score)


def run_HSCNN_cross_validation():

    # #load data
    subject_data =[]
    for subject_id in range(1,10): 
        train_set_4, valid_set_4 = read_data(subject_id=subject_id, high_hz=9, low_hz=4, sample_rate=250)
        train_set_8, valid_set_8 = read_data(subject_id=subject_id, high_hz=15, low_hz=8, sample_rate=250)
        train_set_13, valid_set_13 = read_data(subject_id=subject_id, high_hz=31, low_hz=14, sample_rate=250)
        print("---------------------{}-----------------------------------------------".format(subject_id))
        subject_data.append([train_set_4, valid_set_4, train_set_8, valid_set_8, train_set_13, valid_set_13])

    #run model
    score = []
    for i in range(9):
        train_set_4, valid_set_4, train_set_8, valid_set_8, train_set_13, valid_set_13 = subject_data[i]
        test_acc = run_9010_HSCNN_model(i+1, train_set_4, valid_set_4, train_set_8, valid_set_8, train_set_13, valid_set_13)
        score.append(test_acc)

    print(score)

def run_HSCNN_individual():

    # #load data
    subject_data =[]
    for subject_id in range(1,10): 
        train_set_4, valid_set_4 = read_data(subject_id=subject_id, high_hz=9, low_hz=4, sample_rate=250)
        train_set_8, valid_set_8 = read_data(subject_id=subject_id, high_hz=15, low_hz=8, sample_rate=250)
        train_set_13, valid_set_13 = read_data(subject_id=subject_id, high_hz=31, low_hz=14, sample_rate=250)
        print("---------------------{}-----------------------------------------------".format(subject_id))
        subject_data.append([train_set_4, valid_set_4, train_set_8, valid_set_8, train_set_13, valid_set_13])

    #run model
    score = []
    for i in range(9):
        train_set_4, valid_set_4, train_set_8, valid_set_8, train_set_13, valid_set_13 = subject_data[i]
        test_acc = run_individual_HSCNN_model(i+1, train_set_4, valid_set_4, train_set_8, valid_set_8, train_set_13, valid_set_13)
        score.append(test_acc)

    print(score)

def run_SCCNet_Subject1_SI():

    
    # load data
    train_X_seq = []
    train_Y_seq = []
    for subject_id in range(1,10): 
        if subject_id == 1:
            train_set, valid_set = read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=125)
            train_X_S1, train_Y_S1, valid_X, valid_Y = get_numpy_data(train_set, valid_set)
        else:
            train_set, valid_set = read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=125)
            train_X, train_Y, valid_X, valid_Y = get_numpy_data(train_set, valid_set)  
            train_X_seq.append(train_X)
            train_X_seq.append(valid_X)
            train_Y_seq.append(train_Y)
            train_Y_seq.append(valid_Y)

    train_X = np.concatenate(train_X_seq)
    train_Y = np.concatenate(train_Y_seq)

    print(train_X.shape, train_Y.shape, valid_X.shape, valid_Y.shape)

    # train scheme
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 5e-4
    weight_decay = 1e-3
    batch_size = 128
    n_epochs = 200
    fine_tune_epochs = 100
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = numpy_data_to_dataloader(train_X, train_Y, valid_X, valid_Y, batch_size)

    model = EEGNet().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_acc_seq = []
    test_acc_seq = []
    for epoch in range(n_epochs):
        train_acc = train(model, optimizer, train_loader, criterion)
        test_acc = test(model, test_loader)
        print(
            "EEGNet Epoch [{}/{}], Training Accuracy: {:.4f}%, Testing Accuracy: {:.4f}%".format(
                epoch + 1, n_epochs, train_acc * 100, test_acc * 100
            )
        )
        train_acc_seq.append(train_acc)
        test_acc_seq.append(test_acc)

    print('Final score: {}'.format(test(model, test_loader)))

    # plot acc
    x = np.arange(n_epochs)
    plt.plot(x, train_acc_seq, label ='train accuracy') 
    plt.plot(x, test_acc_seq, label ='test accuracy') 
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy') 
    plt.title("Subject1 SI Scheme")
    plt.legend() 
    plt.show()

    # plot confusion matrix
    y_predict = test_return_predict(model, test_loader) 
    cf_matrix = confusion_matrix(valid_Y, y_predict.cpu().detach().numpy())

    #print(cf_matrix)

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = []
    for i in range(4):
        group_percentages.append(["{0:.2%}".format(value) for value in cf_matrix[i]/np.sum(cf_matrix[i])])
    group_percentages = np.concatenate(group_percentages)

    #print(group_counts)
    #print(group_percentages)

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(4,4)
    ax = sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='')

    ax.set_title('SI confusion matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    labels = ["Left Hand", "Right Hand", "Both Feet", "Tongue"]
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()

def run_SCCNet_Subject1_SD():

    
    # load data
    train_X_seq = []
    train_Y_seq = []
    for subject_id in range(1,10): 
        if subject_id == 1:
            train_set, valid_set = read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=125)
            train_X_S1, train_Y_S1, valid_X, valid_Y = get_numpy_data(train_set, valid_set)
        else:
            train_set, valid_set = read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=125)
            train_X, train_Y, valid_X, valid_Y = get_numpy_data(train_set, valid_set)  
            train_X_seq.append(train_X)
            train_X_seq.append(valid_X)
            train_Y_seq.append(train_Y)
            train_Y_seq.append(valid_Y)

    train_X_seq.append(train_X_S1)  
    train_Y_seq.append(train_Y_S1)  
    train_X = np.concatenate(train_X_seq)
    train_Y = np.concatenate(train_Y_seq)

    print(train_X.shape, train_Y.shape, valid_X.shape, valid_Y.shape)

    # train scheme
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 5e-4
    weight_decay = 1e-3
    batch_size = 128
    n_epochs = 200
    fine_tune_epochs = 100
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = numpy_data_to_dataloader(train_X, train_Y, valid_X, valid_Y, batch_size)

    model = EEGNet().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_acc_seq = []
    test_acc_seq = []
    for epoch in range(n_epochs):
        train_acc = train(model, optimizer, train_loader, criterion)
        test_acc = test(model, test_loader)
        print(
            "EEGNet Epoch [{}/{}], Training Accuracy: {:.4f}%, Testing Accuracy: {:.4f}%".format(
                epoch + 1, n_epochs, train_acc * 100, test_acc * 100
            )
        )
        train_acc_seq.append(train_acc)
        test_acc_seq.append(test_acc)
  
    print('Final score: {}'.format(test(model, test_loader)))

    # plot acc
    x = np.arange(n_epochs)
    plt.plot(x, train_acc_seq, label ='train accuracy') 
    plt.plot(x, test_acc_seq, label ='test accuracy') 
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy') 
    plt.title("Subject1 SD Scheme")
    plt.legend() 
    plt.show()

    # plot confusion matrix
    y_predict = test_return_predict(model, test_loader) 
    cf_matrix = confusion_matrix(valid_Y, y_predict.cpu().detach().numpy())

    #print(cf_matrix)

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = []
    for i in range(4):
        group_percentages.append(["{0:.2%}".format(value) for value in cf_matrix[i]/np.sum(cf_matrix[i])])
    group_percentages = np.concatenate(group_percentages)

    #print(group_counts)
    #print(group_percentages)

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(4,4)
    ax = sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='')

    ax.set_title('SD confusion matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    labels = ["Left Hand", "Right Hand", "Both Feet", "Tongue"]
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()

def run_SCCNet_Subject1_SIFT():

    
    # load data
    train_X_seq = []
    train_Y_seq = []
    for subject_id in range(1,10): 
        if subject_id == 1:
            train_set, valid_set = read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=125)
            train_X_S1, train_Y_S1, valid_X, valid_Y = get_numpy_data(train_set, valid_set)
        else:
            train_set, valid_set = read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=125)
            train_X, train_Y, valid_X, valid_Y = get_numpy_data(train_set, valid_set)  
            train_X_seq.append(train_X)
            train_X_seq.append(valid_X)
            train_Y_seq.append(train_Y)
            train_Y_seq.append(valid_Y)

    train_X = np.concatenate(train_X_seq)
    train_Y = np.concatenate(train_Y_seq)

    print(train_X.shape, train_Y.shape, valid_X.shape, valid_Y.shape)

    # train scheme
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lr = 5e-4
    weight_decay = 1e-3
    batch_size = 128
    n_epochs = 200
    fine_tune_epochs = 100
    criterion = nn.CrossEntropyLoss()
    train_loader, test_loader = numpy_data_to_dataloader(train_X, train_Y, valid_X, valid_Y, batch_size)

    model = EEGNet().to(device=device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    for epoch in range(n_epochs):
        train_acc = train(model, optimizer, train_loader, criterion)
        test_acc = test(model, test_loader)
        print(
            "EEGNet Epoch [{}/{}], Training Accuracy: {:.4f}%, Testing Accuracy: {:.4f}%".format(
                epoch + 1, n_epochs, train_acc * 100, test_acc * 100
            )
        )
        
   #--------------fine tune-------------------------
    for name, param in model.named_parameters():
        if 'classifier' not in name:
            param.requires_grad = False

    
    print(train_X_S1.shape, train_Y_S1.shape, valid_X.shape, valid_Y.shape)

    train_loader, test_loader = numpy_data_to_dataloader(train_X_S1, train_Y_S1, valid_X, valid_Y, batch_size)

    lr = 1e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    train_acc_seq = []
    test_acc_seq = []
    for epoch in range(fine_tune_epochs):
        train_acc = train(model, optimizer, train_loader, criterion, mode='fine tune1')
        test_acc = test(model, test_loader)
        print(
            "EEGNet FT Epoch [{}/{}], Training Accuracy: {:.4f}%, Testing Accuracy: {:.4f}%".format(
                epoch + 1, n_epochs, train_acc * 100, test_acc * 100
            )
        )
        train_acc_seq.append(train_acc)
        test_acc_seq.append(test_acc)

    #-------------fine tune-------------------------
    print('Final score: {}'.format(test(model, test_loader)))

    # plot acc
    x = np.arange(fine_tune_epochs)
    plt.plot(x, train_acc_seq, label ='train accuracy') 
    plt.plot(x, test_acc_seq, label ='test accuracy') 
    plt.xlabel('Epoch') 
    plt.ylabel('Accuracy') 
    plt.title("Subject1 SI+FT Scheme")
    plt.legend() 
    plt.show()

    # plot confusion matrix
    y_predict = test_return_predict(model, test_loader) 
    cf_matrix = confusion_matrix(valid_Y, y_predict.cpu().detach().numpy())

    #print(cf_matrix)

    group_counts = ["{0:0.0f}".format(value) for value in cf_matrix.flatten()]
    group_percentages = []
    for i in range(4):
        group_percentages.append(["{0:.2%}".format(value) for value in cf_matrix[i]/np.sum(cf_matrix[i])])
    group_percentages = np.concatenate(group_percentages)

    #print(group_counts)
    #print(group_percentages)

    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts,group_percentages)]
    labels = np.asarray(labels).reshape(4,4)
    ax = sns.heatmap(cf_matrix, annot=labels, cmap='Blues', fmt='')

    ax.set_title('SI+FT confusion matrix\n\n')
    ax.set_xlabel('\nPredicted Values')
    ax.set_ylabel('Actual Values ')
    labels = ["Left Hand", "Right Hand", "Both Feet", "Tongue"]
    ax.xaxis.set_ticklabels(labels)
    ax.yaxis.set_ticklabels(labels)
    plt.show()


def unused_plot_topo(n_kernel, spatial_weight):
    true_chs = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    electrode=['Fp1','Fp2','F7','F3','Fz','F4','F8','FT7','FC3',
               'FCz','FC4','FT8','T7','C3','Cz','C4','T8','TP7',
               'CP3','CPz','CP4','TP8','P7','P3','Pz','P4','P8',
               'O1','Oz','O2']
    
    info = mne.create_info(
        ch_names=true_chs,
        ch_types=['eeg']*22,
        sfreq=125
    )
    raw = mne.io.RawArray(spatial_weight[n_kernel,0,:,:].detach().numpy(), info)
    montage = mne.channels.make_standard_montage('standard_1020')
    raw.set_montage(montage)
    grad = np.mean(raw.get_data(), 1)
    mne.viz.plot_topomap(grad, raw.info, show=True)


def unused_plot_multi_topo(spatial_weight):
    true_chs = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']
    info = mne.create_info(
        ch_names=true_chs,
        ch_types=['eeg']*22,
        sfreq=125
    )
    montage = mne.channels.make_standard_montage('standard_1020')

    # Grad have shape (n_channels, n_times)    
    fig, ax = plt.subplots(4)
    for i in range(4):
        raw = mne.io.RawArray(spatial_weight[i,0,:,:].detach().numpy(), info)
        raw.set_montage(montage)
        grad = np.mean(raw.get_data(), 1)
        mne.viz.plot_topomap(grad, raw.info, show=False, axes=ax[i])


def Spatial_kernel(topomap, titles):
    """

    :type topomap: list or ndarray
    """
    #conv1,_= get_weight(filepath)
    electrode = ['Fz', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'P1', 'Pz', 'P2', 'POz']

    montage= mne.channels.make_standard_montage('standard_1020')
    montage= montage.get_positions()['ch_pos']

    position= [montage[i] for i in electrode]
    position= np.asarray(position)
    #print(position) 22 * [x y z]
    fig, ax = plt.subplots(2,11 , figsize=(20, 5))
    
    for a in range(len(titles)):
        ax[int(a/11),int(a%11)].set_title(titles[a],fontsize = 14)
    
    for i in range(len(topomap)):
        im,cm  = mne.viz.plot_topomap(data=topomap[i],
                                pos=position[:,0:2],
                                axes= ax[int(i/11), int(i%11)],
                                show= True if i==21 else False,
                            )

    clb = fig.colorbar(im, cax=ax[-1,-1])
    clb.ax.set_title("unit_label",fontsize=14) # title on top of colorbar


def plot_SCCNet_topo():
    model = SCCNet()
    model.load_state_dict(torch.load('SCCNet/SCCNet_S1_Iteration_0'), False)
    for name, param in model.named_parameters():
        if name == 'spatial.0.weight':
            spatial_weight = param.detach().cpu().numpy()

   
    topomap = []
    for i in range(22):
        topomap.append(spatial_weight[i,0,:,0])

    titles = ['Kernel:'+ str(i+1) for i in range(22)]

    Spatial_kernel(topomap, titles)







if __name__ == '__main__':

    # #load data
    # subject_data =[]
    # for subject_id in range(1,10): 
    #     subject_data.append(read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=250))

    # #run model
    # for i in range(9):
    #     train_set, valid_set = subject_data[i]
    #     for n_iter in range(10):
    #         run_SCCNet_model(subject_id= i+1,n_iter=n_iter ,train_set=train_set, valid_set=valid_set)
    
    # score = []
    # for status in [True, False]:
    #     score.append(run_augment_playground(try_aug=status))
    # print(score)

    #run_SCCNet_Subject1_SIFT()

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # model = HSCNN(4)
    # print(model(torch.randn((16,3,1,22,875))).shape)
    # for name, param in model.named_parameters():
    #     print(name, param, type(name), type(param))

    #run_SCCNet(try_aug=False)

    #BCI Course Part1 earning rate = 0.001, batch size = 32

   
    run_TCN_Fusion()