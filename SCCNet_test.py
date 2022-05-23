from braindecode.preprocessing import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.models import ShallowFBCSPNet, EEGNetv1, EEGNetv4, Deep4Net
from braindecode.preprocessing import create_windows_from_events
from torch.utils.data import TensorDataset, DataLoader
from braindecode.util import set_random_seeds
from braindecode.datasets import MOABBDataset
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from braindecode import EEGClassifier
from ModelFamily import SCCNet, SCCTransformer, EEGNet_TCN
from torch import Tensor
from torch import nn
import numpy as np
import argparse
import pickle
import torch
import csv

def read_data(subject_id, high_hz, low_hz, sample_rate):
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

    
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
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

def to_dataloader(train_set, valid_set, batch_size):

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

    train_X = train_X.reshape(-1,1,22,1125)
    valid_X = valid_X.reshape(-1,1,22,1125) 

    train_set_tensor = Tensor(train_X)
    train_label_tensor = Tensor(train_Y).type(torch.LongTensor)

    train_dataset = TensorDataset(train_set_tensor, train_label_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size)

    test_set_tensor = Tensor(valid_X)
    test_label_tensor = Tensor(valid_Y).type(torch.LongTensor)

    test_dataset = TensorDataset(test_set_tensor, test_label_tensor)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, test_loader

def test(model, data_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total

device = 'cuda' if torch.cuda.is_available() else 'cpu'


subject_data =[]
for subject_id in range(1,10): 
    subject_data.append(read_data(subject_id=subject_id, high_hz=38, low_hz=0.5, sample_rate=250))


test_loader_data = []
for i in range(9):
    _ , test_data = to_dataloader(subject_data[i][0], subject_data[i][1],batch_size=64)
    test_loader_data.append(test_data)

model = EEGNet_TCN().to(device)
folder = f'{model.__name__}'

# File Export
filename = "Iteration" + model.__name__ + ".csv"
with open(filename, "a", newline="", encoding="utf-8-sig'") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Iteration",
            "Accuracy",
        ]
    )

#iteration score
for n_iter in range(10):
    score_arr = []
    for i in range(9):
        model.load_state_dict(torch.load(f'EEGNet_TCN/EEGNet_TCN_S{i+1}_Iteration_{n_iter}'))
        test_acc = test(model, test_loader_data[i])
        score_arr.append(test_acc) 

    with open(filename, "a", newline="", encoding="utf-8-sig'") as f:
        writer = csv.writer(f)
        writer.writerow([n_iter,sum(score_arr)/9])


# File Export
filename = "Subject" + model.__name__ + ".csv"
with open(filename, "a", newline="", encoding="utf-8-sig'") as f:
    writer = csv.writer(f)
    writer.writerow(
        [
            "Subject",
            "Accuracy",
        ]
    )

#subject_id score
for i in range(9):
    score_arr = []
    for n_iter in range(10):
        model.load_state_dict(torch.load(f'EEGNet_TCN/EEGNet_TCN_S{i+1}_Iteration_{n_iter}'))
        test_acc = test(model, test_loader_data[i])
        score_arr.append(test_acc) 
    
    with open(filename, "a", newline="", encoding="utf-8-sig'") as f:
        writer = csv.writer(f)
        writer.writerow([i+1,sum(score_arr)/10])





