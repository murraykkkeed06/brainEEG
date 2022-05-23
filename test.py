from statistics import mode
from braindecode.preprocessing import (exponential_moving_standardize, preprocess, Preprocessor, scale)
from braindecode.models import ShallowFBCSPNet, EEGNetv1, EEGNetv4, Deep4Net
from ModelFamily import SCCNet, HSCNN, SCCTransformer, EEGNet, EEGNet_TCN, TCN
from braindecode.preprocessing import create_windows_from_events
from braindecode.util import set_random_seeds
from braindecode.datasets import MOABBDataset
from skorch.helper import predefined_split
from skorch.callbacks import LRScheduler
from braindecode import EEGClassifier
import numpy as np
import argparse
import pickle
import torch
import csv

def load_data(subject_id):
    
    dataset = MOABBDataset(dataset_name="BNCI2014001", subject_ids=[subject_id])

    low_cut_hz = 4.  # low cut frequency for filtering
    high_cut_hz = 38.  # high cut frequency for filtering
    # Parameters for exponential moving standardization
    factor_new = 1e-3
    init_block_size = 1000

    preprocessors = [
        Preprocessor('pick_types', eeg=True, meg=False, stim=False),  # Keep EEG sensors
        Preprocessor(scale, factor=1e6, apply_on_array=True),  # Convert from V to uV
        Preprocessor('filter', l_freq=low_cut_hz, h_freq=high_cut_hz),  # Bandpass filter
        Preprocessor(exponential_moving_standardize,  # Exponential moving standardization
                    factor_new=factor_new, init_block_size=init_block_size)
    ]

    # Transform the data
    preprocess(dataset, preprocessors)

    trial_start_offset_seconds = -0.5
    # Extract sampling frequency, check that they are same in all datasets
    sfreq = dataset.datasets[0].raw.info['sfreq']
    assert all([ds.raw.info['sfreq'] == sfreq for ds in dataset.datasets])
    # Calculate the trial start offset in samples.
    trial_start_offset_samples = int(trial_start_offset_seconds * sfreq)

    windows_dataset = create_windows_from_events(
        dataset,
        trial_start_offset_samples=trial_start_offset_samples,
        trial_stop_offset_samples=0,
        preload=True,
    )
    splitted = windows_dataset.split('session')
    valid_set = splitted['session_E']


    valid_X = []
    for i in range(288):
        valid_X.append(valid_set[i][0])
    valid_X = np.stack(valid_X, axis=0)

    valid_Y = []
    for i in range(288):
        valid_Y.append(np.array(valid_set[i][1]))
    valid_Y = np.stack(valid_Y, axis=0)

    return valid_X, valid_Y



if __name__ == '__main__':


    model = EEGNet_TCN
    folder = f'{model.__name__}'

    # load data
    subject_data = []
    for subject_id in range(1,10):
        subject_data.append(load_data(subject_id))

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
            valid_X, valid_Y = subject_data[i]
            loaded_model = pickle.load(open(f"./{folder}/{model.__name__}_S{i+1}_Iter{n_iter}.sav", 'rb'))
            score_arr.append(loaded_model.score(valid_X, valid_Y)) 
    
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
        valid_X, valid_Y = subject_data[i]
        for n_iter in range(10):
            loaded_model = pickle.load(open(f"./{folder}/{model.__name__}_S{i+1}_Iter{n_iter}.sav", 'rb'))
            score_arr.append(loaded_model.score(valid_X, valid_Y)) 
        
        with open(filename, "a", newline="", encoding="utf-8-sig'") as f:
            writer = csv.writer(f)
            writer.writerow([i+1,sum(score_arr)/10])


    