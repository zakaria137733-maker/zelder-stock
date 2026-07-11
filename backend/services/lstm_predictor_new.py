import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score
import json

MODEL_DIR = '/app/models'
MODEL_PATH = f'{MODEL_DIR}/lstm_model.pt'
SCALER_PATH = f'{MODEL_DIR}/scaler.json'

SEQUENCE_LEN = 6
FEATURES = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.3
EPOCHS = 60
LR = 0.001
