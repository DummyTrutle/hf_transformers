
# import numpy as np
# import pandas as pd
import torch
import torch.nn as nn
# import torch.optim as optim
# import torch.utils.data as data

class nextSPAT(nn.Module):
	def __init__(self):
		super().__init__()
		self.lstm = nn.LSTM(input_size = 6,
							hidden_size = 200,
							num_layers = 3,
							batch_first = True)
		self.linear = nn.Linear(200, 5)

	def forward(self, x):
		x, _ = self.lstm(x)
		x = self.linear(x)
		return x

# import torch
# from modules import nextSPAT
import pandas as pd

def create_artificial_data(dataset_filename="artificial_data.csv", lookback=6, split=0.8):
    ds = pd.read_csv(dataset_filename)
    time_stamp = ds[["timeStamp"]].values.astype("int16")
    light_status = ds[["lightStatus"]].values.astype("int16")
    likely_end_time = ds[["likelyEndTime"]].values.astype("int16")
    min_end_time = ds[["minEndTime"]].values.astype("int16")
    next_duration = ds[["nextDuration"]].values.astype("int16")
    
    train_size = int(len(time_stamp) * split)
    test_size = len(time_stamp) - train_size
    features = torch.cat((torch.tensor(time_stamp).t(), 
                          torch.tensor(light_status).t(), 
                          torch.tensor(likely_end_time).t(), 
                          torch.tensor(min_end_time).t(), 
                          torch.tensor(next_duration).t()), 0)
    features = features.t()

    t_X, t_y = torch.zeros(lookback, 5), torch.zeros(lookback, 5)
    train, test = features[: train_size], features[train_size: ]
    X, y = torch.zeros(1, lookback, 5), torch.zeros(1, lookback, 5)
    for i in range(len(features) - lookback):
        t_X[0] = features[i].clone()
        t_y[0] = features[i+1].clone()
        for t in range(i+1, i+lookback):
            t_X[t-i] = features[t].clone()
            t_y[t-i] = features[t+1].clone()
        X = torch.cat((X, t_X.clone().unsqueeze(0)), 0)
        y = torch.cat((y, t_y.clone().unsqueeze(0)), 0)
    return (torch.tensor(X), torch.tensor(y))

# X, y = create_dataset("artificial_data.csv", 6)