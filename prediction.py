
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from modules import create_artificial_data, nextSPAT


artificial_data = create_artificial_data()
X = artificial_data[0]
y = artificial_data[1]

dataset_size = len(artificial_data)
train_size = int(dataset_size * 0.75)

X_train = X[:train_size]
y_train = y[:train_size]
X_test = X[train_size:]
y_test = y[train_size:]

model = nextSPAT()
optimizer = optim.Adam(model.parameters())
loss_fn = nn.MSELoss()
loader = data.DataLoader(data.TensorDataset(X_train, y_train),
											shuffle=True,
											batch_size=8)

n_epochs = 200
for epoch in range(n_epochs):
	model.train()
	for X_batch, y_batch in loader:
		y_pred = model(X_batch) 	
		# print("\ny_pred shape: ", y_pred.shape, "y_train shape: ", y_train.shape, "\n")
		loss = loss_fn(y_pred, y_batch)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
	if epoch % 100 != 0:
		continue
	model.eval()
	with torch.no_grad():
		y_pred = model(X_train)
		train_rmse = np.sqrt(loss_fn(y_pred, y_train))
		y_pred = model(X_test)
		test_rmse = np.sqrt(loss_fn(y_pred, y_test))
	print("Epoch %d: train RMSE %.4f, test RMSE %.4f" % (epoch, train_rmse, test_rmse))

with torch.no_grad():
	train_plot = np.ones_like(artificial_data) * np.nan
	y_pred = model(X_train)
	print(y_pred)
	# y_pred = y_pred(:, -1, :)