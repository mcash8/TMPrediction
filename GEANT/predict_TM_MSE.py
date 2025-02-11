import numpy as np
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler

import torch
from torch.autograd import *
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data

from utils import * 
import sys


def main():

    read_week = False
    min_max_norm = True
    feature_norm = False

    # Read Abilene Dataset
    if read_week:
        data = read_geant_data(read_week = True, week = 1)
    
    else: 
        data = read_geant_data(read_week = False)
        start_idx = 0 
        end_idx = int((60/15)*24*21) # three weeks of data (same number of samples as abilene for 1 week)

        data = data[start_idx:end_idx]


    # Train Test Split 
    train_data, test_data = train_test_split(data, 0.8)
    window = 10
    
    if min_max_norm:
        # Min Max Normalize the train data and test data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data_norm = normalize_matrix(scaler, train_data)

        scaler = MinMaxScaler(feature_range=(0,1))
        test_data_norm = normalize_matrix(scaler, test_data)
    
    if feature_norm:
        # Feature-wise min-max normalization (normalize each element over time)
        min_vals_train = train_data.min(axis=0)  # Shape (144,)
        max_vals_train = train_data.max(axis=0)  # Shape (144,)
        train_data_norm = (train_data - min_vals_train) / (max_vals_train - min_vals_train + 1e-8)  # Avoid division by zero

        # Feature-wise min-max normalization (normalize each element over time)
        min_vals_test = test_data.min(axis=0)  # Shape (144,)
        max_vals_test = test_data.max(axis=0)  # Shape (144,)
        test_data_norm = (test_data - min_vals_test) / (max_vals_test - min_vals_test + 1e-8)  # Avoid division by zero

    # Window the dataset 
    trainX, trainY= create_dataset(train_data_norm, window) 
    testX, testY = create_dataset(test_data_norm, window) 

    # Define hyperparameters
    num_nodes = 23
    input_size = trainX.shape[2] # Number of features in input
    hidden_size = 200  # Number of features in hidden state
    learn_rate = 0.001 
    epochs = 100
    num_layers = 1
    batch_size = 32
    shuffle = False #don't want to lose the time dependency
    num_workers = 4  # Number of subprocesses to use for data loading

    # Create training dataset and dataloader
    train_loader = get_dataloader(trainX, trainY, batch_size, num_workers, shuffle)
    test_loader = get_dataloader(testX, testY, 1, num_workers, shuffle)

    # Train RNN 
    # Create Model 
    model = RNN(input_size, hidden_size, num_layers)

    # Create optimizer 
    optimizer = optim.Adam(model.parameters(), lr=learn_rate)

    # Create loss function
    criterion = nn.MSELoss()

    # Train the model
    loss = train(model, train_loader, epochs, criterion, optimizer)

    # Plot the training loss
    plot_and_save_training_loss(loss)

    # Test model
    model_outputs, _ = test(model, criterion, test_loader, testY)

    # Save the model 
    PATH = 'models\\geant_global_mse.pth'
    torch.save(model.state_dict(), PATH)

    # Save the model_outputs
    np.save('model_outputs\\geant_global_mse.npy', 
            model_outputs)

    # Plot heat maps 
    plot_and_save_heatmap(test_data_norm, model_outputs, num_nodes, save_path = 'Figs\\heat_maps', 
                          fig_name = 'normalized_model_predictions.png')

    if min_max_norm:
    # Inverse normalize model outputs
        inverse_preds = inverse_normalize_predictions(test_data, model_outputs) 
    
    if feature_norm:
        inverse_preds = model_outputs * (max_vals_test - min_vals_test) + min_vals_test

    plot_and_save_heatmap(test_data, inverse_preds, num_nodes, save_path = 'Figs\\heat_maps',
                          fig_name= 'inverse_normalized_model_predictions.png')
    
    
    # Compute MCF on inverse normalized predictions
    mlu_preds, Nans = mlu_on_preds(inverse_preds, num_nodes=23, capacity=1e7, topo='fc')

    if Nans: 
        print('NaN values in mlu_preds, ending program')
        return
    
    # Save MLUs
    np.save('mlu_baseline\\mlu_preds_geant_global_mse.npy', mlu_preds)

    # Load MCF baseline on original dataset
    mlu_gt = np.load('mlu_baseline\\mlu_baseline_geant_fc.npy')
    mlu_gt = mlu_gt[len(train_data):]
    
    # Plot mlu cdf and mlu comparison
    plot_and_save_ecdf(mlu_gt, mlu_preds, save_path = 'Figs\\ecdfs', fig_name='CDF_mlu_geant_global_mse.png')
    plot_and_save_mlu_compare(mlu_gt, mlu_preds, save_path='Figs\\mlu_compare', fig_name='mlu_compare_geant_global_mse.png')
    plot_and_save_pdf(mlu_gt, mlu_preds, save_path = 'Figs\\pdfs', fig_name='PDF_mlu_geant_global_mse.png')

    return


if __name__ == '__main__': 
    main()