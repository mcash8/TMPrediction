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

class ProphetAngleMetric(nn.modules.loss._Loss):
    ''' 
        Loss function adopted from: 
        @ARTICLE{10250443,
        author={Zhang, Yuntian and Han, Ning and Zhu, Tengteng and Zhang, Junjie and Ye, Minghao and Dou, Songshi and Guo, Zehua},
        journal={IEEE/ACM Transactions on Networking}, 
        title={Prophet: Traffic Engineering-Centric Traffic Matrix Prediction}, 
        year={2024},
        volume={32},
        number={1},
        pages={822-832},
        keywords={Predictive models;Routing;Training;Simulation;Recurrent neural networks;Optimization;IEEE transactions;Traffic Engineering (TE);Traffic Matrix (TM) Prediction;Wide Area Networks (WANs)},
        doi={10.1109/TNET.2023.3293098}}
    '''
    def __init__(self):
        super(ProphetAngleMetric, self).__init__()

    def forward(self, pred_v, target_v):

        a = torch.ones((pred_v.shape[0]))

        angle_loss = a - torch.diag(torch.mm(pred_v, target_v.t())) / (
                torch.mul(torch.norm(pred_v, dim=1), torch.norm(target_v, dim=1)) + 1e-11)  # eps, 1e-11
        angle_loss = torch.sum(angle_loss)

        angle_loss /= pred_v.shape[0]
        
        return angle_loss


def main():

    read_week = True
    min_max_norm = True
    feature_norm = False

    # Read Abilene Dataset
    if read_week:
        data = read_abilene_data(read_week = True, week = 1)
    
    else: 
        data = read_abilene_data(read_week = False)


    # Train Test Split 
    train_data, test_data = train_test_split(data, 0.8)

    if min_max_norm:
        # Min Max Normalize the train data and test data
        scaler = MinMaxScaler(feature_range=(0, 1))
        train_data_norm = normalize_matrix(scaler, train_data)
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

    # Min Max Normalize the train data and test data
    scaler = MinMaxScaler(feature_range=(0, 1))
    train_data_norm = normalize_matrix(scaler, train_data)
    test_data_norm = normalize_matrix(scaler, test_data)

    # Window the dataset 
    trainX, trainY= create_dataset(train_data_norm, 10) 
    testX, testY = create_dataset(test_data_norm, 10) 

    # Define hyperparameters
    num_nodes = 12 
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
    criterion = ProphetAngleMetric()

    # Train the model
    loss = train(model, train_loader, epochs, criterion, optimizer)

    # Plot the training loss
    plot_and_save_training_loss(loss)

    # Test model
    model_outputs, _ = test(model, criterion, test_loader, testY)

    # Save the model 
    PATH = 'models\\abilene_global_cosine.pth'
    torch.save(model.state_dict(), PATH)

    # Save the model_outputs
    np.save('model_outputs\\abilene_global_cosine_outputs.npy', 
            model_outputs)

    # Plot heat maps 
    plot_and_save_heatmap(test_data_norm, model_outputs, num_nodes, save_path = 'Figs\\heat_maps', 
                          fig_name = 'normalized_model_predictions_cosine.png')


    if min_max_norm:
    # Inverse normalize model outputs
        inverse_preds = inverse_normalize_predictions(test_data, model_outputs) 
    
    if feature_norm:
        inverse_preds = model_outputs * (max_vals_test - min_vals_test) + min_vals_test

    plot_and_save_heatmap(test_data, inverse_preds, num_nodes, save_path = 'Figs\\heat_maps',
                          fig_name= 'inverse_normalized_model_predictions_cosine.png')
    
    
    # Compute MCF on inverse normalized predictions
    mlu_preds, Nans = mlu_on_preds(inverse_preds)

    if Nans: 
        print('NaN values in mlu_preds, ending program')
        return

    # Save MLUs
    np.save('model_outputs\\mlu_preds_abilene_global_cosine.npy', mlu_preds)

    # Load MCF baseline on original dataset
    mlu_gt = np.load('mlu_baseline\\mlu_baseline_abilene.npy')
    mlu_gt = mlu_gt[len(train_data):]
    
    # Plot mlu cdf and mlu comparison
    plot_and_save_ecdf(mlu_gt, mlu_preds, save_path = 'Figs\\ecdfs', fig_name='CDF_mlu_abilene_global_cosine.png')
    plot_and_save_mlu_compare(mlu_gt, mlu_preds, save_path='Figs\\mlu_compare', fig_name='mlu_compare_abilene_global_cosine.png')
    plot_and_save_pdf(mlu_gt, mlu_preds, save_path = 'Figs\\pdfs', fig_name='PDF_mlu_abilene_global_cosine.png')
    
    return


if __name__ == '__main__': 
    main()