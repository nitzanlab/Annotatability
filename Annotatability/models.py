import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scanpy as sc
from sklearn.preprocessing import normalize
from sklearn.metrics import roc_auc_score , pairwise_distances
import torch
import scipy.sparse as sp
from numba import jit
from numpy import array
from numpy import argmax
from torch.utils.data import TensorDataset, DataLoader , WeightedRandomSampler
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import scipy.sparse as sp
import random

class Net(nn.Module):
    def __init__(self, input_size, output_size):
        """
        Initializes a feedforward neural network with three fully-connected layers.

        Args:
            input_size (int): Size of the input layer.
            output_size (int): Size of the output layer.
        """
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, int(input_size / 2))
        self.fc2 = nn.Linear(int(input_size / 2), int(input_size / 4))
        self.fc3 = nn.Linear(int(input_size / 4), output_size)

    def forward(self, x):
        """
        Performs the forward pass of the neural network.

        Args:
            x (torch.Tensor): Input data of shape (batch_size, input_size).

        Returns:
            torch.Tensor: Output data of shape (batch_size, output_size).
        """
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x)
        return output

def modify_labels(orig_labeles, probability=0.1):
    # Get the list of all possible labels
    all_labels = list(set(orig_labeles))

    # Create a list to store the modified labels
    modified_labels = []

    # Loop over the labels in the dataset
    for label in orig_labeles:
        # Generate a random number between 0 and 1
        r = random.random()

        # If the random number is smaller than the probability,
        # modify the label by choosing a random label from the list of all labels,
        # excluding the original label
        if r < probability:
            modified_label = random.choice([l for l in all_labels if l != label])
        else:
            modified_label = label

        # Add the modified label to the list
        modified_labels.append(modified_label)

    # Create a DataFrame with the original and modified labels
    df = pd.DataFrame({"label": orig_labeles, "modified_label": modified_labels})


    return df , (np.where(df["label"] != df["modified_label"]))

def one_hot_encode(labels):
    """
    One-hot encodes an array of labels.

    Args:
        labels (numpy.ndarray): Array of labels.

    Returns:
        tuple: One-hot encoded array of labels, and a label encoder.
    """
    values = np.array(labels)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    # binary encode
    onehot_encoder = OneHotEncoder()
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    # invert first example
    #inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    if is_scipy_cs_sparse(onehot_encoded):
        onehot_encoded = onehot_encoded.toarray()
    return onehot_encoded, label_encoder

def one_hot_encode_two_labels(label1, label2):
    """
    One-hot encodes an array of labels.

    Args:
        labels (numpy.ndarray): Array of labels.

    Returns:
        tuple: One-hot encoded array of labels, and a label encoder.
    """
    values = np.array(label1)
    values2 = np.array(label2)

    # integer encode
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)
    integer_encoded2 = label_encoder.fit_transform(values2)
    # binary encode
    onehot_encoder = OneHotEncoder()
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    integer_encoded2 = integer_encoded2.reshape(len(integer_encoded2), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    onehot_encoded2 = onehot_encoder.fit_transform(integer_encoded2)
    # invert first example
    #inverted = label_encoder.inverse_transform([np.argmax(onehot_encoded[0, :])])
    if is_scipy_cs_sparse(onehot_encoded):
        onehot_encoded = onehot_encoded.toarray()
    if is_scipy_cs_sparse(onehot_encoded2):
        onehot_encoded2 = onehot_encoded2.toarray()

    return onehot_encoded, onehot_encoded2, label_encoder

def is_scipy_cs_sparse(matrix):
    """
    Check if a matrix is a Compressed Sparse Row (CSR) matrix using scipy.

    Parameters:
    ----------
    matrix : scipy.sparse.spmatrix
        The input matrix to be checked.

    Returns:
    -------
    bool
        True if the input matrix is a CSR matrix, False otherwise.

    """
    return sp.issparse(matrix) and matrix.getformat() == 'csr'


def follow_training_dyn_neural_net(adata, label_key, iterNum=100, lr=0.001, momentum=0.9,
                    device='cpu', weighted_sampler=True, batch_size=256):
    """
    Initialize and train a neural network on single-cell RNA sequencing data.

    This function initializes a neural network, prepares the dataset, and trains
    the network using stochastic gradient descent.

    Parameters:
    ----------
    adata : AnnData
        Anndata object containing the single-cell RNA sequencing data.

    label_key : str
        The key in adata.obs where the cell labels are stored.

    iterNum : int, optional (default=100)
        Number of training iterations (epochs).

    lr : float, optional (default=0.001)
        Learning rate for the optimizer.

    momentum : float, optional (default=0.9)
        Momentum for the optimizer.

    device : str, optional (default='cpu')
        Device for training the neural network ('cpu' or 'cuda' for GPU).

    weighted_sampler : bool, optional (default=True)
        Whether to use a weighted sampler for class imbalance.

    batch_size : int, optional (default=256)
        Batch size for training.

    Returns:
    -------
    list
        A list of confidence probability losses during training.

    Notes:
    ------
    - This function assumes that you have defined the neural network architecture
      in a separate module as 'Net'.
    - 'one_hot_encode' should be a function that encodes labels as one-hot vectors.
    - 'create_weighted_sampler' should be a function that creates a weighted sampler
      for handling class imbalance.

    - Ensure that the necessary PyTorch and scikit-learn packages are installed.
    """
    one_hot_label, inverted_label = one_hot_encode(adata.obs[label_key])
    net = Net(adata.X.shape[1], output_size=len(adata.obs[label_key].unique()))

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    if is_scipy_cs_sparse(adata.X):
        x_data = adata.X.toarray()
    else:
        x_data = np.array(adata.X)
    tensor_x = torch.Tensor(x_data)  # Transform to torch tensor
    tensor_y = torch.Tensor(one_hot_label)
    tensor_x = tensor_x.to(device)
    tensor_y = tensor_y.to(device)
    my_dataset = TensorDataset(tensor_x, tensor_y)  # Create your dataset

    if weighted_sampler:
        sampler = create_weighted_sampler(adata, label_key)
        trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                  sampler=sampler, num_workers=0)
    else:
        trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
    prob_loss_list = []
    for epoch in range(iterNum):  # Loop over the dataset multiple times
        outputs_all = net(tensor_x)
        prob_all = probability_for_confidence(outputs_all, tensor_y)
        prob_loss_list.append((prob_all.cpu().detach().numpy()))

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if device.type != 'cpu':
                #print('Running with cuda')
                inputs, labels = inputs.cuda(), labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 1:  # Print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
    return prob_loss_list

import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler
from anndata import AnnData

def create_weighted_sampler(adata: AnnData, label_key: str):
    """
    Create a weighted sampler to handle class imbalance for a given label in an AnnData object.

    This function computes class weights based on the distribution of labels and creates
    a weighted sampler for training data to address class imbalance issues.

    Parameters:
    ----------
    adata : AnnData
        Anndata object containing the data.

    label_key : str
        The key in adata.obs where the class labels are stored.

    Returns:
    -------
    WeightedRandomSampler
        A weighted sampler that can be used with DataLoader to handle class imbalance.

    Notes:
    ------
    - Ensure that the AnnData object ('adata') contains a column with class labels specified
      by the 'label_key' parameter.
    - The returned WeightedRandomSampler can be used with PyTorch DataLoader to sample
      batches with class-awareness.
    """
    y_train = adata.obs[label_key]
    y_train_indices = y_train.unique()
    class_sample_count = np.array([sum(y_train == t) for t in y_train_indices])
    weight = 1.0 / class_sample_count
    samples_weight = np.zeros(adata.n_obs)
    for i in range(adata.n_obs):
        for j, t in enumerate(y_train_indices):
            if adata.obs[label_key][i] == t:
                samples_weight[i] = weight[j]
    samples_weight = torch.from_numpy(samples_weight)
    sampler = WeightedRandomSampler(samples_weight.type('torch.DoubleTensor'), len(samples_weight))
    return sampler


def probability_for_confidence(y_pred, y_true):
    """
    Calculates the probability that the predicted class is correct, given the predicted class probabilities and the true class labels.

    Args:
        y_pred (torch.Tensor): Tensor of shape (batch_size, num_classes) with predicted class probabilities.
        y_true (torch.Tensor): Tensor of shape (batch_size, num_classes) with true class labels.

    Returns:
        torch.Tensor: Tensor of shape (batch_size,) with the probability that the predicted class is correct for each sample.
    """
    prob = torch.exp(y_pred)
    return torch.sum(prob * y_true, axis=1)


def probability_list_to_confidence_and_var(prob_list, n_obs, epoch_num):
    """
    Calculates the confidence and variability of the predicted class probabilities, given a list of predicted class probabilities.

    Args:
        prob_list (list): List of tensors of shape (batch_size,) with predicted class probabilities.
        n_obs (int): Number of observations.
        epoch_num (int): Number of epochs.

    Returns:
        tuple: Tensor of shape (batch_size,) with the confidence of the predicted class probabilities, and a tensor of shape (batch_size,) with the variability of the predicted class probabilities.
    """
    confidence = torch.zeros(n_obs)
    for i in range(epoch_num):
        confidence += (prob_list[i])
    confidence = confidence / epoch_num
    variability = torch.zeros(n_obs)
    for i in range(epoch_num):
        variability += torch.square(confidence - (prob_list[i]))
    variability = variability / epoch_num
    variability = torch.sqrt(variability)
    return confidence, variability


def find_cutoff_paramter(adata, label, device='cpu', probability=0.05, percentile=90, epoch_num=50, lr=0.001, weighted_sampler=True, batch_size=256):
    """
    Find cutoff parameters based on modified labels and confidence scores.

    Parameters
    ----------
    adata : AnnData
        Anndata object containing the data and annotations.
    label : str
        The label used for modifying labels and training the neural network.
    device : str, optional
        Device to run the neural network on (e.g., 'cpu' or 'cuda'), by default 'cpu'.
    probability : float, optional
        Probability of label modification, by default 0.05.
    percentile : int, optional
        Percentile for determining cutoff parameters, by default 90.
    epoch_num : int, optional
        Number of epochs for training, by default 50.
    sampler : torch.utils.data.sampler.Sampler, optional
        Sampler for training data, by default None.

    Returns
    -------
    float
        Cutoff parameter based on confidence scores.
    float
        Cutoff parameter based on variance of confidence scores.
    """
    labels = adata.obs[label]
    df, were_changed = modify_labels(labels, probability=probability)
    one_hot_modified_label, inverted_modified = one_hot_encode(df['modified_label'])
    adata.obs['modified_label']=df['modified_label']
    net = Net(adata.X.shape[1], output_size=len(labels.unique()))
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.9)
    
    if is_scipy_cs_sparse(adata.X):
        x_data = adata.X.toarray()
    else:
        x_data = np.array(adata.X)
    
    tensor_y_noisy = torch.Tensor(one_hot_modified_label)

    tensor_x = torch.Tensor(x_data)  # Transform to torch tensor
    #tensor_y = torch.Tensor(one_hot_label)
    tensor_x = tensor_x.to(device)
    tensor_y_noisy = tensor_y_noisy.to(device)
    my_dataset = TensorDataset(tensor_x, tensor_y_noisy)  # Create your dataset



    if weighted_sampler:
        sampler = create_weighted_sampler(adata, 'modified_label')
        trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                  sampler=sampler, num_workers=0)
    else:
        trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
    
    prob_loss_list = train_include_noise(net=net, device=device, trainloader=trainloader, criterion=criterion,
                                         optimizer=optimizer, tensor_x=tensor_x,
                                         tensor_y_noisy=tensor_y_noisy, epoch_num=epoch_num)
    
    all_conf, all_var = probability_list_to_confidence_and_var(prob_loss_list, n_obs=adata.n_obs, epoch_num=epoch_num)
    
    return np.percentile(all_conf[were_changed], percentile), np.percentile(all_var[were_changed], percentile)



def train_include_noise(net, device, trainloader, criterion, optimizer, tensor_x, tensor_y_noisy, epoch_num=60):
    """
    Train a neural network with noisy labels and collect confidence probability losses.

    This function trains a neural network using a DataLoader with noisy labels and
    collects confidence probability losses during training.

    Parameters:
    ----------
    net : torch.nn.Module
        The neural network model to be trained.

    device : str
        Device for training the neural network ('cpu' or 'cuda' for GPU).

    trainloader : DataLoader
        DataLoader containing training data with noisy labels.

    criterion : torch.nn.Module
        The loss function (criterion) to be minimized during training.

    optimizer : torch.optim.Optimizer
        The optimizer to update the neural network parameters.

    tensor_x : torch.Tensor
        The input data tensor.

    tensor_y_noisy : torch.Tensor
        The noisy label tensor corresponding to the input data.

    epoch_num : int, optional (default=60)
        Number of training epochs.

    Returns:
    -------
    list
        A list of confidence probability losses during training.


    Notes:
    ------
    - Ensure that you have a neural network model ('net'), DataLoader ('trainloader'),
      loss function ('criterion'), and optimizer ('optimizer') defined and initialized
      before using this function.
    - 'tensor_x' should be a torch.Tensor containing the input data.
    - 'tensor_y_noisy' should be a torch.Tensor containing the noisy labels corresponding
      to the input data.
    """
    prob_loss_list = []
    for epoch in range(epoch_num):
        outputs_all = net(tensor_x)
        prob_all = probability_for_confidence(outputs_all, tensor_y_noisy)
        prob_loss_list.append((prob_all.cpu().detach().numpy()))

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            if device.type != 'cpu':
                inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 1:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0

    return prob_loss_list


def predict_true_labels(adata, label, new_annotation='CorrectedCellType', device='cpu',
                         high_conf_rank=0.2, epoch_num=100, lr=0.001, momentum=0.9,
                         batch_size=256):
    """
    Predict true labels using a neural network and update the annotation in the AnnData object.

    Parameters
    ----------
    adata : AnnData
        Anndata object containing the data and annotations.
    label : str
        The label used for training the neural network.
    new_annotation : str, optional
        The new annotation key to store the predicted labels, by default 'CorrectedCellType'.
    device : str, optional
        Device to run the neural network on (e.g., 'cpu' or 'cuda'), by default 'cpu'.
    high_conf_rank : float, optional
        High confidence rank threshold, by default 0.2.
    epoch_num : int, optional
        Number of epochs for training, by default 100.
    lr : float, optional
        Learning rate for training the neural network, by default 0.001.
    momentum : float, optional
        Momentum for the SGD optimizer, by default 0.9.
    batch_size : int, optional
        Batch size for training, by default 256.

    Returns
    -------
    AnnData
        Anndata object with updated annotations based on predicted labels.
    """
    bdata = adata.copy()
    cdata = adata.copy()
    labels = bdata.obs[label]
    #bdata.obs['high_conf'] = pd.Categorical(bdata.obs['conf'] > high_conf_rank)
    #bdata = bdata[bdata.obs['high_conf'].isin([True])]
    bdata = bdata[bdata.obs['conf_binaries'].isin([True])]
    one_hot_label, label_encoder = one_hot_encode(bdata.obs[label])
    net = Net(bdata.X.shape[1], output_size=len(labels.unique()))
    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    
    if is_scipy_cs_sparse(bdata.X):
        x_data = bdata.X.toarray()
    else:
        x_data = np.array(bdata.X)
    
    tensor_x = torch.Tensor(x_data)  # transform to torch tensor
    tensor_y = torch.Tensor(one_hot_label)
    my_dataset = TensorDataset(tensor_x, tensor_y)  # create your dataset
    sampler = create_weighted_sampler(bdata, label)
    trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                              sampler=sampler, num_workers=2)
    net = train_neural_net(net, device, trainloader, criterion, optimizer, epoch_num=epoch_num)
    
    adata.obs[new_annotation] = adata.obs[label]
    cdata = cdata[cdata.obs['conf_binaries'].isin([False])]
    cdata_tensor = torch.tensor(cdata.X.toarray())
    cdata_tensor = cdata_tensor.to(device)
    net_output = net(cdata_tensor)
    del cdata_tensor
    
    inverted = label_encoder.inverse_transform(np.argmax(net_output.cpu().detach().numpy(), axis=1))
    adata.obs[new_annotation][cdata.obs_names] = inverted
    
    return adata


import torch

def train_neural_net(net, device, trainloader, criterion, optimizer, epoch_num=60):
    """
    Train a neural network using the specified parameters.

    Parameters
    ----------
    net : torch.nn.Module
        The neural network model to be trained.
    trainloader : torch.utils.data.DataLoader
        DataLoader containing the training dataset.
    criterion : torch.nn.modules.loss._Loss
        Loss function used for training.
    optimizer : torch.optim.Optimizer
        Optimizer used for updating the network weights.
    epoch_num : int, optional
        Number of epochs for training, by default 60.

    Returns
    -------
    torch.nn.Module
        Trained neural network model.
    """
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if device.type != 'cpu':
                #print('Running with cuda')
                inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 10 == 1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
    
    return net



def arg_max_labele(y_pred, y_true):
    prob = torch.exp(y_pred)
    return torch.sum(prob * y_true, axis=1)

def follow_train_dyn_two_lables(adata, label_one, label_two, iterNum=100, lr=0.001, momentum=0.9,
                    device='cpu', weighted_sampler=True, batch_size=256):
    """
    Initialize and train a neural network on single-cell RNA sequencing data.

    This function initializes a neural network, prepares the dataset, and trains
    the network using stochastic gradient descent.

    Parameters:
    ----------
    adata : AnnData
        Anndata object containing the single-cell RNA sequencing data.

    label_key : str
        The key in adata.obs where the cell labels are stored.

    iterNum : int, optional (default=100)
        Number of training iterations (epochs).

    lr : float, optional (default=0.001)
        Learning rate for the optimizer.

    momentum : float, optional (default=0.9)
        Momentum for the optimizer.

    device : str, optional (default='cpu')
        Device for training the neural network ('cpu' or 'cuda' for GPU).

    weighted_sampler : bool, optional (default=True)
        Whether to use a weighted sampler for class imbalance.

    batch_size : int, optional (default=256)
        Batch size for training.

    Returns:
    -------
    list
        A list of confidence probability losses during training.

    Notes:
    ------
    - This function assumes that you have defined the neural network architecture
      in a separate module as 'Net'.
    - 'one_hot_encode' should be a function that encodes labels as one-hot vectors.
    - 'create_weighted_sampler' should be a function that creates a weighted sampler
      for handling class imbalance.

    - Ensure that the necessary PyTorch and scikit-learn packages are installed.
    """
    one_hot_label_1, one_hot_label_2,  inverted_label = one_hot_encode_two_labels(adata.obs[label_one],adata.obs[label_two])
    net = Net(adata.X.shape[1], output_size=len(adata.obs[label_one].unique()))

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    if is_scipy_cs_sparse(adata.X):
        x_data = adata.X.toarray()
    else:
        x_data = np.array(adata.X)
    tensor_x = torch.Tensor(x_data)  # Transform to torch tensor
    tensor_y = torch.Tensor(one_hot_label_1)
    tensor_y_2 = torch.Tensor(one_hot_label_2)
    tensor_x = tensor_x.to(device)
    tensor_y = tensor_y.to(device)
    tensor_y_2 = tensor_y_2.to(device)
    my_dataset = TensorDataset(tensor_x, tensor_y)  # Create your dataset

    if weighted_sampler:
        sampler = create_weighted_sampler(adata, label_one)
        trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                  sampler=sampler, num_workers=0)
    else:
        trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
    prob_loss_list_1 = []
    prob_loss_list_2 = []
    for epoch in range(iterNum):  # Loop over the dataset multiple times
        outputs_all = net(tensor_x)
        prob_all = probability_for_confidence(outputs_all, tensor_y)
        prob_loss_list_1.append((prob_all.cpu().detach().numpy()))
        prob_all = probability_for_confidence(outputs_all, tensor_y_2)
        prob_loss_list_2.append((prob_all.cpu().detach().numpy()))

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if device.type != 'cpu':
                #print('Running with cuda')
                inputs, labels = inputs.cuda(), labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 1:  # Print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
    return prob_loss_list_1 , prob_loss_list_2

def get_genes_dynamics(adata, label_key, iterNum=100, lr=0.001, momentum=0.9,
                    device='cpu', weighted_sampler=True, batch_size=256):
    """
    Initialize and train a neural network on single-cell RNA sequencing data.

    This function initializes a neural network, prepares the dataset, and trains
    the network using stochastic gradient descent.

    Parameters:
    ----------
    adata : AnnData
        Anndata object containing the single-cell RNA sequencing data.

    label_key : str
        The key in adata.obs where the cell labels are stored.

    iterNum : int, optional (default=100)
        Number of training iterations (epochs).

    lr : float, optional (default=0.001)
        Learning rate for the optimizer.

    momentum : float, optional (default=0.9)
        Momentum for the optimizer.

    device : str, optional (default='cpu')
        Device for training the neural network ('cpu' or 'cuda' for GPU).

    weighted_sampler : bool, optional (default=True)
        Whether to use a weighted sampler for class imbalance.

    batch_size : int, optional (default=256)
        Batch size for training.

    Returns:
    -------
    list
        A list of confidence probability losses during training.

    Notes:
    ------
    - This function assumes that you have defined the neural network architecture
      in a separate module as 'Net'.
    - 'one_hot_encode' should be a function that encodes labels as one-hot vectors.
    - 'create_weighted_sampler' should be a function that creates a weighted sampler
      for handling class imbalance.

    - Ensure that the necessary PyTorch and scikit-learn packages are installed.
    """
    one_hot_label, inverted_label = one_hot_encode(adata.obs[label_key])
    net = Net(adata.X.shape[1], output_size=len(adata.obs[label_key].unique()))

    net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    if is_scipy_cs_sparse(adata.X):
        x_data = adata.X.toarray()
    else:
        x_data = np.array(adata.X)
    x_data_scaled = x_data.copy()
    x_data_scaled = normalize(x_data_scaled, axis=0)
    x_data_scaled = torch.Tensor(x_data_scaled)  # Transform to torch tensor
    x_data_scaled = x_data_scaled.to(device)
    tensor_x = torch.Tensor(x_data)  # Transform to torch tensor
    tensor_y = torch.Tensor(one_hot_label)
    tensor_x = tensor_x.to(device)
    tensor_y = tensor_y.to(device)
    my_dataset = TensorDataset(tensor_x, tensor_y)  # Create your dataset
    if weighted_sampler:
        sampler = create_weighted_sampler(adata, label_key)
        trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                  sampler=sampler, num_workers=0)
    else:
        trainloader = torch.utils.data.DataLoader(my_dataset, batch_size=batch_size,
                                                  shuffle=True, num_workers=0)
    prob_loss_list = []
    gene_score_list = []
    for epoch in range(iterNum):  # Loop over the dataset multiple times
        outputs_all = net(tensor_x)
        prob_all = probability_for_confidence(outputs_all, tensor_y)
        gene_score = x_data_scaled.T @ prob_all
        gene_score_list.append((gene_score.cpu().detach().numpy()))
        prob_loss_list.append((prob_all.cpu().detach().numpy()))

        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # Get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if device.type != 'cpu':
                #print('Running with cuda')
                inputs, labels = inputs.cuda(), labels.cuda()

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % 10 == 1:  # Print every 10 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 10:.3f}')
                running_loss = 0.0
    return prob_loss_list , gene_score_list
