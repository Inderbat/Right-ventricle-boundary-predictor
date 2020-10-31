import torch
import os
import ray
import pydicom
import numpy as np
from config import conf
from torch import nn
from PIL import Image
from tqdm import tqdm
from ray import tune
from ray.tune import CLIReporter
from ray.tune.schedulers import ASHAScheduler
from functools import partial
from data_utils import *
from models import *

'''
	Loss function utils
'''
# variance loss function (Code taken from: https://github.com/huiqu18/FullNet-varCE)
class LossVariance(nn.Module):
    def __init__(self):
        super(LossVariance, self).__init__()

    def forward(self, input, target):
        B = input.size(0)

        loss = 0
        for k in range(B):
            unique_vals = target[k].unique()
            unique_vals = unique_vals[unique_vals != 0]

            sum_var = 0
            for val in unique_vals:
                instance = input[k][:, target[k] == val]
                if instance.size(1) > 1:
                    sum_var += instance.var(dim=1).sum()

            loss += sum_var / (len(unique_vals) + 1e-8)
        loss /= B
        return loss

# generators a linear vector of dice coefficients for each batch
def compute_dice_coefficient(estimated_foreground, actual_foreground, epsilon=1e-12):
    
    # flatten the images
    estimated_flatten = estimated_foreground.contiguous().view(estimated_foreground.shape[0], -1)
    target_flatten = actual_foreground.contiguous().view(actual_foreground.shape[0], -1)
    
    # computation of important terms
    intersection = 2 * torch.sum(estimated_flatten * target_flatten, dim=1) + epsilon
    estimated_sum = torch.sum(estimated_flatten, dim=1)
    target_sum = torch.sum(target_flatten, dim=1)
    denominator = estimated_sum + target_sum + epsilon
    dice_vector = torch.div(intersection, denominator)
    return dice_vector


'''
	Hyperparameter tuning utils

    loss_type = FL : Focal Loss
    loss_type = CEDL : Cross entropy Dice Loss
    loss_type = CEDIL : Cross entropy Dice Inverse Dice Loss
    loss_type = SL : Switching Loss
    loss_type = VCE : variance constrained Cross entropy losss

    Learning_rate_scheduler = CARM : Cosine annealing warm restarts
    Learning_rate_scheduler = SLR : Step Learning Rate scheduling
    Learning_rate_scheduler = MLR : Multiplicative learning rate scheduling
    Learning_rate_scheduler = RLROP : Reduce Learning rate on plateau
    
    optimizer_type = SGD : Stochastic gradient descent
    optimizer_type = Adam : Adam
    optimizer_type = Adad : Adadelta
    optimizer_type = Adag : Adagrad
    optimizer_type = RMSp : RMSprop
'''

# generates objects to initiate hyperparameter optimization
def hyperparameter_tuning_initializer(loss_type='SL', learning_rate_scheduler ='CARM'):
    
    # defining the hyperparameters
    if loss_type == 'FL':
        config = {
            'gamma': tune.choice([0.5, 1, 2]),
            'lr': tune.loguniform(1e-4, 1e-3)
        }
    elif loss_type == 'CEDL':
        config = {
            'dice_loss': tune.uniform(0, 3),
            'lr': tune.loguniform(1e-4, 1e-3)
        }
    elif loss_type == 'CEDIL':
        config = {
            'dice_loss': tune.uniform(0, 3),
            'inverse_dice_loss': tune.uniform(0, 3),
            'lr': tune.loguniform(1e-4, 1e-3)
        }
    elif loss_type == 'SL':
        config = {
            'lambda': tune.uniform(0, 1),
            'tau': tune.uniform(0.02, 0.04),
            'lr': tune.loguniform(1e-4, 1e-3)
        }
    elif loss_type == 'VCE':
        config = {
            'var_loss':  tune.uniform(0.5, 5.5),
            'lr': tune.loguniform(1e-4, 1e-3)
        }

    # hyperparameters for learning rate scheduler
    if learning_rate_scheduler == 'CARM':
        config['T_0'] = tune.choice([5, 10, 20, 40, 50])
        config['eta_min_factor'] = tune.loguniform(1e2, 1e4)
    if learning_rate_scheduler == 'SLR':
        config['step_size'] = tune.choice([5, 10, 20, 40, 50])
    if learning_rate_scheduler == 'MLR':
        config['lr_lambda'] = tune.uniform(0.8, 0.99)

        
    # defining the scheduler
    scheduler = ASHAScheduler(
        metric='loss',
        mode='min',
        max_t=conf['max_epochs'] // 20,
        grace_period=1,
        reduction_factor=2
    )
    
    # defining the reporter
    reporter = CLIReporter(metric_columns=['loss', 'avg_dice_coefficient', 'epoch'])
    
    return config, scheduler, reporter

# function for training the network
def customized_training(config, checkpoint_dir=None, data_loader_train=None, data_loader_val=None, loss_type='SL', learning_rate_scheduler='CARM', optimizer_type='Adam'):
    
    # looking for GPU support    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # initializing the optimizer
    u_net = Unet().to(device)
    if optimizer_type == 'SGD':
        optimizer = torch.optim.SGD(u_net.parameters(), lr=config['lr'])
    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(u_net.parameters(), lr=config['lr'])
    if optimizer_type == 'Adad':
        optimizer = torch.optim.Adadelta(u_net.parameters(), lr=config['lr'])
    if optimizer_type == 'Adag':
        optimizer = torch.optim.Adagrad(u_net.parameters(), lr=config['lr'])
    if optimizer_type == 'RMSp':
        optimizer = torch.optim.RMSprop(u_net.parameters(), lr=config['lr'])
    data_loader_train_len = len(data_loader_train)

    # initializing the learning rate scheduler
    if learning_rate_scheduler == 'CARM':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=config['T_0'], eta_min=config['lr'] / config['eta_min_factor'])
    if learning_rate_scheduler == 'SLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=config['step_size'])
    if learning_rate_scheduler == 'MLR':
        lr_lambda = lambda epoch: config['lr_lambda']
        scheduler = torch.optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda=lr_lambda)
    if learning_rate_scheduler == 'RLROP':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # initializing the criteria for computing loss variance
    if loss_type == 'VCE':
        loss_variance = LossVariance()
        
    if checkpoint_dir:
        model_state, optimizer_state = model.load(os.path.join(checkpoint_dir, 'checkpoint'), map_location=device)
        u_net.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        
    # training process and validation
    for epoch in tqdm(range(conf['max_epochs'])):
        
        # training process
        total_loss = 0
        u_net.train()
        for batch_index, data_batch in enumerate(data_loader_train):
            
            # casting the variables to device
            image_data, mask_data = data_batch
            image_data = image_data.to(device, dtype=torch.float)
            mask_data = (mask_data > 0).clone().to(device, dtype=torch.float)
            
            # clearing the buffer
            optimizer.zero_grad()
            
            # computing the loss function
            model_output = u_net(image_data)
            
            # focal loss type
            if loss_type == 'FL':
                pos_loss = -torch.pow(model_output[:, 0, :, :], config['gamma']) * torch.log(model_output[:, 1, :, :] + 1e-12)
                neg_loss = -torch.pow(model_output[:, 1, :, :], config['gamma']) * torch.log(model_output[:, 0, :, :] + 1e-12)
                loss_vector = torch.where(torch.squeeze(mask_data).to(dtype=torch.bool), pos_loss, neg_loss)
                loss = torch.mean(loss_vector)
            
            # cross entropy + dice loss + ?inverse dice loss
            if loss_type == 'CEDL' or loss_type == 'CEDIL':
                ce_loss = -(mask_data * torch.log(model_output[:, 1, :, :]) + (1 - mask_data) * torch.log(model_output[:, 0, :, :]))
                ce_loss_value = torch.mean(ce_loss)
                dice_loss_value = torch.mean(1 - compute_dice_coefficient(model_output[:, 1, :, :], mask_data))
                loss = ce_loss_value + config['dice_loss'] * dice_loss_value
                if loss_type == 'CEDIL':
                    inverse_dice_loss_value = torch.mean(1 - compute_dice_coefficient(model_output[:, 0, :, :], 1 - mask_data))
                    loss += config['inverse_dice_loss'] * inverse_dice_loss_value
                    
            # switching loss
            if loss_type == 'SL':
                switch_ratio = (torch.sum(mask_data) / torch.prod(torch.tensor(mask_data.shape))).item()
                ce_loss = -(mask_data * torch.log(model_output[:, 1, :, :]) + (1 - mask_data) * torch.log(model_output[:, 0, :, :]))
                ce_loss_value = torch.mean(ce_loss)
                dice_loss_value = torch.mean(1 - compute_dice_coefficient(model_output[:, 1, :, :], mask_data))
                inverse_dice_loss_value = torch.mean(1 - compute_dice_coefficient(model_output[:, 0, :, :], 1 - mask_data))
                if switch_ratio > config['tau']:
                    loss = ce_loss_value + config['lambda'] * dice_loss_value + (1 - config['lambda']) * inverse_dice_loss_value
                else:
                    loss = ce_loss_value + config['lambda'] * inverse_dice_loss_value + (1 - config['lambda']) * dice_loss_value

            # Variance Constrained Cross entropy loss
            if loss_type == 'VCE':
                ce_loss = -(mask_data * torch.log(model_output[:, 1, :, :]) + (1 - mask_data) * torch.log(model_output[:, 0, :, :]))
                ce_loss_value = torch.mean(ce_loss)
                var_loss_value = loss_variance(model_output, torch.squeeze(mask_data))
                loss = ce_loss_value + config['var_loss'] * var_loss_value
                    
            # using the loss variable for autograd
            total_loss += loss
            loss.backward()
            optimizer.step()

            # updating the optimizer through scheduler
            if learning_rate_scheduler == 'CARM':
                scheduler.step(epoch + batch_index / data_loader_train_len)
        
        # printing the loss value
        print('total loss:', total_loss, 'epoch:', epoch)

        # updating the optimizer through scheduler
        if learning_rate_scheduler == 'SLR' or learning_rate_scheduler == 'MLR':
            scheduler.step()
        if learning_rate_scheduler == 'RLROP':
            scheduler.step(total_loss.detach().item())
        
        # validation process
        u_net.eval()
        no_of_batches = 0
        val_loss = 0
        total_dice_coefficient = 0
        with torch.no_grad():
            for batch_index, data_batch in enumerate(data_loader_val):

                # casting the variables to device
                image_data, mask_data = data_batch
                image_data = image_data.to(device, dtype=torch.float)
                mask_data = (mask_data > 0).clone().to(device, dtype=torch.float)
                model_output = u_net(image_data)
                
                # focal loss type
                if loss_type == 'FL':
                    pos_loss = -torch.pow(model_output[:, 0, :, :], 0.5) * torch.log(model_output[:, 1, :, :] + 1e-12)
                    neg_loss = -torch.pow(model_output[:, 1, :, :], 0.5) * torch.log(model_output[:, 0, :, :] + 1e-12)
                    loss_vector = torch.where(torch.squeeze(mask_data).to(dtype=torch.bool), pos_loss, neg_loss)
                    loss = torch.mean(loss_vector)

                # cross entropy + dice loss + ?inverse dice loss
                if loss_type == 'CEDL' or loss_type == 'CEDIL':
                    ce_loss = -(mask_data * torch.log(model_output[:, 1, :, :]) + (1 - mask_data) * torch.log(model_output[:, 0, :, :]))
                    ce_loss_value = torch.mean(ce_loss)
                    dice_loss_value = torch.mean(1 - compute_dice_coefficient(model_output[:, 1, :, :], mask_data))
                    loss = ce_loss_value + config['dice_loss'] * dice_loss_value
                    if loss_type == 'CEDIL':
                        inverse_dice_loss_value = torch.mean(1 - compute_dice_coefficient(model_output[:, 0, :, :], 1 - mask_data))
                        loss += config['inverse_dice_loss'] * inverse_dice_loss_value

                # switching loss
                if loss_type == 'SL':
                    switch_ratio = (torch.sum(mask_data) / torch.prod(torch.tensor(mask_data.shape))).item()
                    ce_loss = -(mask_data * torch.log(model_output[:, 1, :, :]) + (1 - mask_data) * torch.log(model_output[:, 0, :, :]))
                    ce_loss_value = torch.mean(ce_loss)
                    dice_loss_value = torch.mean(1 - compute_dice_coefficient(model_output[:, 1, :, :], mask_data))
                    inverse_dice_loss_value = torch.mean(1 - compute_dice_coefficient(model_output[:, 0, :, :], 1 - mask_data))
                    if switch_ratio > config['tau']:
                        loss = ce_loss_value + config['lambda'] * dice_loss_value + (1 - config['lambda']) * inverse_dice_loss_value
                    else:
                        loss = ce_loss_value + config['lambda'] * inverse_dice_loss_value + (1 - config['lambda']) * dice_loss_value
                
                # Variance Constrained Cross entropy loss
                if loss_type == 'VCE':
                    ce_loss = -(mask_data * torch.log(model_output[:, 1, :, :]) + (1 - mask_data) * torch.log(model_output[:, 0, :, :]))
                    ce_loss_value = torch.mean(ce_loss)
                    var_loss_value = loss_variance(model_output, torch.squeeze(mask_data))
                    loss = ce_loss_value + config['var_loss'] * var_loss_value
                
                # updating the values
                val_loss += loss.detach().item()
                no_of_batches += len(mask_data)
                total_dice_coefficient += torch.sum(compute_dice_coefficient((model_output[:, 1, :, :] > 0.5).to(dtype=torch.float), mask_data)).detach().item()
                
        # printing the validation results
        print('total loss:', val_loss / no_of_batches, 'epoch:', epoch, 'avg_dice_coeff:', total_dice_coefficient / no_of_batches)
        
        # communicating with ray tune
        if epoch % 20 == 0:
            with tune.checkpoint_dir(epoch // 20) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, 'checkpoint')
                torch.save((u_net.state_dict(), optimizer.state_dict()), path)
        
            tune.report(loss=val_loss / no_of_batches, avg_dice_coefficient=total_dice_coefficient / no_of_batches, epoch=epoch // 20)


# helper for saving the models trained using different schemes
'''
    the entities in a criteria will be varied
    criteria = ['loss_type', 'learning_rate_scheduler', 'optimizer_type']
'''
def save_models(criteria):

	# looking for GPU support
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # conditioning on aspects
    if criteria == 'loss_type':
        props = ['FL', 'CEDL', 'CEDIL', 'SL', 'VCE']
    elif criteria == 'learning_rate_scheduler':
        props = ['CARM', 'SLR', 'RLR', 'RLROP']
    elif criteria == 'optimizer_type':
        props = ['SGD', 'Adam', 'Adad', 'Adag', 'RMSp']

    # supervision levels
    supervision_levels = np.arange(0.2, 1.1, 0.2)
    supervision_string = ['2', '4', '6', '8', '10']

    for (sp_level, sp_string) in zip(supervision_levels, supervision_string):
        
        # loading the data
        data_loader_train, data_loader_test = load_data('train_dir', clahe=True, supervision_level=sp_level)
        data_loader_train = augment_training_data_loader(data_loader_train)
        
        # changing the properties
        for prop in props:

            # hyperparameter process initialize
            ray.shutdown()
            ray.init()

            # setting the properties
            loss_type = prop if criteria == 'loss_type' else 'SL'
            learning_rate_scheduler = prop if criteria == 'learning_rate_scheduler' else 'CARM'
            optimizer_type = prop if criteria == 'optimizer_type' else 'Adam'

            # intializing the hyperparameters
            config, scheduler, reporter = hyperparameter_tuning_initializer(loss_type=loss_type, learning_rate_scheduler=learning_rate_scheduler)

            # hyperparameter tuning begins
            result = tune.run(
                partial(
                    customized_training, data_loader_train=data_loader_train, data_loader_val=data_loader_test,
                    loss_type=loss_type, learning_rate_scheduler=learning_rate_scheduler, optimizer_type=optimizer_type,
                ),
                config=config,
                resources_per_trial={'cpu': 2, 'gpu': 0.5},
                scheduler=scheduler,
                progress_reporter=reporter,
                num_samples=10,
            )

            # retrieving the best model
            best_trial = result.get_best_trial("avg_dice_coefficient", "max", "last")
            print("Best trial config: {}".format(best_trial.config))
            print("Best trial final validation avg_dice_coefficient: {}".format(best_trial.last_result["avg_dice_coefficient"]))
            u_net = Unet().to(device)
            best_checkpoint_dir = best_trial.checkpoint.value
            model_state, optimizer_state = torch.load(os.path.join(best_checkpoint_dir, "checkpoint"))
            u_net.load_state_dict(model_state)
            torch.save(u_net.state_dict(), 'unet_baseline_{}_{}_{}_{}.pt'.format(loss_type, optimizer_type, learning_rate_scheduler, sp_string))