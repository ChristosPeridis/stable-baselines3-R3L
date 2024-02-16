#!/usr/bin/python

# import matplotlib.pyplot as plt
import numpy as np
import os, sys, time
import numpy.random as rn
import torch

class Task_free_continual_learning():

    def __init__(self,
                verbose=False,
                seed=123,
                dev='cpu',
                dim=4,
                hidden_units=100,
                learning_rate=0.005,
                ntasks=2,
                gradient_steps=5,
                loss_window_length=5,
                loss_window_mean_threshold=0.2,
                loss_window_variance_threshold=0.1, 
                MAS_weight=0.5,
                recent_buffer_size=30,
                hard_buffer_size=30):

        torch.manual_seed(seed)
        device = torch.device(dev)
        
        # Save settings
        self.verbose=verbose
        self.dim=dim
        self.ntasks=ntasks
        self.gradient_steps=gradient_steps
        self.loss_window_length=loss_window_length
        self.loss_window_mean_threshold=loss_window_mean_threshold
        self.loss_window_variance_threshold=loss_window_variance_threshold
        self.MAS_weight=MAS_weight
        self.recent_buffer_size=recent_buffer_size
        self.hard_buffer_size=hard_buffer_size



        # Create training model
        self.model = torch.nn.Sequential(
                  torch.nn.Linear(dim, hidden_units, bias=True),
                  torch.nn.ReLU(),
                  torch.nn.Linear(hidden_units, hidden_units, bias=True),
                  torch.nn.ReLU(),
                  torch.nn.Linear(hidden_units, 2, bias=False),
        ).to(device)
        # define loss and optimizer, our method can work with any other loss.
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        self.optimizer=torch.optim.SGD(self.model.parameters(), lr=learning_rate)
        
        # initialize model
        for p in self.model.parameters():
            torch.nn.init.normal_(p, 0, 0.1)


    
    

def evaluate_hard_buffer(self, use_hard_buffer, hard_buffer, xh, yh):
        total_loss = 0  # Initialize total_loss if it's used outside this method as well
        if use_hard_buffer and len(hard_buffer) != 0:
            # Evaluate hard buffer
            yh_pred = self.model(torch.from_numpy(np.asarray(xh).reshape(-1, self.dim)).type(torch.float32))
            yh_sup = torch.zeros(len(yh), 2).scatter_(1, torch.from_numpy(np.asarray(yh).reshape(-1, 1)).type(torch.LongTensor), 1.).type(torch.FloatTensor)
            
            hard_loss = self.loss_fn(yh_pred, yh_sup)
            total_loss += torch.sum(self.loss_fn(yh_pred, yh_sup))

        return total_loss

def add_mas_regularization(self, total_loss, continual_learning, star_variables, omegas):
        # Initialize total_loss if it's being used to accumulate loss outside this method as well
        if continual_learning and len(star_variables) != 0 and len(omegas) != 0:
            for pindex, p in enumerate(self.model.parameters()):
                mas_reg_loss = self.MAS_weight / 2. * torch.sum(torch.from_numpy(omegas[pindex]).type(torch.float32) * (p - star_variables[pindex]) ** 2)
                total_loss += mas_reg_loss

        return total_loss

def save_training_accuracy(self, use_hard_buffer, hard_buffer, x, y, xh, yh, recent_loss, hard_loss=None):
        losses = []  # Assuming losses is a list to track accuracy or other metrics
        msg = ""  # Initialize message string

        # Combine data from the primary dataset and the hard buffer if applicable
        if use_hard_buffer and len(hard_buffer) != 0:
            xt = x + xh
            yt = y + yh
        else:
            xt = x[:]
            yt = y[:]

        # Make predictions and calculate accuracy
        yt_pred = self.model(torch.from_numpy(np.asarray(xt).reshape(-1, self.dim)).type(torch.float32))
        accuracy = np.mean(np.argmax(yt_pred.detach().numpy(), axis=1) == yt)

        # Update message with recent loss
        msg += ' recent loss: {0:0.3f}'.format(np.mean(recent_loss.detach().numpy()))

        # If using hard buffer, update message with hard loss
        if use_hard_buffer and len(hard_buffer) != 0 and hard_loss is not None:
            msg += ' hard loss: {0:0.3f}'.format(np.mean(hard_loss.detach().numpy()))

        # Append accuracy to losses list
        losses.append(np.mean(accuracy))

        return msg, losses, accuracy


def update_loss_window_and_detect_peak(self, first_train_loss):
        '''Needs carefull engineering with the first_train_loss'''
        # Add current loss to the loss window
        self.loss_window.append(np.mean(first_train_loss))
        # Keep the loss window size within the specified length
        if len(self.loss_window) > self.loss_window_length:
            del self.loss_window[0]

        # Calculate current window's statistics
        loss_window_mean = np.mean(self.loss_window)
        loss_window_variance = np.var(self.loss_window)

        # Check for new peak detection based on the statistics of the current window
        if not self.new_peak_detected and loss_window_mean > self.last_loss_window_mean + np.sqrt(self.last_loss_window_variance):
            self.new_peak_detected = True

        # Update the last loss window's statistics for future comparisons
        self.last_loss_window_mean = loss_window_mean
        self.last_loss_window_variance = loss_window_variance

        # Optionally, return current statistics and peak detection status for external use
        return self.new_peak_detected, loss_window_mean, loss_window_variance

def update_importance_weights_and_variables(self, continual_learning, new_peak_detected, loss_window_mean, loss_window_variance, hard_buffer):
        
        if continual_learning and loss_window_mean < self.loss_window_mean_threshold and loss_window_variance < self.loss_window_variance_threshold and new_peak_detected:
            self.count_updates += 1
            self.update_tags.append(0.01)
            self.last_loss_window_mean = loss_window_mean
            self.last_loss_window_variance = loss_window_variance
            new_peak_detected = False
            
            # Calculate importance weights and update star_variables
            gradients = [0 for _ in self.model.parameters()]
            
            # Calculate importance based on each sample in the hard buffer
            for sx in [x['state'] for x in hard_buffer]:
                self.model.zero_grad()
                y_pred = self.model(torch.from_numpy(np.asarray(sx).reshape(-1, self.dim)).type(torch.float32))
                torch.norm(y_pred, 2, dim=1).backward()
                for pindex, p in enumerate(self.model.parameters()):
                    g = p.grad.data.clone().detach().numpy()
                    gradients[pindex] += np.abs(g)
            
            # Update the running average of the importance weights
            omegas_old = self.omegas[:]
            self.omegas = []
            self.star_variables = []
            for pindex, p in enumerate(self.model.parameters()):
                if len(omegas_old) != 0:
                    self.omegas.append(1 / self.count_updates * gradients[pindex] + (1 - 1 / self.count_updates) * omegas_old[pindex])
                else:
                    self.omegas.append(gradients[pindex])
                self.star_variables.append(p.data.clone().detach())
        else:
            self.update_tags.append(0)
        self.loss_window_means.append(loss_window_mean)
        self.loss_window_variances.append(loss_window_variance)
        
        # Optionally return the updated statistics and flags
        return {
            "update_tags": self.update_tags,
            "loss_window_means": self.loss_window_means,
            "loss_window_variances": self.loss_window_variances,
            "omegas": self.omegas,
            "star_variables": self.star_variables,
            "new_peak_detected": new_peak_detected
        }

def update_hard_buffer(self, use_hard_buffer, recent_loss, hard_loss, xt, yt):
        if use_hard_buffer:
            # Determine the loss for updating the hard buffer
            if len(self.hard_buffer) == 0:
                loss = recent_loss.detach().numpy()
            else:
                loss = torch.cat((recent_loss, hard_loss))
                loss = loss.detach().numpy()

            # Reset hard buffer and calculate mean loss
            self.hard_buffer = []
            loss = np.mean(loss, axis=1)

            # Sort inputs and targets by loss
            sorted_inputs = [np.asarray(lx) for _, lx in reversed(sorted(zip(loss.tolist(), xt), key=lambda f: f[0]))]
            sorted_targets = [ly for _, ly in reversed(sorted(zip(loss.tolist(), yt), key=lambda f: f[0]))]

            # Update the hard buffer with the highest-loss examples
            for i in range(min(self.hard_buffer_size, len(sorted_inputs))):
                self.hard_buffer.append({'state': sorted_inputs[i], 'trgt': sorted_targets[i]})

def evaluate_test_accuracy(self, data, loss_fn):
        msg = ""  # Initialize message string
        for i in range(self.ntasks):
            # Predict the outputs for test inputs of the ith task
            y_pred = self.model(torch.from_numpy(data.test_inputs[i].reshape(-1, self.dim)).type(torch.float32))
            # Create a one-hot encoded tensor of test labels
            y_sup = torch.zeros(len(data.test_inputs[i]), 2).scatter_(
                1, torch.from_numpy(np.asarray(data.test_labels[i]).reshape(-1, 1)), 1.).type(torch.FloatTensor)
            
            # Optional: Calculate loss if needed. Uncomment the next line if test loss is required
            # loss = loss_fn(y_pred, y_sup).detach().numpy()
            
            # Calculate and append test accuracy for the ith task
            test_accuracy = np.mean(np.argmax(y_pred.detach().numpy(), axis=1) == data.test_labels[i])
            self.test_loss[i].append(test_accuracy)
            msg += ' test[{0}]: {1:0.3f}'.format(i, test_accuracy)
        
        return msg, self.test_loss