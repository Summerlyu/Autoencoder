import gzip
import itertools
import os
import sys
import numpy as np
import lasagne
import theano
import theano.sparse 
import pylab
import matplotlib.pyplot as plt
import theano.tensor as T
import time
import matplotlib.pyplot as plt
import load_real_data as lrd
from lasagne.objectives import squared_error,aggregate 
from lasagne.layers import get_output
from lasagne.layers import get_all_params
from sklearn.metrics import classification_report, accuracy_score


max_epochs = 50
batch_size = 1000
learning_rate = 0.05
momentum = 0.9
io_dim = 1200
chosen_batch = 9
activation_threshold = -0.57 # between -0.5 --- -0.638

def autoencoder(input_d, output_d):
    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, input_d),
    )
    l_hidden1 = lasagne.layers.DenseLayer(
        l_in,
        num_units=300,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden2 = lasagne.layers.DenseLayer(
        l_hidden1,
        num_units=10,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_hidden3 = lasagne.layers.DenseLayer(
        l_hidden2,
        num_units=300,
        nonlinearity=lasagne.nonlinearities.rectify,
    )
    l_out = lasagne.layers.DenseLayer(
        l_hidden3,
        num_units=output_d,
        nonlinearity=lasagne.nonlinearities.linear, # linear
    )
    return l_out

def net_training(iter_funcs, data):
    num_batches_train = data['num_train_data']//batch_size
    # print("num_batches_train:{}".format(num_batches_train))
    num_batches_valid = data['num_valid_data']//batch_size
    num_batches_test = data['num_test_data']//batch_size

    for epoch in itertools.count(1):

        batch_train_losses = []
        batch_valid_losses = []
        batch_test_losses = []

        # Training 
        for i_batch in range(num_batches_train):
            batch_train_loss = iter_funcs['train'](i_batch)
            batch_train_losses.append(batch_train_loss)
        # print("batch_train_losses:{}".format(batch_train_losses))
        train_loss_mean = np.mean(batch_train_losses)

        # Validation
        for i_batch in range(num_batches_valid):
            batch_valid_loss = iter_funcs['valid'](i_batch)
            # print("i_batch:{}".format(i_batch))
            batch_valid_losses.append(batch_valid_loss)
        # print("batch_valid_losses:{}".format(batch_valid_losses))

        valid_loss_mean = np.mean(batch_valid_losses)

        # Testing
        for i_batch in range(num_batches_test):
            batch_test_loss, accuracy = iter_funcs['test'](i_batch)
            batch_test_losses.append(batch_test_loss)
            # print("accuracy:{}".format(accuracy))           
        test_loss_mean = np.mean(batch_test_losses)

        # print(len(iter_funcs['network_output'](9)[0]))
 
        result = dict(
            epoch = epoch,
            train_loss = train_loss_mean,
            valid_loss = valid_loss_mean,
            test_loss = test_loss_mean,
            network_output = iter_funcs['network_output'](chosen_batch)[0],
            network_input = iter_funcs['network_input'](chosen_batch)[0],
            target = iter_funcs['target'](chosen_batch)[0]
            )
        return result

def batch_iterations(data, output_layer):
    batch_index = T.iscalar('batch_index')
    X_batch = T.matrix('x')
    y_batch = T.matrix('y')
    batch_slice = slice(batch_index * batch_size,
                        (batch_index + 1) * batch_size)
    output = get_output(output_layer, X_batch, deterministic=False)
    
    loss_train = squared_error(output, y_batch).mean() # change to mean square error 
    output_test = get_output(output_layer, X_batch, deterministic=True)  
    loss_test = squared_error(output_test, y_batch).mean()
    predicted_out = theano.sparse.basic.sub(output_test, T.zeros_like(y_batch))

    # print(T.classification_report(output_test, y_batch)) # Classification on each digit
    accuracy = T.mean(T.eq(output_test, y_batch), dtype=theano.config.floatX)

    all_params = get_all_params(output_layer)

    updates = lasagne.updates.nesterov_momentum(
        loss_train, all_params, learning_rate, momentum)

    iter_train = theano.function(
        [batch_index], loss_train,
        updates=updates,
        givens={
            X_batch: data['X_train'][batch_slice],
            y_batch: data['y_train'][batch_slice],
        },
    )
    
    iter_valid = theano.function(
        [batch_index], [loss_test, accuracy],
        givens={
            X_batch: data['X_valid'][batch_slice],
            y_batch: data['y_valid'][batch_slice],
        },
    )

    iter_test = theano.function(
        [batch_index], [loss_test, accuracy],
        givens={
            X_batch: data['X_test'][batch_slice],
            y_batch: data['y_test'][batch_slice],
        },
    )

    network_input = theano.function(
        [batch_index], data['X_valid'][batch_slice],
    )

    network_output = theano.function(
        [batch_index], predicted_out,
        givens={
            X_batch: data['X_valid'][batch_slice],
            y_batch: data['y_valid'][batch_slice],
        },
    )

    target = theano.function(
        [batch_index], data['y_valid'][batch_slice],
    )
   
    return dict(
        train=iter_train,
        valid=iter_valid,
        test=iter_test,
        network_output = network_output,
        network_input = network_input,
        target = target,
    )

def save_activation(activation_list):
    binary_list = []
    for each_value in activation_list:
        if each_value > activation_threshold:
            binary_list.append(1)
        else:
            binary_list.append(0)
    return binary_list

def classification_main():
    epoch_no = []
    train_loss = []
    valid_loss = []
    test_loss = []
    valid_accuracy = []
    print("--------------------------------------------")
    print("Loading data...")
    data = lrd.load_real_data_main()
    print("--------------------------------------------")
    print("Building model...")

    output_layer = autoencoder(input_d=data['input_d'],output_d=data['output_d'])

    print("--------------------------------------------")
    print("Training model...")
    iter_funcs = batch_iterations(data, output_layer)

    for each_epoch in range(max_epochs):
        result = net_training(iter_funcs, data)
        epoch_no.append(each_epoch)

        network_output = result['network_output']
        network_input = result['network_input']
        target = result['target']
        binary_net_pre = save_activation(network_output)
        binary_net_tgt = save_activation(target)

        accuracy = accuracy_score(binary_net_tgt, binary_net_pre)
        print("Epoch {} of {} Accuracy:{:.2f}".format(each_epoch+1, max_epochs,accuracy))
        print("train_loss:{:.2f}%".format(result['train_loss']*100))
        print("valid_loss:{:.2f}%".format(result['valid_loss']*100))
        print("test_loss:{:.2f}%".format(result['test_loss']*100))
        train_loss.append(result['train_loss'])
        valid_loss.append(result['valid_loss'])
        test_loss.append(result['test_loss'])


        # Plot activation figure for network input, network output and target
        if each_epoch==max_epochs-1:
            # Binary matrix for accuracy and evaluation
            print(classification_report(binary_net_tgt, binary_net_pre))

            plt.subplot(3, 1, 1)
            plt.plot(network_output, 'b-')
            plt.ylabel('Network Output')
            plt.grid()

            plt.subplot(3, 1, 2)
            plt.plot(target, 'r-', label='Wash Machine')
            plt.legend(fontsize = 'x-small')
            plt.ylabel('Target')
            plt.grid()

            plt.subplot(3, 1, 3)
            plt.plot(network_input, 'y-')
            plt.ylabel('Network Input')
            
            plt.xlabel('Time')
            
            plt.grid()
            plt.show()
            plt.close()

    plt.plot(epoch_no, train_loss)
    plt.plot(epoch_no, valid_loss)
    plt.plot(epoch_no, test_loss)
    plt.ylabel('Error')
    plt.xlabel('No_epoch')
    plt.legend(['train_loss', 'valid_loss', 'test_loss'], loc='upper right')
    plt.show()


if __name__ == '__main__':
    classification_main()
