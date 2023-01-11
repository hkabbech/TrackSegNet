"""
.. module:: generate_lstm_model.py
   :synopsis: This module implements functions to generate a LSTM model
"""

# Third-party modules
import os
import pickle
from datetime import datetime
from contextlib import redirect_stdout
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc
import seaborn as sns
from tensorflow.keras.utils import to_categorical

from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import LSTM, Dense, TimeDistributed, Bidirectional 

# sns.set(color_codes=True)
# rc('text', usetex=True)

def generate_lstm_model(sim_df, par):
    """
        Generate LSTM model
    """
    print('\nTrain LSTM model...')
    start_time = datetime.now()

    ## DATA PREPARATION
    ###################
    all_features = np.array([sim_df[sim_df['track_id']==N][['displ_x', 'displ_y', 'dist', 'mean_dist_1', 'mean_dist_2', 'angle']].to_numpy()[1:-1]\
        for N in sim_df['track_id'].unique()])
    all_true_states = np.array([sim_df[sim_df['track_id']==N]['state'].to_numpy()[1:-1]\
        for N in sim_df['track_id'].unique()])
    # Split data into train and test set (with shuffeling)

    all_true_states = to_categorical(all_true_states, num_classes=par['num_states'])

    # split in training and test data
    num_train = round((1-par['percentage']['test'])*len(all_features))
    train_set = {'features': all_features[:num_train], 'states': all_true_states[:num_train]}
    test_set = {'features': all_features[num_train:], 'states': all_true_states[num_train:]}


    ## LSTM MODEL
    #############
    model = Sequential()
    model.add(Bidirectional(LSTM(par['hidden_units'], return_sequences=True),
                            input_shape=(None, par['num_features']), merge_mode='concat'))
    model.add(TimeDistributed(Dense((par['num_states']), activation='softmax'))) # Softmax to get
    # 'probability distribution'
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    # Training stops once the model performance does not improve on a hold out validation dataset
    # The best mode lis saved
    callbacks = [EarlyStopping(monitor='val_loss', patience=par['patience']),\
                 ModelCheckpoint(filepath=str(par['model_path']/'best_model.h5'), monitor='val_loss',
                                 save_best_only=True)]
    history = model.fit(train_set['features'], train_set['states'], callbacks=callbacks,
                        epochs=par['epochs'], batch_size=par['batch_size'],
                        validation_split=par['percentage']['val'], shuffle=True)
    plot_training_curves(par['model_path'], history, par['patience'])
    with open(par['model_path']/'model_summary.txt', 'w') as file:
        with redirect_stdout(file):
            model.summary()


    ## EVALUATION
    #############
    # Predicting on test set and convert binary matrix into vector
    best_model = load_model(par['model_path']/'best_model.h5')
    train_evaluation = best_model.evaluate(train_set['features'], train_set['states'], verbose=0)
    test_evaluation = best_model.evaluate(test_set['features'], test_set['states'], verbose=0)

    with open(par['model_path']/'evaluation_model.txt', 'w') as file:
        print(f'Train loss:\t{train_evaluation[0]:.2f}\tTrain accuracy:\t{train_evaluation[1]:.2f}', file=file)
        print(f'Test loss:\t{test_evaluation[0]:.2f}\tTest accuracy:\t{test_evaluation[1]:.2f}', file=file)
    predicted_test = best_model.predict(test_set['features'])
    predicted_states_test = predicted_test.argmax(axis=2) # Predicted transition states
    true_states_test = test_set['states'].argmax(axis=2) # Real transition states
    booleans = np.array(predicted_states_test == true_states_test, dtype=int)
    plot_accuracy_over_window(par['model_path'], booleans, par['track_length']-2)


    ## TIMER
    ########
    time = datetime.now() - start_time
    hours = time.seconds // 3600
    minutes = time.seconds // 60 % 60
    print(f'\nTotal runtime: {hours}h{minutes:0>2d}')


def plot_training_curves(path, history, patience):
    """
        Plot loss and accuracy training curves
    """
    path = path/'training_plots'
    os.makedirs(path, exist_ok=True)

    x_axis = range(len(history.history['loss']))
    # Plot loss curve
    plt.plot(x_axis, history.history['loss'], label='Training')
    plt.plot(x_axis, history.history['val_loss'], label='Validation')
    plt.axvline(x_axis[-1]-patience, color='k', linestyle='--', label='best model')
    plt.title('Training and validation loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend(loc='best')
    plt.savefig(path/'loss_curve.png', bbox_inches='tight')
    plt.close()
    # Plot accuracy curve
    plt.plot(x_axis, history.history['accuracy'], label='Training')
    plt.plot(x_axis, history.history['val_accuracy'], label='Validation')
    plt.axvline(x_axis[-1]-patience, color='k', linestyle='--', label='best model')
    plt.title('Training and validation accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend(loc='best')
    plt.savefig(path/'accuracy_curve.png', bbox_inches='tight')
    plt.close()

def plot_accuracy_over_window(path, booleans, window_size):
    """
        Plotting a bar graph of the accuracy gradient over window for specified window_size
        and number of hidden units
    """
    path = path/'training_plots'
    os.makedirs(path, exist_ok=True)
    # ws in par['window_size'] and hu in par['hidden_units']
    booleans_mean = booleans.mean(axis=0)
    plt.bar(np.arange(window_size)+1, booleans_mean)
    plt.gca().set_xticks(np.arange(window_size)+1)
    plt.xlabel('Window position')
    plt.ylabel('Accuracy')
    plt.title('Accuracy gradient over window (test set)', y=1.04)
    plt.gca().set_ylim([min(booleans_mean - 0.02), max(booleans_mean + 0.02)])
    plt.tight_layout()
    plt.savefig(path/f'accuracy_over_gradient_w{window_size}.png', bbox_inches='tight')
    plt.close()