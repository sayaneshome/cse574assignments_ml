from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.utils import np_utils
from keras import optimizers
from sklearn.metrics import confusion_matrix
import numpy as np
import time
from keras.layers.convolutional import Conv2D

np.random.seed(7)
#set the work directory here 
data_dir = "/Users/sayaneshome/Desktop/cse573_Shome/" 
#training data
data = np.loadtxt(data_dir + "optdigits.tra", delimiter=",")
X = data[:, 0:64].astype('float32')
X = X/16
cnn_train = X.reshape(X.shape[0], 1, 1, 64)
label = data[:, 64]
Y = np_utils.to_categorical(label)

#test data
data1 = np.loadtxt(data_dir + "optdigits.tes", delimiter=",")
X1 = data1[:, 0:64].astype('float32')/16
cnn_test = X1.reshape(X1.shape[0], 1, 1, 64)
label1 = data1[:, 64]

#function to print the accuracy and confusion matrix for both the cases
def print_statistics(model, x, y):
    predictions = model.predict(x)
    list = []
    for row in predictions:
        list.append(np.argmax(row))
    m = confusion_matrix(y, list)
    sum = 0
    print("Class Accuracies:")
    for i in range(10):
        sum += m[i][i]
        print("Class ", i, ": ", round(m[i][i]/np.sum(m[i])*100, 4))
    print("Overall Accuracy: ", round(sum/np.sum(m), 4)*100)
    print("Confusion Matrix:\n", m)

#Question 1 
#Hyper-Parameters for fully-feed fully-connected neural networks
#best parameters for NN have been shown here
#For running only this part,Comment out all the part below Question 2
momentum_rate = 0.98
filters = 1024
batch_size = 100
learning_rate = 0.05
neurons = 200
# Function for Neural Network Model
def neural_network_model(hidden_units, error_function, data, label):
    startTime = time.clock()
    model = Sequential()
    model.add(Dense(neurons, input_dim=64, activation=hidden_units))  # First hidden layer
    model.add(Dense(neurons, activation=hidden_units))  # Second hidden layer
    model.add(Dense(neurons, activation=hidden_units))  # Third hidden layer
   # model.add(Dense(neurons, activation=hidden_units))  # Fourth hidden layer
  #  model.add(Dense(neurons, activation=hidden_units)) #fifth hidden layer
    model.add(Dense(10, activation='softmax'))  # Softmax function for output layer
    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=momentum_rate, nesterov=True)
    model.compile(loss=error_function, optimizer=sgd, metrics=['accuracy'])
    model.fit(X, Y, validation_split=0.2, epochs=10, batch_size=200, verbose=0)
    endTime = time.clock()
    print("Time = ", endTime-startTime)
    print_statistics(model, data, label)

print("Hyper-parameter values:\n")
print('Momentum Rate =',momentum_rate,'\n')
print('Filters =',filters,'\n')
print('Batch_size =',batch_size,'\n')
print('Number of neurons =',neurons,'\n')
print("Training Data Statistics:\n")
print("Relu Hidden Units with Cross-Entropy Error Function")
neural_network_model('relu', 'categorical_crossentropy', X, label)
print("\n--------------------------------------------------------------\n")

print("Relu Hidden Units with Sum-of-Square Error Function")
neural_network_model('relu', 'mean_squared_error', X, label)
print("\n--------------------------------------------------------------\n")

print("Tanh Hidden Units with Cross-Entropy Error Function")
neural_network_model('tanh', 'categorical_crossentropy', X, label)
print("\n--------------------------------------------------------------\n")

print("\nTest Data Statistics:\n")

print("Relu Hidden Units with Cross-Entropy Error Function")
neural_network_model('relu', 'categorical_crossentropy', X1, label1)
print("\n--------------------------------------------------------------\n")

print("Relu Hidden Units with Sum-of-Square Error Function")
neural_network_model('relu', 'mean_squared_error', X1, label1)
print("\n--------------------------------------------------------------\n")

print("Tanh Hidden Units with Cross-Entropy Error Function")
neural_network_model('tanh', 'categorical_crossentropy', X1, label1)

#Question-2


#Second question is on experimenting hyper parameter values with CNN
#For running second question,change the parameters here
#best Hyper parameters for CNN
momentum_rate = 0.95
filters = 1024
batch_size = 100
learning_rate = 0.05
neurons = 200

def convolutional_neural_network(x, y):
    startTime = time.clock()
    model = Sequential()
    model.add(Conv2D(filters, (1, 1), input_shape=(1, 1, 64), activation='relu'))
    model.add(Flatten())
    model.add(Dense(neurons, activation='relu')) # first hidden layer
    model.add(Dense(neurons, activation='relu')) # second hidden layer
    model.add(Dense(neurons, activation='relu')) # third hidden layer
  #  model.add(Dense(neurons, activation='relu')) #fourth hidden layer
 #   model.add(Dense(neurons, activation='relu')) #fifth hidden layer
    model.add(Dense(10, activation='softmax'))
    sgd = optimizers.SGD(lr=learning_rate, decay=1e-6, momentum=momentum_rate, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.fit(cnn_train, Y, validation_split=0.2, epochs=10, batch_size=100, verbose=0)
    endTime = time.clock()
    print("Time = ", endTime-startTime)
    print_statistics(model, x, y)

print("Hyper-parameter values:\n")
print('Momentum Rate =',momentum_rate,'\n')
print('Filters =',filters,'\n')
print('Batch_size =',batch_size,'\n')
print('Number of neurons =',neurons,'\n')

print("\nTraining Data Statistics:\n")

print("CNN Model with Relu Hidden Units and Cross-Entropy Error Function:")
convolutional_neural_network(cnn_train, label)
print("\n--------------------------------------------------------------\n")

print("\nTest Data Statistics:\n")

print("CNN Model with Relu Hidden Units and Cross-Entropy Error Function:")
convolutional_neural_network(cnn_test, label1)
