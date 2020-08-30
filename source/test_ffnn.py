# import PyTorch
import torch
# import Feed Forward Neural Network class from nn_simple module
from nn_simple import ffnn
import pandas as pd

# sample input and output value for training
X = pd.read_csv('advertising.csv', usecols=[0,1,2]) # Read columns 'TV', 'Radio' and 'Newspaper' in dataset
X = X.values.tolist() # Convert dataframe to list
X = torch.tensor(X[:20], dtype=torch.float) # Convert to tensor pytorch, get 20 head rows

y = pd.read_csv('advertising.csv', usecols=[3]) # Read columns 'Sales' in dataset
y = y.values.tolist() # Convert dataframe to list
y = torch.tensor(y[:20], dtype=torch.float) # Convert to tensor pytorch, get 20 head rows


# scale units by max value
X_max, _ = torch.max(X, 0)
X = torch.div(X, X_max)
y = y / 100  # for max test score is 100


input_shape = X.shape[1] # Input shape
# create new object of implemented class
NN = ffnn.FFNeuralNetwork(input_size = input_shape, hidden_size = 20, activation_func = 'tanh')

# trains the NN 1,000 times
for i in range(1000):
    # print mean sum squared loss
    print("#" + str(i) + " Loss: " + str(torch.mean((y - NN(X)) ** 2).detach().item()))
    # training with learning rate = 0.1
    NN.train(X, y, 0.1)
# save weights
NN.save_weights(NN, "NN")

# load saved weights
NN.load_weights("NN")

# sample input x for predicting
x_predict = torch.tensor(([45, 39, 45]), dtype=torch.float)  # 1 X 3 tensor

# scale input x by max value
x_predict_max, _ = torch.max(x_predict, 0)
x_predict = torch.div(x_predict, x_predict_max)

# predict x input
NN.predict(x_predict)
