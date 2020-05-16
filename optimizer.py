# ANN for handwritten digits classification using different optimizers implemented from scratch
# Nishchal M N    - PES1201701523
# Ravichandra K G - PES1201701581

import numpy as np
from sklearn.utils import shuffle
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import time
import warnings

%tensorflow_version 1.x

warnings.filterwarnings('ignore')

switch = 0

# For printing testing items using a switch
def printt(*argv):
  if(switch == 1):
    for args in argv:
      print(args)

# Training Data and Labels
mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=False)
train = mnist_data.test                     # currently taking only test images of the dataset which has 10000 images for demonstration
images, labels = train.images, train.labels
printt(images.shape,labels.shape)

index_0,index_1 = np.where(labels==0)[0], np.where(labels==1)[0]                 # picking the images of 2 digits(0 and 1) for binary classification
images_0,labels_0 = images[[index_0]], np.expand_dims(labels[[index_0]],axis=1)
images_1,labels_1   = images[[index_1]], np.expand_dims(labels[[index_1]],axis=1)

images = np.vstack((images_0,images_1))       # putting all the 0 and 1 images in one container
labels = np.vstack((labels_0,labels_1))       # similarly labels too
images,labels = shuffle(images,labels)

n_test, n_train = 70,100     # taking only 80 images for training as it takes time for computations

test_images, test_lables = images[:n_test, :], labels[:n_test, :]           # splitting accordingly
train_images, train_labels = images[n_test : n_test + n_train, :], labels[n_test : n_test + n_train, :]

printt(train_images.shape)
printt(train_labels.shape)


# Activation functions and their derivatives

def sigmoid(x):
    return 1 / (1 + np.exp(-1 * x))
def d_sigmoid(x):
    return sigmoid(x) * ( 1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)
def d_tanh(x):
    return 1 - np.tanh(x) ** 2 

def ReLu(x):
    positive = (x > 0.0) * 1.0
    return x * positive
def d_ReLu(x):
    positive = (x > 0.0) * 1.0
    return positive    

def elu(x):
    negative = (x <= 0) * 1.0
    less_zero = x * negative
    positive =  (x > 0) * 1.0
    greater_zero = x * positive
    final = 3.0 * (np.exp(less_zero) - 1) * less_zero     # alpha = 3 chosen for elu function
    return greater_zero + final
def d_elu(x):
    positive = (x > 0) * 1.0
    negative = (x <= 0) * 1.0
    temp = x * negative
    final = (3.0 * np.exp(temp)) * negative
    return (x * positive) + final


# Initializing the weights

w1 = np.random.randn(784,256) * 0.2     # multiplied by 0.2 so that weights are less than 1 and becomes computationally easier
w2 = np.random.randn(256,128) * 0.2
w3 = np.random.randn(128,1) * 0.2

weights = {'SGD' : [], 'Momentum' : [], 'RMSprop' : [], 'Adam' : []}

def SGD(w1, w2, w3):
  # Hyperparameters initialization
  eta = 0.0003
  total_cost = 0
  temp_cost = []

  for i in range(epochs):
      for image_index in range(len(train_images)):
          
          image = np.expand_dims(train_images[image_index], axis=0)
          image_label = np.expand_dims(train_labels[image_index], axis=1)
          printt(image_label, train_labels[image_index])      

          # Forward_propagation
          layer1 = image.dot(w1)
          layer1A = elu(layer1)

          layer2 = layer1A.dot(w2)
          layer2A = tanh(layer2)       

          layer3 = layer2A.dot(w3)
          layer3A = sigmoid(layer3)   

          printt(layer3A, image_label)

          cost = np.square(layer3A - image_label).sum() * 0.5
          total_cost = total_cost + cost

          # Backpropagation
          l3_step_1 = layer3A - image_label
          l3_step_2 = d_sigmoid(layer3)
          l3_step_3 = layer2A
          del_w3 = l3_step_3.T.dot(l3_step_1 * l3_step_2)    

          l2_step_1 = (l3_step_1 * l3_step_2).dot(w3.T)
          l2_step_2 = d_tanh(layer2)
          l2_step_3 = layer1A
          del_w2 = l2_step_3.T.dot(l2_step_1 * l2_step_2)

          l1_step_1 = (l2_step_1 * l2_step_2).dot(w2.T)
          l1_step_2 = d_elu(layer1)
          l1_step_3 = image
          del_w1 = l1_step_3.T.dot(l1_step_1 * l1_step_2)

          # Weight_updation
          w3 = w3 - eta * del_w3
          w2 = w2 - eta * del_w2
          w1 = w1 - eta * del_w1
      
      if i % 10 == 0 :
          print("SGD Itereration {}, total error: {}".format(i, total_cost))
      temp_cost.append(total_cost)
      total_cost = 0
  cost_array.append(temp_cost)
  
  weights['SGD'].extend([w1,w2,w3])


def Momentum(w1,w2,w3):
  # Hyperparameters initialization
  eta = 0.0009
  prev1, prev2, prev3 = 0,0,0
  alpha = 0.001
  total_cost = 0
  temp_cost = []

  for i in range(epochs):
      for image_index in range(len(train_images)):
          
          image = np.expand_dims(train_images[image_index],axis=0)
          image_label = np.expand_dims(train_labels[image_index],axis=1)

          # Forward_propagation
          layer1 = image.dot(w1)
          layer1A = elu(layer1)

          layer2 = layer1A.dot(w2)
          layer2A = tanh(layer2)       

          layer3 = layer2A.dot(w3)
          layer3A = sigmoid(layer3)   

          cost = np.square(layer3A - image_label).sum() * 0.5
          total_cost = total_cost + cost

          # Backpropogation
          l3_step_1 = layer3A - image_label
          l3_step_2 = d_sigmoid(layer3)
          l3_step_3 = layer2A
          del_w3 = l3_step_3.T.dot(l3_step_1 * l3_step_2)    

          l2_step_1 = (l3_step_1 * l3_step_2).dot(w3.T)
          l2_step_2 = d_tanh(layer2)
          l2_step_3 = layer1A
          del_w2 = l2_step_3.T.dot(l2_step_1 * l2_step_2)

          l1_step_1 = (l2_step_1 * l2_step_2).dot(w2.T)
          l1_step_2 = d_elu(layer1)
          l1_step_3 = image
          del_w1 = l1_step_3.T.dot(l1_step_1 *l1_step_2)

          # Weight_updation
          prev3 = prev3 * alpha + eta * del_w3
          prev2 = prev2 * alpha + eta * del_w2
          prev1 = prev1 * alpha + eta * del_w1

          w3 = w3 - prev3
          w2 = w2 - prev2
          w1 = w1 - prev1
      if i % 10 == 0 :
          print("Momentum Itereration {}, total error: {}".format(i, total_cost))
      temp_cost.append(total_cost)
      total_cost = 0
  cost_array.append(temp_cost)

  weights['Momentum'].extend([w1,w2,w3])
  

def RMSprop(w1,w2,w3):
  # Hyperparameters initialization
  eta = 0.0003
  RMSprop_1, RMSprop_2, RMSprop_3 = 0, 0, 0
  RMS_beta = 0.9
  RMS_epsilon = 0.00000001
  total_cost = 0
  temp_cost = []

  for i in range(epochs):
      for image_index in range(len(train_images)):
          
          image = np.expand_dims(train_images[image_index], axis=0)
          image_label = np.expand_dims(train_labels[image_index], axis=1)

          # Forward_propagation
          layer1 = image.dot(w1)
          layer1A = elu(layer1)

          layer2 = layer1A.dot(w2)
          layer2A = tanh(layer2)       

          layer3 = layer2A.dot(w3)
          layer3A = sigmoid(layer3)   

          cost = np.square(layer3A - image_label).sum() * 0.5
          total_cost = total_cost + cost

          # Backpropagation
          l3_step_1 = layer3A - image_label
          l3_step_2 = d_sigmoid(layer3)
          l3_step_3 = layer2A
          del_w3 =  l3_step_3.T.dot(l3_step_1 * l3_step_2)    

          l2_step_1 = (l3_step_1 * l3_step_2).dot(w3.T)
          l2_step_2 = d_tanh(layer2)
          l2_step_3 = layer1A
          del_w2 = l2_step_3.T.dot(l2_step_1 * l2_step_2)

          l1_step_1 = (l2_step_1 * l2_step_2).dot(w2.T)
          l1_step_2 = d_elu(layer1)
          l1_step_3 = image
          del_w1 = l1_step_3.T.dot(l1_step_1 *l1_step_2)

          # Weight updation
          RMSprop_3 = RMS_beta * RMSprop_3 + (1 - RMS_beta) * del_w3**2
          RMSprop_2 = RMS_beta * RMSprop_2 + (1 - RMS_beta) * del_w2**2
          RMSprop_1 = RMS_beta * RMSprop_1 + (1 - RMS_beta) * del_w1**2

          w3 = w3 - (eta / np.sqrt(RMSprop_3 + RMS_epsilon)) * del_w3
          w2 = w2 - (eta / np.sqrt(RMSprop_2 + RMS_epsilon)) * del_w2
          w1 = w1 - (eta / np.sqrt(RMSprop_1 + RMS_epsilon)) * del_w1
      if i % 10 == 0 :
          print("RMSprop Iteration {}, total error: {}".format(i, total_cost))
      temp_cost.append(total_cost)
      total_cost = 0
  cost_array.append(temp_cost)

  weights['RMSprop'].extend([w1,w2,w3])


def Adam(w1,w2,w3):
  # Hyperparameter Initialization
  eta = 0.0003
  Adam_m_1,Adam_m_2,Adam_m_3 = 0, 0, 0
  Adam_v_1,Adam_v_2,Adam_v_3 = 0, 0, 0
  Adam_Beta_1 ,Adam_Beta_2 = 0.9, 0.999
  Adam_epsilon = 0.00000001
  total_cost = 0
  temp_cost = []

  for i in range(epochs):
      for image_index in range(len(train_images)):
          
          image = np.expand_dims(train_images[image_index], axis=0)
          image_label = np.expand_dims(train_labels[image_index], axis=1)

          # Forward_propagation
          layer1 = image.dot(w1)
          layer1A = elu(layer1)

          layer2 = layer1A.dot(w2)
          layer2A = tanh(layer2)       

          layer3 = layer2A.dot(w3)
          layer3A = sigmoid(layer3)   

          cost = np.square(layer3A - image_label).sum() * 0.5
          total_cost = total_cost + cost

          l3_step_1 = layer3A - image_label
          l3_step_2 = d_sigmoid(layer3)
          l3_step_3 = layer2A
          del_w3 = l3_step_3.T.dot(l3_step_1 * l3_step_2)    

          l2_step_1 = (l3_step_1 * l3_step_2).dot(w3.T)
          l2_step_2 = d_tanh(layer2)
          l2_step_3 = layer1A
          del_w2 =    l2_step_3.T.dot(l2_step_1 * l2_step_2)

          l1_step_1 = (l2_step_1 * l2_step_2).dot(w2.T)
          l1_step_2 = d_elu(layer1)
          l1_step_3 = image
          del_w1 =   l1_step_3.T.dot(l1_step_1 *l1_step_2)

          Adam_m_3 = Adam_Beta_1 * Adam_m_3 + ( 1 - Adam_Beta_1 ) * del_w3
          Adam_m_2 = Adam_Beta_1 * Adam_m_2 + ( 1 - Adam_Beta_1 ) * del_w2
          Adam_m_1 = Adam_Beta_1 * Adam_m_1 + ( 1 - Adam_Beta_1 ) * del_w1

          Adam_v_3 = Adam_Beta_2 * Adam_v_3 + ( 1 - Adam_Beta_2 ) * del_w3 ** 2 
          Adam_v_2 = Adam_Beta_2 * Adam_v_2 + ( 1 - Adam_Beta_2 ) * del_w2 ** 2 
          Adam_v_1 = Adam_Beta_2 * Adam_v_1 + ( 1 - Adam_Beta_2 ) * del_w1 ** 2 
          
          Adam_m_3_hat = Adam_m_3 / ( 1 - Adam_Beta_1)
          Adam_m_2_hat = Adam_m_2 / ( 1 - Adam_Beta_1)
          Adam_m_1_hat = Adam_m_1 / ( 1 - Adam_Beta_1)
          
          Adam_v_3_hat = Adam_v_3 / ( 1 - Adam_Beta_2)
          Adam_v_2_hat = Adam_v_2 / ( 1 - Adam_Beta_2)
          Adam_v_1_hat = Adam_v_1 / ( 1 - Adam_Beta_2)
          
          w3 = w3 - (eta / (np.sqrt(Adam_v_3_hat) + Adam_epsilon)) * Adam_m_3_hat
          w2 = w2 - (eta / (np.sqrt(Adam_v_2_hat) + Adam_epsilon)) * Adam_m_2_hat
          w1 = w1 - (eta / (np.sqrt(Adam_v_1_hat) + Adam_epsilon)) * Adam_m_1_hat
          
      if i % 10 == 0 :
          print("Adam Iteration {}, total error: {}".format(i, total_cost))
      temp_cost.append(total_cost)
      total_cost = 0
  cost_array.append(temp_cost)

  weights['Adam'].extend([w1,w2,w3])


def test(optimizer):
  w1 = weights[optimizer][0]
  w2 = weights[optimizer][1]
  w3 = weights[optimizer][2]
  ans=0
  for image_index in range(len(test_images)):
      
          image = np.expand_dims(test_images[image_index],axis=0)
          image_label = np.expand_dims(test_lables[image_index],axis=1)   

          layer1 = image.dot(w1)
          layer1A = elu(layer1)

          layer2 = layer1A.dot(w2)
          layer2A = tanh(layer2)       

          layer3 = layer2A.dot(w3)
          layer3A = sigmoid(layer3)

          if(layer3A > 0.5):
            if(image_label[0][0] == 1):
              ans+=1
          else: 
            if(image_label[0][0] == 0):
              ans+=1
              
  print('accuracy given by {} is : {}'.format(optimizer, (ans/len(test_images))*100))
  printt(ans, len(test_images))

# main 
cost_array = []
epochs = 100

print('-----------')
SGD(w1,w2,w3)
test('SGD')

print('-----------')
Momentum(w1,w2,w3)
test('Momentum')

print('-----------')
RMSprop(w1,w2,w3)
test('RMSprop')

print('-----------')
Adam(w1,w2,w3)
test('Adam')

# plotting graph
colors = ['red', 'blue', 'green', 'black']
opts = ['SGD', 'Momentum', 'RMRprop', 'Adam']

for i in range(len(cost_array)):
    plt.plot(np.arange(100), cost_array[i], color = colors[i], linewidth = 3, label=str(opts[i]))

plt.title("Total Cost per Training")
plt.xlabel('No_of_epochs', fontsize=18)
plt.ylabel('Error', fontsize=16)
plt.legend()
plt.show()