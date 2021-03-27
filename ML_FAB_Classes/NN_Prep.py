import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
import torch.optim as optim
import numpy as np
from matplotlib import pyplot as plt
class MLPNN_Regression:

        
 
    
    def __init__(self,data,labels):
        
        self.data = data
        self.labels = labels
        
      
 
    
    class Net(nn.Module):
        
        def __init__(self,n_features,hidden_dimension,n_classes):
        
            ##### calling the constructor of the parent class - gets everything we need from Pytorch
            super(MLPNN_Regression.Net, self).__init__()
            
            ''' When dealing with nn.Linear, the first input is the size of the input data,
            and the second input is how big you want the next layer to be '''
            
            ### The data enters here, then we make the next layer (hidden neurons)
            self.input_layer = nn.Linear(n_features,hidden_dimension)
            
            ### hidden layer #1
            self.layer1 = nn.Linear(hidden_dimension,hidden_dimension)
            
            ### hidden layer #2
            self.layer2 = nn.Linear(hidden_dimension, hidden_dimension)
            
            ### The output layer, where we end up with a series of nodes corresponding to each of our uniquelabels
            self.output_layer = nn.Linear(hidden_dimension,n_classes)
            
          
            self.relu = nn.ReLU()
       
    
    
        def forward(self,batch):
            
            ## put the data into the input layer of the neural network
            batch = self.input_layer(batch)
            
            batch = self.relu(batch)
            batch -= batch.min(1, keepdim=True)[0]
            batch /= batch.max(1, keepdim=True)[0]
            ## put the transformed data into the first hidden layer of the neural network
            batch = self.layer1(batch)
            
            ## apply the ReLU function to the output of the 1st hidden layer
            batch = self.relu(batch)
            batch -= batch.min(1, keepdim=True)[0]
            batch /= batch.max(1, keepdim=True)[0]
            ## put the transformed data into the second hidden layer of the neural network
            batch = self.layer2(batch)
            
            
            ## apply the ReLU function to the output of the 1st hidden layer
            batch = self.relu(batch)
            batch -= batch.min(1, keepdim=True)[0]
            batch /= batch.max(1, keepdim=True)[0]
            ## put the transformed data into the output layer of the neural network
            batch = self.output_layer(batch)
            
            ### return the probability distribution via the softmax function
            #return nn.functional.softmax(batch)
            return nn.functional.tanh(batch)
    

        
    def train_test(self,test_size,n_epochs,hidden_dimensions,batch_size,lr):
            
        ### splitting the data into a training/testing set
        train_data,test_data,train_labels,test_labels = train_test_split(self.data,self.labels, test_size=test_size)
        
        ## creating the batches using the batchify function
        train_batches,train_label_batches = batchify(train_data,train_labels,batch_size=batch_size)
        
    
        # Vahid
        neural_network = MLPNN_Regression.Net(len(train_data[0]),hidden_dimensions,1)
        
        
     
        optimizer = optim.Adam(neural_network.parameters(), lr=lr)
        
        
      
        loss_function = nn.MSELoss()
        
                
        neural_network.train()
        
        
        ''' This loop moves through the data once for each epoch'''
        for i in range(n_epochs):
            
            ### track the number we get correct
            correct = 0
            
            ''' This loop moves through each batch and feeds into the neural network'''
            for ii in range(len(train_batches)):
                
                ''' 
                Clears previous gradients from the optimizer - the optimizer,
                in this case, does not need to know what happened last time
                '''
                optimizer.zero_grad()
                
                
                batch = train_batches[ii]
                labels = train_label_batches[ii]

                
                ''' 
                Puts our batch into the neural network after converting it to a tensor
                
                Pytorch wants numeric data to be floats, so we will convert to a float as well 
                using np.float32
                
                              '''
                predictions = neural_network(torch.tensor(batch.astype(np.float32)))
                
                
                ''' 
                We put our probabilities into the loss function to calculate the error for this batch
                
                '''
                
                loss = loss_function(predictions,torch.tensor(labels.astype(np.float32)))
                #display(loss)
                '''
                loss.backward calculates the partial derivatives that we need to optimize
                '''
                loss.backward()
                
                
                '''
                optimizer step calculates the weight updates so the neural network can update the weights 
                '''
                optimizer.step()
                
                
                '''
                We extract just the data from our predictions, not other stuff Pytorch includes in that object
                
                We can then use the argmax function to figure out which index corresponds to the highest probability.
                If it is the 0th index, and the label is zero, we add one to correct. 
                If it is the 1st index, and the label is one, we add one to correct.
                
                This is why the labels need to start at zero and increase sequentially!
                '''
                
                pred=predictions.data.detach().numpy()  
                error_p=np.sum(abs((labels- pred)))/len(pred)
                correct = 1-error_p
                    #display(correct)
                      
         
            print("Accuracy for Epoch # " + str(i) + ": " + str(correct))

        print()
        
        plt.clf()
        plt.plot(predictions.data, '--', label='Training data', alpha=0.5)
        plt.plot(labels, '-', label='Predictions', alpha=0.5)
        plt.legend()
        plt.show(block=False)   
        plt.title("Plot of Trained data")             
        '''
        The eval function tells the neural network that it is about to be tested on blind test data
        and shouldn't change any of its internal parameters
        
        This function should always be called before eval
        '''
        neural_network.eval()
        
        test_correct = 0
        
        ''' input our test data into the neural network'''
        predictions = neural_network(torch.tensor(test_data.astype(np.float32)))
        
        ''' this checks how many we got right - very simple!'''
        pred=predictions.data.detach().numpy()  
        error_p=np.sum(abs((test_labels- pred)))/len(pred)
        test_correct = 1-error_p
                    
        print("Accuracy on test set: " + str(test_correct))
        plt.clf()
        plt.plot(predictions.data, '--', label='Test data', alpha=0.5)
        plt.plot(test_labels, '-', label='Predictions', alpha=0.5)
        plt.legend()
        plt.show(block=False)
        
        plt.title("Plot of Test data")
        return neural_network
        
   


''' Utility Function - function to turn the data into batches'''

def batchify(data,labels,batch_size=16):
    
    batches= []
    label_batches = []


    for n in range(0,len(data),batch_size):
        if n+batch_size < len(data):
            batches.append(data[n:n+batch_size])
            label_batches.append(labels[n:n+batch_size])

    if len(data)%batch_size > 0:
        batches.append(data[len(data)-(len(data)%batch_size):len(data)])
        label_batches.append(labels[len(data)-(len(data)%batch_size):len(data)])
        
    return batches,label_batches