"""Part 1: Importing libraries
It imports the required libraries as well as the class that is written by me for neural network training
"""

import torch
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import random
from matplotlib import pyplot as plt
import seaborn as sns
from ML_FAB_Classes.NN_Prep import MLPNN_Regression
from scipy.io import loadmat
import pyxdf
from sklearn.preprocessing import MinMaxScaler
import os
"""Part 2:load the data object
This section loads the data from an xdf file and coverts it to a panda data frame format
"""
#### load the data object
xdfdata, header = pyxdf.load_xdf('AG_42.xdf')


y0=xdfdata[0]
y1=xdfdata[1]

if y0['info']['name']==['RightLeg']:
    t = y0['time_stamps']
    outp = y0['time_series']
    t0 = np.array(t)
    out0 = np.array(outp)
    t = y1['time_stamps']
    outp = y1['time_series']
    t1 = np.array(t)
    out1 = np.array(outp)
else:
    t = y0['time_stamps']
    outp = y0['time_series']
    t1 = np.array(t)
    out1 = np.array(outp)
    t = y1['time_stamps']
    outp = y1['time_series']
    t0 = np.array(t)
    out0 = np.array(outp)
t0=t0-t0[0]
t1=t1-t1[0]
out0_ready= np.delete(out0, [0,4,6,7], axis=1)
out1_ready = np.delete(out1, [0,4,6,7], axis=1)
panda_data0=pd.DataFrame(out0_ready,columns=["Joint Angle","Torque","Force sensor","State number"])
panda_data1=pd.DataFrame(out1_ready,columns=["Joint Angle","Torque","Force sensor","State number"])

"""Part 3:data initialization
This section removes noises from the data and also does hot encoder operation on the discreet features 
"""

new_features0 = {}
new_features1 = {}
panda_data0['State number']=round(panda_data0['State number'])
panda_data1['State number']=round(panda_data1['State number'])

#panda_data0['Torque']=round(panda_data0['Torque'])
#panda_data1['Torque']=round(panda_data1['Torque'])

for item in set(panda_data0['State number'].values):
    new_features0[item] = []
    new_features1[item] = []

for value in panda_data0['State number'].values:

    for key in new_features0:


        if value == key:
            new_features0[key].append(1)

        ### we need to add a zero to all other colors
        else:
            new_features0[key].append(0)
for value in panda_data1['State number'].values:

    for key in new_features1:


        if value == key:
            new_features1[key].append(1)

        ### we need to add a zero to all other colors
        else:
            new_features1[key].append(0)




for key in new_features0:
    key=key.astype(int)
    panda_data0[key] = new_features0[key]
    panda_data1[key] = new_features1[key]


panda_data0 = panda_data0.drop(['State number'], 1)
panda_data1 = panda_data1.drop(['State number'], 1)



#sns.pairplot(panda_data0,vars=["Joint Angle","Force sensor",10,11,12,13,14],hue="Torque",height=1,aspect=1)
#plt.show()

data=panda_data0[["Joint Angle","Torque","Force sensor",10,11,12,13,14]]
labels=panda_data0[["Torque"]]
"""Part 4: Normalization for ML
This section normalizes both the features and continuous labels  
"""
scalerd = MinMaxScaler()
scaled_data = scalerd.fit_transform(data)
scalerl = MinMaxScaler()
scaled_labels = scalerl.fit_transform(labels)
"""Part 5: Training
This section trains the neural networks and get the trained model  
"""

testclass = MLPNN_Regression(scaled_data,scaled_labels)
model = testclass.train_test(test_size=0.2,n_epochs=2,hidden_dimensions=20,batch_size=8,lr=0.02)

PATH= "modelnn.pt"
torch.save(model.state_dict(),PATH)
"""Part 6: Results and plots
This section shows how much error the model has based on all available data ( which includes the train and test data)
It also plots the labels of all available data and compare it with predicted labels come from the trained model
"""
scaled_prediction=model(torch.tensor(scaled_data.astype(np.float32)))
unscaled_pred=scalerl.inverse_transform(scaled_prediction.detach().numpy())
error_totall=np.sum(abs(unscaled_pred-labels))/len(unscaled_pred-labels)
plt.show()
plt.clf()
plt.plot(t0,unscaled_pred,label='Predictions')
plt.plot(t0,labels,label='Real data')
plt.legend()
print(error_totall)
plt.title("Plot of all data")
plt.show()
