import matplotlib.pyplot as plt
import numpy as np
import math

sine = np.array([math.sin(x) for x in np.arange(200)])

#sinusoidal values
print(sine)

plt.plot(sine[:50])
plt.show()

#creating the data the will be used to train RNN the network
X = []
Y = []

seq_len = 50
num_records = len(sine) - seq_len

for i in range(num_records - 50):
    X.append(sine[i:i+seq_len])
    Y.append(sine[i+seq_len])

X = np.array(X)
print(X.shape)
X = np.expand_dims(X, axis = 2)
print(X.shape)

Y = np.array(Y)
print(Y.shape)
Y = np.expand_dims(Y, axis=1)
print(Y.shape)


