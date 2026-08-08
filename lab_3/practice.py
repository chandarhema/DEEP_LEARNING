import numpy as np
# X=np.random.randint(1,10,4)
X=np.array([5,6,7,10])
print(X)
print(np.shape(X))
W=np.array([[0.1,0.1,0.2],
            [1.1,0.4,1.1],
            [0.2,0.3,0.1],
            [0.7,0.5,1.2]])
W1=np.array([[0.1,0.2],
             [0.1,0.2],
             [0.3,0.1]])
W2=np.array([[0.2],
             [0.1]])
b=1
def relu(x):
    return np.maximum(0,x)

#for one input and one output
print("======= for one input and one output ======")
XX=np.array([0.1,0.15,0.2,0.3])
WW=np.array([0.07,0.05,-0.12,0.01])
summation=np.dot(X,W)
fu=relu(summation)
print(fu)

print("======= for one input, two hidden layers & one output ======")
hidden_layer_1=[]
for i in range(W.shape[1]):
    s=0
    for j in range(W.shape[0]):
        s+=X[i]*W[j][i]
    hidden_layer_1.append(relu(s))
print('hidden_layer_1')
print(hidden_layer_1)
hidden_layer_2=[]
for i in range(W1.shape[1]):
    s=0
    for j in range(W1.shape[0]):
        s+=hidden_layer_1[i]*W1[j][i]
    hidden_layer_2.append(relu(s))
print('hidden_layer_2')
print(hidden_layer_2)
out_put=[]
for i in range(W2.shape[1]):
    s=0
    for j in range(W2.shape[0]):
        s+=hidden_layer_2[i]*W2[j][i]
    out_put.append(relu(s))
print('out_put')
print(out_put)