# Implement a 2-layer (input layer, hidden layer and output layer) neural network
# from scratch for the XOR operation.
# This includes implementing forward and backward passes from scratch.
# The truth table for XOR is given by



import numpy as np
X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y = np.array([[0],[1],[1],[0]], dtype=float)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_derivative(x):
    return x*(1-x)

np.random.seed(2000)
# print("hi")
W1 = np.random.randn(2,2)
print("W1:",W1)
b1 = np.zeros((1,2))
print("b1:",b1)
W2 = np.random.randn(2,1)
print("W2:",W2)
b2 = np.zeros((1,1))
print("b2:",b2)

learning_rate = 0.5
# iteration = 10000

for epoch in range(10000):
    Z1 = np.dot(X,W1)+b1
    A1 = sigmoid(Z1)
    Z2 = np.dot(A1,W2)+b2
    A2 = sigmoid(Z2)

    loss = np.mean((y-A2)**2)

    error_output = y-A2
    delta_output = error_output*sigmoid_derivative(A2)
    error_hidden = np.dot(delta_output,W2.T)
    delta_hidden = error_hidden*sigmoid_derivative(A1)

    W2 = W2+np.dot(A1.T,delta_output)*learning_rate
    b2 = b2+np.sum(delta_output,axis=0,keepdims=True)*learning_rate
    W1 = W1+np.dot(X.T,delta_hidden)*learning_rate
    b1 = b1+np.sum(delta_hidden,axis=0,keepdims=True)*learning_rate

    if epoch%1000==0:
        print("Epoch:",epoch,"Loss:",loss)

print("\nXOR Predictions:")
hidden = sigmoid(np.dot(X,W1)+b1)
output = sigmoid(np.dot(hidden,W2)+b2)

for i in range(len(X)):
    prediction = 1 if output[i][0]>=0.5 else 0
    print("Input:",X[i].astype(int),"Actual:",int(y[i][0]),"Predicted:",prediction)