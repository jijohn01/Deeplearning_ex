import numpy as np
import sigmoid as sig

x1 = float(input("x1:"))
x2 = float(input("x2:"))

X = np.array([x1, x2])
W1 = np.array([[0.1, 0.3, 0.5],[0.2 , 0.4 , 0.6]])
B1 = np.array([0.1 , 0.2, 0.3])

A1 = np.dot(X,W1) + B1
#입력과 가중치를 곱하고 편향을 더함

Z1 = sig.sigmoid(A1)
print(A1)
print(Z1)

W2 = np.array([[0.1, 0.4],[0.2, 0.5],[0.3,0.6]])
B2 = np.array([0.1, 0.2])

A2 = np.dot(Z1,W2) + B2
Z2 = sig.sigmoid(A2)
print(A2)
print(Z2)
#은닉층 2

W3 = np.array([[0.1 , 0.2],[0.4, 0.2]])
B3 = np.array([0.2,0.1])

A3 = np.dot(Z2,W3) + B3
Y = A3 #sigma ()
print(Y)