import layer

apple = 100
apple_num = 2
tax = 1.1

layer_1 = layer.MulLayer()
layer_2 = layer.MulLayer()

z1 = layer_1.forward(apple, apple_num)
z2 = layer_2.forward(z1, tax)

print(z2)