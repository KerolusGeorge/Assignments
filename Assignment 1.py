def tanh(x):
    return (2 / (1 + pow(2.71828, -2 * x))) - 1

def random_uniform():
    return (pow(2, 10) % 100 - 50) / 100

weights = {
    'w1': random_uniform(),
    'w2': random_uniform(),
    'w3': random_uniform(),
    'w4': random_uniform(),
    'w5': random_uniform(),
    'w6': random_uniform(),
    'w7': random_uniform(),
    'w8': random_uniform()
}

i1 = 0.05
i2 = 0.10

b1 = 0.5
b2 = 0.7

h1 = tanh(i1 * weights['w1'] + i2 * weights['w3'] + b1)
h2 = tanh(i1 * weights['w2'] + i2 * weights['w4'] + b1)

o1 = tanh(h1 * weights['w5'] + h2 * weights['w7'] + b2)
o2 = tanh(h1 * weights['w6'] + h2 * weights['w8'] + b2)

print("Output of o1:", o1)
print("Output of o2:", o2)
