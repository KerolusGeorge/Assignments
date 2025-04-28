import numpy as np


my_text = "cat chases mouse runs"
words = my_text.split()


unique_words = list(set(words))
word_to_num = {w:i for i,w in enumerate(unique_words)}
num_to_word = {i:w for i,w in enumerate(unique_words)}

input_seq = [word_to_num[w] for w in words[:-1]]
target_word = word_to_num[words[-1]]

def make_onehot(idx, size):
    vec = [0]*size
    vec[idx] = 1
    return vec

X = [make_onehot(i, len(unique_words)) for i in input_seq]
y = make_onehot(target_word, len(unique_words))


hidden_units = 4
np.random.seed(42)  

W_xh = np.random.randn(hidden_units, len(unique_words)) * 0.1
W_hh = np.random.randn(hidden_units, hidden_units) * 0.1
W_hy = np.random.randn(len(unique_words), hidden_units) * 0.1
b_h = np.zeros((hidden_units, 1))
b_y = np.zeros((len(unique_words), 1))

learning_rate = 0.05
epochs = 100
losses = []

for epoch in range(epochs):
    h_prev = np.zeros((hidden_units, 1))
    for t in range(len(X)):
        x_t = np.array(X[t]).reshape(-1, 1)
        h_t = np.tanh(np.dot(W_xh, x_t) + np.dot(W_hh, h_prev) + b_h)
        y_t = np.dot(W_hy, h_t) + b_y
        h_prev = h_t
    
    probs = np.exp(y_t) / np.sum(np.exp(y_t))
    loss = -np.log(probs[target_word])
    losses.append(loss[0])
    
    dy = probs
    dy[target_word] -= 1
    
    dWhy = np.dot(dy, h_t.T)
    dby = dy
    
    dh = np.dot(W_hy.T, dy)
    dWxh = np.zeros_like(W_xh)
    dWhh = np.zeros_like(W_hh)
    dbh = np.zeros_like(b_h)
    
    for t in reversed(range(len(X))):
        x_t = np.array(X[t]).reshape(-1, 1)
        dh_raw = (1 - h_t * h_t) * dh
        
        dbh += dh_raw
        dWxh += np.dot(dh_raw, x_t.T)
        dWhh += np.dot(dh_raw, h_prev.T)
        
        dh = np.dot(W_hh.T, dh_raw)
        h_prev = np.array(X[t-1]).reshape(-1, 1) if t > 0 else np.zeros((len(unique_words), 1))

    W_xh -= learning_rate * dWxh
    W_hh -= learning_rate * dWhh
    W_hy -= learning_rate * dWhy
    b_h -= learning_rate * dbh
    b_y -= learning_rate * dby

h = np.zeros((hidden_units, 1))
for word in input_seq:
    x = make_onehot(word, len(unique_words))
    x = np.array(x).reshape(-1, 1)
    h = np.tanh(np.dot(W_xh, x) + np.dot(W_hh, h) + b_h)

output = np.dot(W_hy, h) + b_y
predicted_idx = np.argmax(output)
print(f"Input sequence: {' '.join(words[:-1])}")
print(f"Predicted next word: {num_to_word[predicted_idx]}")
print(f"Actual next word: {words[-1]}")