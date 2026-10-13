# PPO Loss Notes

## Value Loss
Use MSE — difference between expected return and real return squared:

```
L = [V(s) - G_t]^2
```

## Policy Loss
Since we output a probability distribution over actions, MSE doesn't apply.

**Step 1 — compute the ratio** between new and old action probability for the same action:

```
r_t = π_new(a_t | s_t) / π_old(a_t | s_t)
```

**Step 2 — clip it:**

```
CLIP(r_t, 1-eps, 1+eps)
```

**Step 3 — take the min:**

```
min(r_t * A_t, CLIP(r_t) * A_t)
```

This stops rewards from getting too large, but allows negatives to continue (clipping only meaningfully activates on large positive updates).

## Intuition
If `r_t > 1`, the new policy increased the likelihood of action `a_t`. If the advantage was large, we want to increase that likelihood further — maximizing the ratio maximizes the expected return.

Simple: **increase the probabilities of actions that lead to good outcomes.**

---

## Original Numpy Implementation (too lazy  for backprop)

```python
class PolicyNetwork:
    def __init__(self, input_size, output_size):
        rng = np.random.default_rng()
        self.W1 = rng.random((input_size, 64)) * 0.01
        self.W2 = rng.random((64,64))* 0.01
        self.W3 = rng.random((64,output_size))* 0.01
        self.b1 = rng.random((64,))* 0.01
        self.b2 = rng.random((64,))* 0.01
        self.b3 = rng.random((output_size,))* 0.01
    def forward(self, input):
        H1 =  input @ self.W1 + self.b1
        Z1 = self.relu(H1)
        H2  = Z1 @ self.W2 + self.b2
        Z2 = self.relu(H2)
        output = Z2 @ self.W3 + self.b3
        probs = self.softmax(output)
        return probs
    def relu(self, layer):
        return np.maximum(0, layer)
    def softmax(self, layer):
        e_x = np.exp(layer - np.max(layer))
        return e_x / e_x.sum()

class ValueNetwork:
    def __init__(self, input_size):
        rng = np.random.default_rng()
        self.W1 = rng.random((input_size, 64)) * 0.01
        self.W2 = rng.random((64,64))* 0.01
        self.W3 = rng.random((64,1))* 0.01
        self.b1 = rng.random((64,))* 0.01
        self.b2 = rng.random((64,))* 0.01
        self.b3 = rng.random((1,))* 0.01
    def forward(self, input):
        H1 =  input @ self.W1 + self.b1
        Z1 = self.relu(H1)
        H2  = Z1 @ self.W2 + self.b2
        Z2 = self.relu(H2)
        output = Z2 @ self.W3 + self.b3
        return output.squeeze()
    def relu(self, layer):
        return np.maximum(0, layer)
```
