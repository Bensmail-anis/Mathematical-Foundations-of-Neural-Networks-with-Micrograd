from mlp import MLP
from micrograd_engine import Value
import random

n = MLP(5, [5, 5, 3])
inputs = [
  [8.0, 2.0, 3.0, -1.0,2.0],
  [4.6, 3.0, -1.0, 0.5 , 5.5],
  [3.3 ,0.5, 2.3, 1.0, 1.0],
  [1.0, 1.0, -1.0 , -1.0 , 0.2],
]

num_samples = 4
num_classes = 3


# Generate random one-hot encoded actuals
actuals = []
for _ in range(num_samples):
    one_hot = [0] * num_classes
    one_hot[random.randint(0, num_classes - 1)] = 1
    actuals.append([Value(data=value) for value in one_hot])


preds = [n(x) for x in inputs]


loss = sum(
    (actual - pred) ** 2
    for actual_list, pred_list in zip(actuals, preds)
    for actual, pred in zip(actual_list, pred_list)
)


for k in range(20):

  # forward pass
  preds = [n(x) for x in inputs]
  loss = sum((actual - pred) ** 2 for actual_list, pred_list in zip(actuals, preds) for actual, pred in zip(actual_list, pred_list))

  # backward pass
  for p in n.parameters():
    p.grad = 0.0
  loss.backward()

  # update
  for p in n.parameters():
    p.data += -0.01 * p.grad # optimizer , it updates the weights , remember that if the grad is negative , we have to increase the weight by a learning rate defined here as 0.01 , if the grad is positive , the weight must be decreased

  print(k, loss.data)

