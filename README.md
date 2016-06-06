# Weighted Multi-label Binary Cross-entropy Criterion

Repository containing the 'one-versus-all logistic loss' function described in the paper "[Learning Visual Features from Large Weakly Supervised Data](http://arxiv.org/abs/1511.02251)".

It's only been minimally tested,  but I couldn't really find any existing code so it might be useful for someone else.

## Usage:

```lua
criterion = nn.WeightedMultiLabelBinaryCrossEntropyCriterion([label_count_tensor], [dataset_size])
```

As with any Torch loss: by default, the losses are averaged over observations for each minibatch. However, if the field `sizeAverage` is set to `false`, the losses are instead summed for each minibatch.


## Example:

```Lua
N = 100 # 100 Images
C = 10 # 10 Classes

labels = torch.Tensor(N,C):uniform() # Label is one when class is present, 0 when not.

NNK = torch.sum(labels, 1) # Occurence counts for each label, normally you'd obtain these from training set

criterion = nn.WeightedMultiLabelBinaryCrossEntropyCriterion(NNK, N)
```

