# `dl/` — Basic Deep Learning

Neural-network building blocks from scratch (NumPy backprop) and in PyTorch.
This is also where most **training techniques** are demonstrated — see
[`../training-techniques/README.md`](../training-techniques/README.md).

| Sub-folder | Algorithms / focus |
|---|---|
| `mlp/` | feed-forward net + backprop; weight init; activations |
| `optimizers/` | SGD, Momentum, Nesterov, AdaGrad, RMSProp, Adam, AdamW |
| `regularization/` | L1/L2, dropout, batch/layer norm, early stopping |
| `cnn/` | conv/pool from scratch (LeNet); ResNet (skip connections), VGG, Inception, MobileNet |
| `rnn/` | vanilla RNN (**vanishing gradients**), LSTM, GRU, seq2seq |
| `autoencoder/` | vanilla / denoising / sparse / contractive |

Run any module directly, e.g. `python dl/mlp/mlp.py`.
