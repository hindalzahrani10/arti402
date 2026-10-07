# ARTI 402 — Deep Learning
**IAU | College of Computer Science and Information Technology**

**Computer Engineering Department**
 
Lab material for ARTI 402. Each folder contains the notebook and any data files for that week's lab.
 
## Course overview
 
This lab builds deep learning skills from the ground up — starting with neural networks implemented from scratch in NumPy, then moving to TensorFlow and PyTorch. Topics include feedforward networks and back-propagation, CNNs and transfer learning, RNNs/LSTMs/GRUs, optimization and regularization, autoencoders, attention, and generative models (VAEs, GANs). The focus is on understanding what happens inside a model, then training and debugging real ones with modern frameworks.
 
## Labs
 
| Lab | Topic | Week |
|---|---|---|
| [lab01](./lab01) | From a Single Neuron to a Layer | 1 |
| [lab02](./lab02) | Activations, Loss, and How a Network Learns | 2 |
| [lab03](./lab03) | CNN Architecture | 3 |
| [lab04](./lab04) | RNN & LSTM Architecture | 4 |
| [lab05](./lab05) | Optimization Algorithms | 5 |
| [lab06](./lab06) | Optimization Algorithms (cont.)| 6 |
<!-- | [lab07](./lab07) | Topic| 7 | -->




Each lab folder contains everything that lab needs. Download the **whole folder**, not just the notebook.
  
## How to use this repo
 
**Students** — your instructor will share the link to the relevant lab folder each week. Clone or download that folder, work through the notebook, and submit via your own repository as instructed.


## Interactive tools
 
- **[The Analog Perceptron](https://msalmazyad.github.io/analog-perceptron/)** — Rosenblatt's perceptron as a physical machine: toggle switches are the inputs, knobs are the weights, and an analog needle reads the weighted sum. Train it, step through it one knob at a time, see the decision boundary it draws, and watch the 3-D error surface trace the path the run actually took. Three boards: 2 inputs (AND, OR, NAND, NOR — and not XOR), 4 x 4 inputs (X vs O, T vs J), and 2 layers (XOR, trained with backpropagation). One self-contained HTML file, interface in English and Arabic.

  Pair it with **lab01** — the single neuron, weights and biases as knobs, the step function, and a layer — and with **lab02** — why one layer cannot solve XOR, gradient descent, and backpropagation.

  Inspired by Welch Labs, [*ChatGPT is made from 100 million of these*](https://www.youtube.com/watch?v=l-9ALe3U-Fg) — worth watching before Lab 1; it is where the physical board comes from.

## Reference books
 
- Kinsley, H. & Kukieła, D. — *Neural Networks from Scratch in Python*
- Ekman, M. — *Learning Deep Learning*

---
