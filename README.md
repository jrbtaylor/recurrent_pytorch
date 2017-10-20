# recurrent_pytorch
This repository is destined to hold code for my RNN experiments in Pytorch (which will be learned in parallel), to be used for weird experiment ideas and eventually blog posts/papers/garbage.

### To-do:
* Install Pytorch at home
* Write input pipeline to perform basic augmentation and then shuffle pixels with a fixed random pattern
* Write basic RNN
  * Make initial state learnable
  * Use layer norm for stability
* Write function for plotting training curves
* Write function for visualizing the important pixels used for classification
* Write function to generate gif to visualize test case:
  * On the left: the image appears one pixel at a time
  * In the center: the hidden activations (reshaped to be a square-ish image)
  * On the right: the network outputs as a bar chart with the correct label highlighted
* Baseline experiments with RNN
* Write GRU
* Baseline experiments with GRU
* Surprise gate: the network predicts the next input from its hidden state & gives higher value to less predictable inputs
* Explore multiplicative terms and hypernets...
  * Implement hyperRNN and multiplicative RNN
  * Plot gradient stats throughout training and then try to figure out what happened
