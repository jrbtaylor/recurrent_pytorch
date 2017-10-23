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
* Figure out how to obtain/plot information theoretic results:
  * Mutual information between hidden state and input/output/next hidden state
  * Does mutual information track with weight correlation/symmetry (between the input/output/recurrent weight vectors for each unit)?
  * Track redundancies in the hidden units throughout training, where redundancies are defined as 2*H(o|h_i,h_j)-H(o|h_i)-H(o|h_j) or maybe I(h_i;h_j)
  * How does the mutual information between hidden unit activations relate to the Hessian? Can we calculate the Hessian in the case where n_hidden is small? Measure "how diagonal" the Hessian is and compare to: redundancies, long-term memory, training speed
* Explore multiplicative terms and hypernets...
  * Implement hyperRNN and multiplicative RNN
  * Plot gradient stats and redundancies throughout training, then try to figure out what happened
  * 
