# recurrent_pytorch
This repository is destined to hold code for my RNN experiments in Pytorch (which will be learned in parallel), to be used for weird experiment ideas and eventually blog posts/papers/garbage.

### To-do:
- [X] Install Pytorch at home
- [X] Write input pipeline to shuffle pixels with a fixed random pattern
- [X] Write basic RNN
  - [X] Make initial state learnable
  - [X] Use layer norm for stability
- [X] Write GRU
- [X] Add zoneout
- [X] Baseline experiments with RNN & GRU
  - [X] Write function for plotting training curves during training
  - [X] Figure out how to save model checkpoints to enable reloading trained model
- [X] Post-training analysis of saved model
  - [X] Write function for visualizing the important pixels used for classification
  - [X] Write function to save collage of images w/ arbitrary printout below
     - [X] Make collages of images with lowest and highest loss
  - [X] Write function to generate gif to visualize test case:
    * On the left: the image appears one pixel at a time
    * On the right: the network outputs as a bar chart with the correct label highlighted

### Miscellaneous ideas:
- [ ] Can we learn intermediate labels (i.e. half-way through the sequence, something that starts to resemble true label) for additional supervision?
	* add fully-connected net to map intermediate hidden states to final states
	* use output weights from rnn to produce soft intermediate labels from the predicted final states for each intermediate hidden state
- [ ] Add minimalRNN (https://arxiv.org/abs/1711.06788)
- [ ] Active learning in noisy datasets where the model is allowed to reject up to a certain percentage of the training examples after seeing the loss (need to add corrupted labels)
	* Separate meta net takes model output and loss, produces reject probability
	* Start with high temperature (exploration) and reduce as learning progresses
- [ ] Are recurrent nets more or less susceptible to adversarial examples than convolutional nets?
    * Note: input is lower dimensionality, but recurrence may lead to exploitable instabilities (could go either way)
    - [ ] Try thermometer encoding (https://openreview.net/pdf?id=S18Su--CW)
- [ ] Add aleatoric uncertainty
      - [ ] Backprop uncertainty to pixels and highlight highly-confusing pixels in validation examples with highest uncertainty
- [ ] Surprise gate: the network predicts the next input from its hidden state & gives higher value to less predictable inputs
- [ ] Figure out how to obtain/plot information theoretic results:
    * Mutual information between hidden state and input/output/next hidden state
    * Does mutual information track with weight correlation/symmetry (between the input/output/recurrent weight vectors for each unit)?
    * Track redundancies in the hidden units throughout training, where redundancies are defined as 2*H(o|h_i,h_j)-H(o|h_i)-H(o|h_j) or maybe I(h_i;h_j)
    * How does the mutual information between hidden unit activations relate to the Hessian? Can we calculate the Hessian in the case where n_hidden is small? Measure "how diagonal" the Hessian is and compare to: redundancies, long-term memory, training speed
- [ ] Explore multiplicative terms and hypernets...
    * Implement hyperRNN and multiplicative RNN
    * Plot gradient stats and redundancies throughout training, then try to figure out what happened
- [ ] Experiment with evolutionary strategies for training
- [ ] Experiment with decoupled neural interfaces
