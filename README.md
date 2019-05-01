A convolutional neural network for recognizing handwritten characters from the MNIST data set.

Author: Adrian Alberto

Architect: Dr. Evangelos Yfantis (UNLV)

Instructions:
-------------

To compile on GNU/Linux:

`g++ mnist_neural_net.cpp -o cnn.out`

To train a new model:

`./cnn.out`

After each epoch, a model file called `epochN_correctNNNN.txt` will be created.

You can test this model by running:

`./cnn.out epochN_correctNNNN.txt`

After testing, you will be prompted to either continue training or show an example test sample.

```
Enter an index 0-9999 to test the model, or -1 to resume training.
INPUT: 64


     O##O#OOOOO::O::
    :##################
      ::::::O##########
                   ###O
                  :##O
                  ##O
          :      ###
         O####OO###O
         O###########OO
           OO##########:
             ###OO
            O##O
           O##:
          :##O
          ##O
         :##
        :##:
       :###
       ###
       ##O

Predicted: 7, Actual: 7
```
