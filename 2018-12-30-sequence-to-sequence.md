---
title: Illustration of an Encoder-Decoder Sequence-to-Sequence neural network
tags: [ML]
layout: post
display_image: /assets/2018-12-30-ex1.png
---

In my [master's thesis]({% post_url 2018-12-30-thesis %}), I worked with encoder-decoder sequence-to-sequence neural networks. I'm cross-posting this diagram and description I use to describe seq2seq in the context of lemmatization for a midterm report.

## Diagram

The diagram is executed from bottom-to-top. Dark squares represent the intermediate vectors. White rectangles represent some function, which may use adjacent vectors depending on the implementation. The "time" arrow is a misnomer for steps in the sequence.

<img src='/assets/2018-12-30-seq2seq.png'>

Sketch of an encoder-decoder sequence-to-sequence model with attention which lemmatizes the input 'leaves' to produce the output 'leaf.' The data used to generate the character 'l' in the output is highlighted.

(1, input) The input sequence, the word in the case of lemmatization, is broken into the input representation such as one-hot encodings of individual characters. Additional information about the word, such as the part-of-speech, could also be passed in.

(2, input embedding) The input sequence is passed through an embedding mechanism to become a continuous-valued vector.

(3, encoding) The embedded values is then passed into the Encoder. For example, the encoder could be a bi-directional LSTM or convolutional layers.  The result of this could be a sequence of vectors.

(4, attention) In this example, the attention mechanism requires the string is fully-processed before the system begins decoding. For each decoder timestep, an Attention mechanism may use the encoded output vectors to create a mask to mask out some of encoder output vectors. This step is sometimes plotted as because it's a little interpretable.

<img src='/assets/2018-12-30-attn.png'>

(5, decoding) The resulting masked encoded output is passed to a Decoder layer. This could be another LSTM. This step could also take in the output of the previous timestep (for example, it may be passed 'l' when generating 'e').

(6, output embedding) The decoded vectors pass through another embedding layer to map back to characters.

(7, output) The decoding component can continue producing decoded output until the output creates a special stop character.
The final result is the output sequence!

## See Also

Many thanks go to Antreas Antoniou for explaining sequence-to-sequence to me! I think this diagram is basically vectorizing what he drew!
