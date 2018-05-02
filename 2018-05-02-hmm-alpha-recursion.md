---
title: Inference in discrete state hidden markov models using numpy
tags: [numpy, jupyter]
layout: post
---

[This post is mostly a redirect to the notebook](https://github.com/jessstringham/blog/blob/master/notebooks/2018-05-02-HMM.ipynb).

In my latest attempts to translate the probability and statistics stuff I'm learning into code, I made a demo that does exact inference using alpha recursion on an HMM.

The idea was that someone was moving around a room, and given the bumps/creaks you hear, you need to predict where in the room they are.

![Three sets of 10 images, second row showing a dot moving around a grid, third row showing highlighted area where model thinks the dot is](/assets/2018-05-02-filtering.png)

The first row represents the bumps/creaks you hear. The second row represents where the person actually is. The third row represents where alpha recursion guesses the person is.

[I go through implementation in a lot of detail in the notebook](https://github.com/jessstringham/blog/blob/master/notebooks/2018-05-02-HMM.ipynb).
