---
title: Inference in discrete state hidden markov models using numpy
tags: [numpy, jupyter]
layout: post
---

[This post is mostly a redirect to the notebook](https://github.com/jessstringham/blog/blob/master/notebooks/2018-05-02-HMM.ipynb).

In my latest attempts to translate the probability and statistics stuff I'm learning into code, I implemented exact inference for an HMM using `numpy`.

The idea was that someone was moving around a room, and given the bumps/creaks you hear, you need to predict where in the room they are.

![Three sets of 10 images, second row showing a dot moving around a grid, third row showing highlighted area where model thinks the dot is](/assets/2018-05-02-filtering.png)

The first row represents the bumps/creaks you hear. The second row represents where the person is. The third row represents where exact inference guesses where the person is.

Again, the good stuff is in [the notebook](https://github.com/jessstringham/blog/blob/master/notebooks/2018-05-02-HMM.ipynb).
