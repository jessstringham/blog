---
title: SmallNORB notebook
tags: [numpy, jupyter]
layout: post
---

[Like the last post, this post is mostly a redirect to a notebook](https://github.com/jessstringham/blog/blob/master/notebooks/2018-05-03-SmallNORB.ipynb).

![Car rotating](/assets/2018-05-03-smallnorb.png)

For a project last semester, my team used the [SmallNORB](https://cs.nyu.edu/~ylclab/data/norb-v1.0-small/) dataset. We were implementing Capsule Networks and the SmallNORB dataset is potentially supposed to show how great Capsule Networks are at encoding pose!

I fought with `struct` and endianness, and finally found a few lines of Python code that would [parse the file](https://github.com/jessstringham/blog/blob/master/notebooks/2018-05-03-SmallNORB.ipynb) into a numpy array. I also explored the dataset a little and created an interactive widget that lets you rotate the little figures.


<img src="/assets/2018-05-03-instances.gif" width="300" alt="Cycling through images of toys from the SmallNORB dataset.">
<img src="/assets/2018-05-03-rotate.gif" width="300" alt="Cycling through images of rotating cars from the SmallNORB dataset.">


[Maybe check it out!](https://github.com/jessstringham/blog/blob/master/notebooks/2018-05-03-SmallNORB.ipynb)
