---
title: Pasta machine processing images with numpy
tags: [python, numpy]
layout: post
display_image: /assets/2018-04-08-ex.png
---

![Tiled cat image](/assets/2018-04-08-lizzy-header.png)
Here's a quick blog post to celebrate turning in my last coursework for my master's.

There was this video of using a pasta machine to creatively split an image into four images. It felt like a `numpy` challenge, so I tried to faithfully reproduce it!

<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">Top breeder with nothing but a pasta machine <a href="https://t.co/74b5D1N7dY">pic.twitter.com/74b5D1N7dY</a></p>&mdash; RΛMIN NΛSIBOV (@RaminNasibov) <a href="https://twitter.com/RaminNasibov/status/981834971403911168?ref_src=twsrc%5Etfw">April 5, 2018</a></blockquote> <script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>


If I want the pasta width to be a pixel, the code is kind of cute:

<script src="https://gist.github.com/jessstringham/1fab85175358650242c57d99817ed413.js"></script>

![Tiled cat image](/assets/2018-04-08-lizzy-header.png)

But that doesn't quite look like the original, so here's how I could do it with a wider pasta setting.

<script src="https://gist.github.com/jessstringham/b2fe92a7f1412f00443b0fb124f08bd9.js"></script>

![Tiled cat image made up of blocks of the image](/assets/2018-04-08-lizzy.png)
