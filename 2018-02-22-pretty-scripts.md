---
title: Pretty scripts with Python's print statement
tags: [bash, python]
layout: post
display_image: 2018-02-22-ex.png
---

![List of experiment runs, with little checkboxes](/assets/2018-02-22-run-list.png)

For my personal projects, I found a place between "boring print statements that are easy to code" and "pretty GUI web app that requires a lot of set-up."
Even without reaching for [`curses`](https://docs.python.org/3/library/curses.html#module-curses), there are some neat things you can do with Python's print statement!

I used this last week for a school project that involved training a bunch of TensorFlow models. I threw together a script to help view results.

Disclaimer: Because I use this for personal scripts, I'm cheating here by not worrying about compatibility on devices I'm not using. Your results might vary!

## Terminal Colors

I can change the color of text in the terminal by adding a little weird character before and after the text I want to change. There are also options for bold or underlined text.

{% highlight python %}
print('\033[1m{}\033[0m'.format('This text is bold!'))
{% endhighlight %}

![Python terminal printing bold text](/assets/2018-02-22-bold-text.png)

Here's more details on how to do this in [Python](https://stackoverflow.com/questions/287871/print-in-terminal-with-colors).

For the deep learning experiments, I used it to highlight the epoch with the best validation accuracy in my experiment results.

![List of numbers with one row highlighted](/assets/2018-02-22-run-view.png)

One thing is that copy-pasting doesn't take the formatting with it, which is why I also print an asterisk.

## Emoji

With scripts I run often, I sometimes figure out the error message by the shape of the text block anyway. Inspired by Homebrew, I sometimes print out emoji/Unicode in my scripts.
 I started throwing in emojis, like 🔥, to tell me when things broke or not.

{% highlight python %}
print('✅')
{% endhighlight %}

My terminal messes up the spacing, but this is a hacky script that hopefully will never see the light of day, so I just added spaces.

For the deep learning experiments, I used it in my recent experiment list to mark which experiments had reached their early-stopping criteria.

![List of experiment runs, with little checkboxes](/assets/2018-02-22-run-list.png)



## See Also
 - Or you could use [`curses`](https://docs.python.org/3/library/curses.html#module-curses)
 - [TensorBoard](https://www.tensorflow.org/programmers_guide/summaries_and_tensorboard) might be the right way to do this.
 - Because I'm posting about Unicode: [The Absolute Minimum Every Software Developer Absolutely, Positively Must Know About Unicode and Character Sets (No Excuses!)](https://www.joelonsoftware.com/2003/10/08/the-absolute-minimum-every-software-developer-absolutely-positively-must-know-about-unicode-and-character-sets-no-excuses/)
