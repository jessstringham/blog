---
title: Bookmarklet
tags: [lil-post]
layout: post
---

Today I learned how to write a bookmarklet and wrote one that updates
the current page's url.  Bookmarklets seem like a great way to help
automate all the things.

Today I wanted to simplify my morning Gmail-checking workflow.  I
start with clicking one of the labels in the sidebar, then filtered
down to emails that are in my inbox or unread.

Before, I'd add something like `{label:unread label:inbox}` to the
search box and hit search. Now, I can click the bookmarklet to run the
filter.

The one I'm using looks something like this:

{% highlight javascript %}
    javascript:(function() {
        window.location=window.location.toString().replace(
            /^https:\/\/mail.google.com\/mail\/u\/0\/#label\/(.*)/,
            'https:\/\/mail.google.com\/mail\/u\/0\/#search\/label%3A$1+%7Blabel%3Ainbox+label%3Aunread%7D'
        );
    })()
{% endhighlight %}

This works great for my workflow, and I'm excited to have another tool
in my toolbox!

## See also

- [xkcd, is it worth the time?](https://xkcd.com/1205/)
- [bookmarklets](https://en.wikipedia.org/wiki/Bookmarklet)
