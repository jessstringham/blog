---
title: ‣ Unicode bullets in org-mode
tags: [emacs, org-mode]
layout: post
---

I enjoy trying to find tiny cosmetic changes to my environment that
completely transform how a frequent activity feels: like moving around
furniture, or changing my default Emacs font (❤︎ Source Code Pro).
Emacs and org-mode provide plenty of opportunities for little tweaks.
I'm sharing one of my favorites for this week's mini-post: replacing
plain list's hyphen bullets with unicode triangular bullets.

([I first found this idea (here)](http://www.howardism.org/Technical/Emacs/orgmode-wordprocessor.html))

When taking notes or exploring a problem, I end up with a lot of plain
lists in org-mode:

    Tips
     - do this
     - and that
       - but by *that*, I don't mean _that_
     - but maybe do it anyway

The dashes are fine, but a bit dull.  After adding the following code to my init.el

{% highlight lisp %}
(font-lock-add-keywords 'org-mode
                        '(("^ +\\([-*]\\) "
                           0 (prog1 () (compose-region (match-beginning 1) (match-end 1) "‣")))))
{% endhighlight %}

my list becomes

    Tips
    ‣ do this
    ‣ and that
      ‣ but by *that*, I don't mean _that_
    ‣ but maybe do it anyway

(to be honest, I like it more in the Source Code Pro font I use. The triangles are more equilateral.)

I added another one that changes `->` into `→`.

{% highlight lisp %}
(font-lock-add-keywords 'org-mode
                        '(("^ +\\([-*]\\) "
                           0 (prog1 () (compose-region (match-beginning 1) (match-end 1) "‣")))
                          ("\\(->\\)"
                           0 (prog1 () (compose-region (match-beginning 1) (match-end 1) "→")))))
{% endhighlight %}

I like how org-mode's formatting magically appears on top of my
typing.  With the above, and `font-lock-add-keywords` in general, I
can add more magic to org-mode and Emacs.

## See also

- [Blog where I first heard of this, and more tips!](http://www.howardism.org/Technical/Emacs/orgmode-wordprocessor.html)
- [Source Code Pro](https://github.com/adobe-fonts/source-code-pro)
