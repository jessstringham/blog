---
title: 'Reading Jupyter notebooks into Python'
tags: [jupyter, project]
layout: post
mathjax: true
ipynb: https://github.com/jessstringham/notebooks/tree/master/2018-05-07-reading-jupyter-notebooks-into-Python.ipynb
---




For my [digitized notes](https://jessicastringham.net/2018/05/06/notebook-tour.html) project, I wrote a few scripts that read Markdown cells from Jupyter notebook files. Specifically, I read a notebook's non-empty Markdown cells and used them for my search index and flashcard database. 

## Reading Jupyter notebooks as data
Reading Jupyter notebooks as data is pretty easy! Below I'll read the non-empty markdown cells.



{% highlight python %}
import nbformat

path = '2018-05-02-HMM.ipynb'

NB_VERSION = 4

with open(path) as f:
    nb = nbformat.read(f, NB_VERSION)

markdown_cells = [
    cell['source']
    for cell in nb['cells']  # go through the cells
    if cell['cell_type'] == 'markdown' and cell['source']  # skip things like 'code' cells, and empty markdown cells
]
{% endhighlight %}




## Rendering Markdown and LaTeX

Below shows how to render markdown in a iPython notebook to show what I can do with a dictionary of Jupyter notebook data. This is also how I render flashcards in [digitized notes](https://jessicastringham.net/2018/05/06/notebook-tour.html).



{% highlight python %}
from IPython.display import display, Markdown

display(Markdown("**Below is data loaded from [this other file]({})!** \n\n".format(path)))
display(Markdown(markdown_cells[0]))
{% endhighlight %}




## Bonus application (added 2018/05/08),

I realized that GitHub doesn't always render Jupyter notebooks nicely, so here's how I [generate Jekyll posts from my Jupyter notebooks](https://gist.github.com/jessstringham/1ff8ec24dafc0fcff15d4a0e88be074e). This post is an example! Another example is my [alpha recursion post](http://localhost:4000/2018/05/02/hmm-alpha-recursion.html).