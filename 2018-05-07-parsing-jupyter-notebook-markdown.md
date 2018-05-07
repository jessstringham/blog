---
title: Reading Jupyter notebooks into Python
tags: [jupyter]
layout: post
---

For my [digitized notes]({% post_url 2018-05-06-notebook-tour %}) project, I wrote a few scripts that read Markdown cells from Jupyter notebook files. Specifically, I read a notebook's non-empty Markdown cells and used them for my search index and flashcard database. Reading Jupyter notebooks as data is pretty easy! Below I'll read the non-empty markdown cells.


```python
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
```

## Rendering Markdown and LaTeX

Below shows how to render markdown in a iPython notebook to show what I can do with a dictionary of Jupyter notebook data. This is also how I render flashcards in [digitized notes]({% post_url 2018-05-06-notebook-tour %}).

```python
from IPython.display import display, Markdown

for source in markdown_cells:
    display(Markdown(source))
```
