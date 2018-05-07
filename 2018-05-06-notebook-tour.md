---
title: Tour of digitized notes in Jupyter
tags: [note-taking]
layout: post
---

This post is a tour of my notebook system's frontend. See [Digitizing notes as a learning tool]({% post_url 2018-05-06-studying %}) for a description of the project.

## Search

When I save my notebooks, I have a `post_save_hook` in the `jupyter_notebook_config.py` that also updates a search index.
Then I have an [Alfred](http://alfredapp.com) workflow that runs my search script.
For example, one command searches for bolded terms and headers:

<img src="/assets/2018-05-06-search.gif" width="400px">

And open selected notebooks in my browser:

<img src="/assets/2018-05-06-open.gif" width="400px">

## Notes and flashcards

These are what my notes from [Digitizing notes as a learning tool]({% post_url 2018-05-06-studying %}) look like:

![Example of notes](/assets/2018-05-06-notes.png)

### Separate flashcard page

On notebook save, another `post_save_hook` adds flashcard-notes to a little `sqlite` database. I use another Jupyter notebook to load cards from the database. I display flashcard buttons using `ipywidgets` and display the markdown using `IPython.display.Markdown`.

<img src='/assets/2018-05-06-flashcard-page.gif' alt='Example of flashcards' width='400px'>

* I don't use answer-difficulty information yet, but the idea is that I can use it for some fun [spaced repetition](https://en.wikipedia.org/wiki/Spaced_repetition) stuff!

### Viewing notes as flashcards

Back in the notes themselves, I also have buttons defined using Jupyter's `custom.js` which switch to this alternate flashcard-view:

![Example of flashcards](/assets/2018-05-06-flashcards.png)

This is not optimal, but I also have Jupyter buttons that map to the {easy, perfect, mixed, hard} choices:

![Example of buttons](/assets/2018-05-06-buttons.png)





