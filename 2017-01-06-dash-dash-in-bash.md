---
title: -- in bash
tags: [bash, lil-post]
layout: post
---

The `--` option in bash commands, as in `tox -- -s`, says there are no
more options, and the things that follow are arguments. It's also
really hard to Google for, and so I only learned what it meant while
reading a book on shell scripting.

It's also in `man bash`'s `OPTIONS` section

       --        A  --  signals the end of options and disables further option
                 processing.  Any arguments after the -- are treated as  file-
                 names and arguments.  An argument of - is equivalent to --.

So for example:

    test_program --awesome  # awesome is an option for test_program
    test_program -- --awesome  # "--awesome" is an argument

As an example use case, I might need to collect commands and pass them
to a child script:

    test_program -- --awesome
    # test_program calls other_program with the options --awesome

If `test_program` also did something with `--awesome`, then the `--`
helps tell `other_program` `--awesome`, instead of `test_program`.
