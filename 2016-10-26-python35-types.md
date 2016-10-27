---
title: Python3.5 type hints
tags: [python]
layout: post
---

I was playing with Python3.5's type hints for a few hours, and thought
I'd summarize what I found here.

A quick disclaimer and blogging note: I have around a dozen unfinished
blog post drafts about various subjects, so I thought I'd focus on
getting this published more than about being super thorough.  Proceed
with caution.


### The Project 

I started a small project in Python2.7.  My style nowadays involves
calling a bunch of functions on namedtuple models.

After a little refactor, and a long time battling confusing bugs and
saying for the *n*<sup>th</sup> time *"this could have been avoided if
Python checked types"*, I remembered hearing something about Python
having type-checking.

### Not quite

First, things are more subtler than "Python has type-checking now."
It's not type-checking like Haskell or Rust. Namely:

 * The interpreter more-or-less ignores types. It will run the code
whether or not you provide type annotations.

 * Python 3 has had type annotation syntax for a while.  Python3.5
introduced the [typing](https://docs.python.org/3/library/typing.html)
module, which allows for more complex types, e.g. List[int].

 * `mypy-lang` is a Python package that gives you `mypy`, a program
that does static type-checking.  It's up to you to run mypy.  It runs
over all of the code you point at, not just the code that is executed.
Like [Guido says](https://www.youtube.com/watch?v=2wDvzy6Hgxg), it's
more like a really powerful linter.

 * Types feel like an afterthought. It's not a huge deal, just some
extra work at the moment. Don't expect package documentation to
mention types, like Haskell or Rust documentation does.

But aside from that, you can get that type-checking of your dreams.

### What worked for me

Type-checking with mypy is kind of all-or-nothing. Type annotations
are optional. But `mypy` will silently fail to notice issues if it
can't infer types. So it requires _enough_ types hints to be useful.

I used these guidelines:

In the package, I annotate all function definitions and models, and
optionally annotate everything else.  I call `mypy
--disallow-untyped-defs` to enforce type annotations on all function
definitions.

In tests, I switched things up a bit: I don't annotate test functions
and use `mypy --check-untyped-defs` option to make sure type-checking
happens inside of the test. (Note: I still haven't gotten a handle of
`--check-untyped-defs`. I feel like sometimes it's skipping chunks of
code.  Since these are already tests, I feel okay about it.)

I run `mypy` on my package and tests every time I run my full test
suite.

I require the output of `mypy` to be empty if nothing is wrong.  To
avoid `error: No library stub file for module 'feedparser'`, I added
minimal stub files (.pyi) in my package for dependent packages, and
call with `MYPYPATH=package_name mypy ... `. (See also
[python/typeshed](https://github.com/python/typeshed))

The complete commands look like

{% highlight makefile %}
check_types: venv
	MYPYPATH=podcast $(PREFIX)/mypy podcast --disallow-untyped-defs
	MYPYPATH=podcast $(PREFIX)/mypy tests --check-untyped-defs
{% endhighlight %}


If following these guidelines, I'd recommend adding types to a project
as early as possible. I wouldn't use this approach to add types to an
existing large project. It was a lot of work to add all annotations,
and I needed to do that before I started getting value from them.

#### Example 1: Why use `--disallow-untyped-defs`

Okay, a couple of concrete examples. Let's say I have code like

{% highlight python %}
def say_hi(name):
    return "Hello " + name
    
def do_stuff(name):
    return say_hi(name)
    
    
do_stuff(123)
{% endhighlight %}

I'd love for mypy to notice that I was doing something silly with
`"Hello " + 123`. It doesn't at the moment.

So I add type annotations to the `say_hi` function.

{% highlight python %}
def say_hi(name: str) -> str:
    return "Hello " + name
    
def do_stuff(name):
    return say_hi(name)
    

do_stuff(123)
{% endhighlight %}

mypy won't notice this either.

I think this is for two reasons. For one, `do_stuff` isn't annotated,
so it won't check the body without `mypy --check-untyped-defs`. 

Another reason is that it doesn't know `do_stuff`'s type. I can show
this using `mypy`'s method for debug types by adding
`reveal_type(...)` to my code.

{% highlight python %}
def say_hi(name: str) -> str:
    return "Hello " + name
    
def do_stuff(name):
    return say_hi(name)
    
    
reveal_type(do_stuff)

do_stuff(123)
{% endhighlight %}


`mypy` will output
```
Revealed type is 'def (name: Any) -> Any'
```

I haven't read up on the type system, but the rule-of-thumb I use is
once an `Any` type shows up, mypy says any code interacting with it is
fine, and won't detect errors.

With `--disallow-untyped-defs`, I'd remember to annotate all
functions, and could end up with something like this:

{% highlight python %}
def say_hi(name: str) -> str:
    return "Hello " + name

def do_stuff(name: int) -> str:
    return say_hi(name)
    
do_stuff(123)
{% endhighlight %}

Finally! `mypy` will complain

`Argument 1 to "say_hi" has incompatible type "int"; expected "str"`


#### Example 2: Other places to use types

Aside from function defs, it can be useful to add type-checking to
weirder structures. For example, I'd annotate a mapping of command
names to functions.

{% highlight python %}
def print_status(radio: Radio) -> Radio:
...

def update_radio(channel: Channel) -> Radio:  # uh oh
...

def download_radio(radio: Radio) -> Radio:
...

radio_action = {
    'status': print_status,
    'update': update_radio,  # nope, not allowed
    'download': download_radio,
}  # type: typing.Dict[str, typing.Callable[[Radio], Radio]]

radio = radio_action[args.command](radio)
{% endhighlight %}

`mypy` will give us the error `List item 1 has incompatible type "Tuple[str, Callable[[Channel], Radio]]"`

Ignoring the mentions of `List` and `Tuple`, it's saying something's
wrong with the second item in this dictionary definition.

### Catching bugs

After finding an `mypy` setup that caught issues, and doing the
busy-work of annotating my functions, I found a few errors!

In my actual project, I caught:

 - accidentally iterating through a namedtuple instead of one of its fields
 - typo in a namedtuple field name
 - swapping method arguments
 
It was super gratifying.

I might have caught some of these with 100% test coverage, but there's
always a possibility of having a bug return a reasonable but
completely incorrect result.  I'd also argue the type-checking error
messages are easier to debug than failing tests.



# See Also


 - [typing](https://docs.python.org/3/library/typing.html)
 - `mypy --help` and [mypy command line](http://mypy.readthedocs.io/en/latest/command_line.html)
 - [PEP 484](https://www.python.org/dev/peps/pep-0484/#the-typing-module)
(I, uh, didn't actually read this.)
 - [Guido's talk at PyCon2015 on types](https://www.youtube.com/watch?v=2wDvzy6Hgxg)
