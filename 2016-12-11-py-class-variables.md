---
title: "Class variables"
tags: [python, things I should have known]
layout: post
---

I've been coding in Python for a long time. Recently, I learned some
nitty-gritty differences between class and instance variables. It's
scary to find things about Python that I probably should have
known. But it's great fodder for blog posts. In the interest of
helping others avoid the same mistake, here's what I got wrong and
what I learned.


_This post is inspired and kinda stolen from
[a question on StackOverflow](http://stackoverflow.com/questions/68645/static-class-variables-in-python),
especially millerdev's answer and Dubslow's comment. The topic is
authoritatively covered by a chapter in
[The Python Tutorial](https://docs.python.org/2/tutorial/classes.html). I
totally should reread the tutorial._

_Code examples were tested in the Python 3.5 and Python 2.7 REPL._

### OOP, more like oops

In Python, classes and instances have attributes attached to them. For
example, they might have the data `total_seen` and `count`:

{% highlight python %}
class Vegetable:
    total_seen = 0
    
    def __init__(self):
        self.count = 0
{% endhighlight %}

My first mistake was thinking `total_seen` and `count` were
interchangeable ways of initializing a default value for an
instance. And, in a lot of cases, it worked the way I imagined it
should.

But I was wrong.

### They are different

To warm up, here's an easy difference between them:

{% highlight python %}
>>> Vegetable.total_seen
0  # total_seen is part of Vegetable!
>>> Vegetable.count
...
AttributeError: class Vegetable has no attribute 'count'  # count isn't!

>>> eggplant = Vegetable()
>>> eggplant.count
0  # count is part of eggplant

>>> eggplant.__dict___
{'count': 0}

>>> Vegetable.__dict___
{'total_seen': 0, ...}  # no count
{% endhighlight %}

`total_seen` is only associated with `Vegetable`, not `eggplant`.

`count` only exists after we create `eggplant`, and is only
associated with `eggplant`. `Vegetable` doesn't treat `self.count`
special at all.


#### Words

From the
[Python tutorial](https://docs.python.org/2/tutorial/classes.html#class-and-instance-variables),
I'm talking about the difference between class and instance
variables. `total_seen` is a _class variable_ and is associated with
`Vegetable`. `name` is an _instance variable_ and is associated with
`eggplant`.


## Globals

I like to collect spooky programming examples. My favorite type of
example is one where the code doesn't throw errors and looks correct
to the untrained eye, but does the wrong thing.

Now that we know class and instance variables are different, let's
come up with a spooky example.

Let's say we want all classes to actually share the same variable,
like to track the global total number of vegetables seen. (Aside:
Globals freak me out, and this seems like a questionable approach. But
I'll use it for a contrived example.)

Here's the spooky code:

{% highlight python %}
class Vegetable: 
    total_seen = 0
    
    def __init__(self):
        self.total_seen += 1  # This is the spooky part

>>> eggplant = Vegetable()
>>> eggplant.total_seen
1
>>> kale = Vegetable()
>>> eggplant.total_seen
1  # Wrong! We've seen 2 vegetables!

>>> eggplant.__dict__  # or kale.__dict__
{'total_seen': 1}
>>> Vegetable.__dict__
{'total_seen': 0, ...}  # it never changed!
{% endhighlight %}

When `eggplant` is created, it looks for `self.total_seen`, and finds
the class attribute, which has a value of `0`. Then it adds one, and
assigns the result to `total_seen`'s evil twin, the _instance
attribute_. The _class attribute_ remains as `0`.

### Let's try that again

Compare that to when we use the class-level attribute

{% highlight python %}
class Vegetable: 
    total_seen = 0
    
    def __init__(self):
        Vegetable.total_seen += 1  # This part changed

>>> eggplant = Vegetable()
>>> eggplant.total_seen
1
>>> kale = Vegetable()
>>> eggplant.total_seen  # or Vegetable.total_seen, or kale.total_seen
2  # Whee, now it's 2

>>> Vegetable.__dict__
{'total_seen': 2, ...}  # it changed!
>>> eggplant.__dict__
{}  # eggplant doesn't have total_seen at all
{% endhighlight %}


### Flashbacks of mutability gotchas

Let's try modifying `self`'s instance-level attribute again, but mix
it up.  Instead of using `+=` on an `int`, let's `append` to a
list. Since the `int` was immutable, Python created a copy for us. In
the case of a list, which is mutable, we're going to mutate the
variable.

{% highlight python %}
class Vegetable: 
    all_names = []
    
    def __init__(self, name):
        self.all_names.append(name)  # interesting

>>> eggplant = Vegetable("eggplant")
>>> eggplant.all_names
["eggplant"]
>>>
>>> kale = Vegetable("kale")
>>> eggplant.all_names  # or Vegetable.all_names, or kale.all_names
["eggplant", "kale"]
>>> eggplant.__dict__
{}  # woah, we used self, but no all_names
{% endhighlight %}

Surprise! `self` mutates the global in this case. It doesn't create
anything in the instance.

And, as far as I can tell (and I admit I'm still fuzzy in this area)
I'm not sure if there's a functional difference here between
`self.all_names.append()` and `Vegetable.all_names.append()`. From a
coding clarity perspective, I'd rather use `Vegetable.all_names`, so
it's clear we're modifying a class-level attribute.


We could turn this into another spooky case by flipping the example:
what if I wanted `all_names` to be an instance variable? It'll fit in
nicely with the "`def fun(self, some_list=[])` might not do what you
think it does" spooky example.

## So tell me the story where you broke everything!

Here's the part where I assure myself that I probably never broke
things. I normally wouldn't use class variables for global things or
initializing lists, and if I did, I'd hope I'd catch it in tests.

Actually, code like that last example instigated this post. I found
code like that and updated it to use the class variable explicitly. I
couldn't come up with a test that broke before the change and worked
after.




## See Also
 * The original
   [StackOverflow question and discussion](http://stackoverflow.com/questions/68645/static-class-variables-in-python)
 * [Python tutorial](https://docs.python.org/3/tutorial/classes.html#class-objects)

