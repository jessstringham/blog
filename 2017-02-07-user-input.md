---
title: Fun with command line interfaces
tags: []
layout: post
---

Most programs need to interact with the outside world. For personal
programming projects, I have a lot of freedom to choose how it does
that. I could write it as a web application! Or a mobile app! Or a
physical interface with Raspberry Pi buttons! But for some projects,
the interface is just boilerplate I need to get out of the way so I
can work on the interesting part.

For these, I like using command line interfaces. First, CLIs are
probably the easiest interface to code. But also a lot of legit
programs use CLIs (git! docker!), so for some of my projects, a CLI
isn't the hacky prototype, but a reasonable final product. Being
polished and easy to code is a great combination.

### CLI doesn't have to mean interactive prompts

When I started coding, I thought a command line interface meant an
interactive prompt.

If I'm writing a program that tells me the weather for a city, I might
need the city name and some options. The program could have a command
line interface with an interactive prompt like

    $ ./weather.py
    Where?
    > Seattle
    Emoji? [Y/n]
    > Y
    It is cold, dark, and rainy. 🌧

I rarely code interactive prompts anymore. I think, uh, "regular"
commands are easier to code and are often a better user experience.


### Command line tools

If I did the weather application today, I'd probably have the
interface be more like

    $ weather Seattle --emoji
    It is cold, dark, and rainy. 🌧

As the programmer, this takes less work to write than the interactive
prompt. I can use Python's
[argparse](https://docs.python.org/3/library/argparse.html).

As the user, I like this because I can enter everything at once. If I
make a mistake, I can go back into my history and fix the command.  I
can also start stringing command line tools together, like

    $ weather --emoji `get_current_city`



#### Config files

There are a few other patterns that make command line tools even
cooler.

Instead of typing `Seattle --emoji` each time, I could write the
program to take in a config file that looks like

    city: Seattle
    emoji: true
    
And then add code to read from a config file. Then I can call

    ./weather -c my_configs.yaml

If I travel to another city, I just need to edit the config file. I
can also share the configs across different computers, or put them
under version control.

This is admittedly a bit more code since I need to parse and validate
the config file. I have some boilerplate using
[JSON Schema](http://json-schema.org) to help me out here. But
depending on the project, it could be useful.

#### Files and stdin

I could also build a command line tool that uses stdin. Maybe I'd have
a file like

    - city: Seattle
      date: 2016-01-01
    - city: Portland
      date: 2016-02-01
    - city: San Francisco
      date: 2016-03-01
      
    cat test_file | ./weather -c configs.yaml
    Seattle was cold and rainy 🌧
    Portland was cold and rainy 🌧
    San Francisco was not cold nor rainy 🌤


To modify the program input, I just need to open the YAML file and
edit the line I want to change. 

With stdin, it's easy to glue programs together. If I have one program
that prints JSON, and one program that takes JSON through stdin, I can
string them together. I can also be hacktastical and use Google
Spreadsheets to edit a file, export as CSV, and use that as input.

Like the config, this does require a bit more code. But parsing a
machine-readable file format is pretty easy.

### Final observations

Like shinier user interfaces, command line interfaces need some
considerations for user experience. A cool thing about command line
interfaces is that I get exposed to them a lot. I have many
opportunities to see useful and less useful ways to design a command
line interface.


## See Also
 - Acknowledgments to [Brad](http://bradjensen.net), for pointing out
   about how files are totally legit interface for programs a few
   years ago.
