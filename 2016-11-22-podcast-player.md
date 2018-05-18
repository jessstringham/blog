---
title: Podcast Player, overview
tags: [projects]
layout: post
---

I've got this stereo that was built before I was born. There's no
touchscreen or keyboards. It has physical buttons dedicated to
functions. They each feel different. Each media has its own device: a
Tape Player, a CD Player, and a Record Player.

I also listen to a lot of podcasts, so I wanted to build a similar
physical device dedicated to podcasts, like, a, uh, Podcast Player.

## Project

I'm building the Podcast Player using a
[Raspberry Pi](https://www.raspberrypi.org), a tiny computer that's
easy to hook up to a breadboard. Here's the first version:

Eventually I'd like to build up a physical box to blend in with the
rest of the stereo, with a LCD and nice-feeling buttons. But for now,
I've put together a prototype that can download and play Podcasts
using a few buttons.

![Podcast Player v1](/assets/2016-11-22-podcast-player.jpg)

Beautiful!

Heads up, I'm still working on the project. But, having learned from
my handful unpublished drafts of previous projects, I'm going to break
this one into a few smaller installments.

In this post, I'll give an overview of the project and what's left.

### Use Cases

The Podcast Player could do many different things, but I narrowed it
down to three main use cases:

 - I want to be able to press a button, and a podcast start playing.
 - If I've heard the podcast already, I should be able to mark it as
played and never hear it again. Then I should be able to start playing
the next most recent podcast.
 - If I want to hear a specific podcast, I should be able to scroll
through the list of podcast channels, choose one, and start playing
the most recent podcast from that channel.

To do these, I needed some way to download and annotate podcasts.

### Podcast management library

I wrote a
[Podcast management library](https://github.com/jessstringham/podcasts)
that handled downloading and annotating podcasts. I can say more about
it, and I'll try to in a follow-up post, but in this post, I'll focus
on how it relates to the rest of the Podcast Player.

Podcasts usually have an RSS feed that lists new podcasts. So I give
the library a list of podcast feeds to visit, and it checks them for
new podcasts and downloads them. I can schedule this to run every few
hours using `cron`.

![RSS Feed](/assets/2016-11-22-rss-feed.png)

Along with managing downloads, it records information about the
podcasts. It can show the most recently published podcast in the
library, and where that podcast is located on my system. It also
deletes a podcast when I tell it to, and notes down not to download
that podcast again.

I had fun with this part, trying out
[type hinting](/2016/10/26/python35-types.html) and
thinking about
[state machines](https://en.wikipedia.org/wiki/Finite-state_machine),
but the novel part of this project for me was building the physical
box.

### Raspberry Pi

The Raspberry Pi provides a ton of features out of the box, like WiFi
and an audio jack. Given those, my project leaned on the
software side of things. For electronics I just needed two buttons.

I bought a few electronics: breadboards, LEDs, resistors, wires, and
buttons. It's nice because they're just a few dollars each, but it was
tricky to get the right combination of things. I bought the wrong type
of jumper cables. I have a LCD, but I don't have a soldering iron yet
so I can't do anything with it.

Once I had the right things, it went smoothly. There are a lot of
resources and YouTube videos about hooking electronics up to the
Raspberry Pi. I wired up an LED and some buttons and coded them
[to do things](/2016/11/07/raspberry-pi-talking-led.html).

With that, I had the podcast data and knew how to code the Raspberry
Pi. I just needed a little script to glue it all together. But that
little script started growing, so I built some libraries to help me.


### Menu UI

![gratuitous graphics, two buttons](/assets/2016-11-22-buttons.png)

For the first iteration of the Podcast Player, I won't have a display,
a dial, or very many buttons, so I'm using two buttons and audio.

One button rotates through the menu items, and the other selects
items. Every time it moves to a new item, it "highlights the item", or
in this case, reads the item's title using the text-to-speech program
`espeak`. The audio menu is reminiscent of a phone menu's "Press 3
for..."

I wrote a little class to help with the menu. It also supports
sub-menus, and "back" buttons. It worked pretty well, and
[didn't use much code](https://github.com/jessstringham/raspberrypi/blob/5e514425de4f13df405959a7cbfc5eec5e7c7e9e/io_helpers.py#L102).

I wrote those and a few other helper classes to work with the Buttons,
so all that's left is putting it all together.

### [The remaining 10% of the project](https://en.wikipedia.org/wiki/Ninety-ninety_rule)

My next two milestones is making this work end-to-end with one
podcast, and then making it work for the rest of the podcasts I want
it to download . I see two hurdles left before I have something
working end-to-end.

I need to populate a menu based on the Podcast library. I'm still
deciding whether to have a Python API, or follow the `cron` setup and
just make calls to the CLI. The API sounds right.

Also, I need to figure out how I want to run a menu program while
playing podcasts. I want to be able to pause or stop the podcast and
mark it as done with the buttons. Time to dive into subprocesses.


Once I have something working end-to-end, the podcast management
library will need some work before I'll call this project done. It
works great with the single podcast feed, but every time I add a new
podcast channel, I run into issues: one podcast has huge files that
take a long time to download; another had a key missing from its
feed. I might look into using an external library for this.

## Seeya

I'm going to see if I can wrap up this project during the long
weekend.

After playing with the Raspberry Pi, and building a box I can interact
with in the real world, I have a few other Raspberry Pi projects in
mind. I've also bumped up the priority of reading
[The Design of Everyday Things](https://en.wikipedia.org/wiki/The_Design_of_Everyday_Things)
since realizing how much I have to learn about designing user interfaces.

# See Also
 - [Podcasts library](https://github.com/jessstringham/podcasts)
 - [LED](/2016/11/07/raspberry-pi-talking-led.html)
 - [Helper functions](https://github.com/jessstringham/podcasts)
