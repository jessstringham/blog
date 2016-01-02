---
title: Clocks in org-mode
tags: [emacs, org-mode]
layout: post
---

For my first ever post on my blog's *n*<sup>th</sup> incarnation, I'm
going to talk about clocking in org-mode. Because starting at the
beginning is overrated.

Instead, I'm rudely assuming you know emacs and org-mode and already
want to start clocking the time you spend on tasks. Come back in a few
months, and I'll have more background info. In the meantime, here are
a few favorite org-mode sites and blogs.

 - [Org-mode Manual](http://orgmode.org)
 - [Sacha Chua's blog](http://sachachua.com/blog/category/geek/emacs/org/)
 - [Org-mode - Organize Your Life In Plain Text!](http://doc.norang.ca/org-mode.html)


## The _keys_ to success. Sorry.

If you check out the
[org-mode manual](http://orgmode.org/manual/Clocking-work-time.html#Clocking-work-time) entry on clocking, you'll soon get to the [keybindings](http://orgmode.org/manual/Clocking-commands.html), which look like... fun? (Makes you wish you had chosen vim, eh?)

**Pro Tip:** Many of org-mode's cool commands start with `C-c C-x`.

For org-mode clocking, the keys that follow `C-c C-x` are easy to remember. 

`C-c C-x C-i` clocks **i**n, `C-c C-x C-o` clocks **o**ut, and `C-c
C-x C-j` **j**umps to the task you're clocked in to.

The fun continues in org-mode outside of clocking! You can view tasks in the
lovely [**c**olumn mode](http://orgmode.org/manual/Column-view.html#Column-view) with `C-c C-x C-c` (edit org-mode tasks in a text-based spreadsheet!) and edit a task's
[**p**roperties](http://orgmode.org/manual/Properties-and-Columns.html) with `C-c C-x p`.

## Beginner clocks

Before I started clocking in, my org-mode-based days worked something like this:

 - View trusty agenda and pick a task to work on.
 - Do work! Take some notes in org-mode.
 - Probably get distracted and switch tasks. Maybe C-c c to create a new task.
 - Go back into org-mode and mark a few tasks as done.
 
Sprinkling clock commands into this was easy.

### 1. View trusty agenda

`C-c a` to pull up my agenda (or whatever keys are bound to `org-agenda`). Move the point to a task. For example, I
have a task that pops up every morning (using
[habits](http://orgmode.org/manual/Tracking-your-habits.html)) to
empty my inbox.

### 2. Clock in to your first task

From the agenda, you can clock in using `I`. If you're in your .org file and the point is in a task,`C-c C-x C-i` will clock in to it.

You'll see a snazzy little timestamp that looks something like this:

`CLOCK: [2016-01-01 Fri 10:00]`

Whee. Now do things!

### 3. Clock in to your second task

Time to switch tasks! Once again, `C-c C-x C-i` or `I` in the agenda to clock-in to a different task. You'll notice your old clock gets a conclusion:

`CLOCK: [2016-01-01 Fri 10:00]--[2016-01-01 Fri 10:15] =>  0:15`

and your current task will get a new clock:

`CLOCK: [2016-01-01 Fri 10:15]`

(Incidentally, outside of org-mode, emacs sometimes interprets `C-x C-i` as `TAB`. Which makes for interesting error messages when I try to clock in from the wrong place!)

### 4. More things for another time!

There are a few more things you can here, like automatically clock in to
newly captured tasks. My set up is based on a couple of configs, and those are coming in a future post. Spoilers, it's a slight variation on
[Bernt Hansen's set up](http://doc.norang.ca/org-mode.html#Clocking).

### 5. Clock out when you're taking a break

`C-c C-x C-o` to clock out and go eat food or stretch or go outside or something. You can track these too, if you want to.

## Switching tasks, take 2

Perfect! Or, not quite. A few times a day, I'll switch tasks but
forget to clock in exactly when I start the task. I'm clocking
entirely for myself, but I still want accurate data.

No worries, it's all good in plain text org-mode land.

So say you've clocked in at 10:45 but really started a few minutes earlier at 10:40. Remember that string you got:

`CLOCK: [2015-10-31 Sat 10:45]--[2015-10-31 Sat 11:00] => 0:15`

After clocking in to the task you missed, you can go in and manually
change that 10:45 to a 10:40. Then with the cursor still in the
clocking line, type `C-c C-c` to recompute the total time.

`CLOCK: [2015-10-31 Sat 10:40]--[2015-10-31 Sat 11:00] => 0:20`

But wait, there's a better way! Move the point over a part of the
timestamp, press `S-M-<up>`, and it will increment it in the way timestamps should!

`CLOCK: [2015-10-31 Sat 10:30]--[2015-10-31 Sat 11:00] => 0:30`

`CLOCK: [2015-10-31 Sat 10:25]--[2015-10-31 Sat 11:00] => 0:35`

`CLOCK: [2015-10-31 Sat 10:20]--[2015-10-31 Sat 11:00] => 0:40`

Super cool. Try it out with your point on the date or the hour
or the minute. This also adjusts the timestamp of the task you were
clocked in to before, so it's perfect for correcting interruptions.

## Now what?

That's mostly for another post. But here's one thing:

You can pull up an agenda and see how you spent your day today by
going into your agenda `C-c a` (or whatever keys do `org-agenda`), and
then 'view clockcheck' by typing `v c`.  From here, like typical
org-mode agenda, `b` goes back a day. This technically checks for
issues with your clocks, but it works nice as an out-of-the-box
summary without fussing much with clocktables.

    file_name:    10:00-10:30 Clocked:  (0:30) Task A
    file_name:    10:30-11:00 Clocked:  (0:30) Task B
    file_name:    11:00-11:15 Clocked:  (0:15) DONE Task C
    
## See more

For more clocking fun in the meantime:

 - [Org-mode Manual on Clocking](http://orgmode.org/manual/Clocking-work-time.html), and [clock tables](http://orgmode.org/manual/The-clock-table.html)
 - [Sacha Chua's blog](http://sachachua.com/blog/category/geek/emacs/org/)
 - [Org-mode - Organize Your Life In Plain Text!](http://doc.norang.ca/org-mode.html#Clocking)
