---
title: 'Daylight Project: Part 2, travel.org'
tags: [projects, emacs, org-mode]
layout: post
display_image: 2016-12-21-daylight.png
---

I made a visualization of the length of the day in the cities I've
been in.  Most of the "application" backing it is really the magical
powers of Emacs and Org-mode, and an Org file I called
`travel.org`. In this post, I'll talk about a few Emacs and org-mode
features, and how I used them to throw together a hacktastical
application that draws squiggly lines.


![daylight](/assets/2016-12-21-daylight.png)

_See [Part 1]({% post_url 2016-12-14-daylight-map %}) for an overview
of the project, and more squiggly lines._



I built most this project in Org-mode and Emacs. I don't just mean I
used Emacs as a text editor: an Org file serves as the database, and
it even uses built-in [Emacs commands](https://www.gnu.org/software/emacs/manual/html_node/emacs/Sunrise_002fSunset.html) that compute the minutes of daylight. It's bizarre, but
it was a fun excuse to learn more about Org-mode and Emacs.

I'm going to give a whirlwind tour of a few cool Org-mode and Emacs
features, and how I used them to build the travel database and compute
the length of the day. For best results, you should know what
[Org-mode](http://orgmode.org) is.

### 1. Org-mode Headlines and Properties

I have the travel data itself. I represent each travel as a
[headline](http://orgmode.org/manual/Headlines.html). I use
[properties](http://orgmode.org/guide/Properties.html) to note the
city, start date, and end date.

    *** Portland visit
    :PROPERTIES:
    :DateRange: [2010-02-01]--[2010-03-01]
    :City:     us__portland__or
    :END:

### 2. Emacs Keyboard Macros

Before I started on `travel.org`, I had tediously collected
information about my travels in another format. The file I had looked
like:

    PLACE		   	START DATE	END DATE
    Seattle, WA, US 	2010-01-01	2010-02-01
    Portland, OR, US	2010-02-01	2010-03-01
    Vancouver, CA		2010-03-01	2010-04-01

I wanted to format it using the headlines and properties, like:

    *** Portland visit
    :PROPERTIES:
    :DateRange: [2010-02-01]--[2010-03-01]
    :City:     us__portland__or
    :END:

For this one-time transformation, I had a good excuse to practice
using Emacs'
[keyboard macros](https://www.gnu.org/software/emacs/manual/html_node/emacs/Keyboard-Macros.html).

Emacs keyboard macros let you record a series of commands and replay
them. [EmacsWiki](https://www.emacswiki.org/emacs/KeyboardMacrosTricks)
has some cool examples and tricks.

It's fun to puzzle out which series of macros will do the right thing,
and watch the file transform in the process.

    Seattle,WA,US 	2010-01-01	2010-02-01
    Portland,OR,US	2010-02-01	2010-03-01
    Vancouver,CA	2010-03-01	2010-04-01

...

    *** Seattle
    us,seattle,wa	[2010-01-01]--[2010-02-01]

    *** Portland
    us,portland,or	[2010-02-01]--[2010-03-01]

    *** Vancouver
    ca,vancouver	[2010-03-01]--[2010-04-01]

...

    *** Seattle
    :PROPERTIES:
    :City:      us__seattle__wa
    :DateRange: [2010-01-01]--[2010-02-01]
    :END:

    *** Portland
    :PROPERTIES:
    :City:      us__portland__or
    :DateRange: [2010-02-01]--[2010-03-01]
    :END:

    *** Vancouver
    :PROPERTIES:
    :City:      ca__vancouver
    :DateRange: [2010-03-01]--[2010-04-01]
    :END:


### 3. Org-mode Column View

After the keyboard macro transformation, I had `travel.org`, an
Org-mode database of my travels. I dug through my old photos and added
the last year of travel data as new headlines and properties. Now I
wanted to check the new data for mistakes.

`C-c C-x C-c` switches headlines and properties to
[column view mode](http://orgmode.org/manual/Using-column-view.html#Using-column-view). For example, it switches something like:

    *** Seattle
    :PROPERTIES:
    :DateRange: [2010-01-01]--[2010-02-01]
    :City:     us__seattle__wa
    :END:

    *** Portland
    :PROPERTIES:
    :DateRange: [2010-02-01]--[2010-03-01]
    :City:     us__portland__or
    :END:

    *** Vancouver
    :PROPERTIES:
    :DateRange: [2010-03-01]--[2010-04-01]
    :City:     obvious_typo
    :END:

into something like:

    #+COLUMNS:  %DateRange %City %10ITEM
    | DateRange                   | City             | ITEM         |
    |-----------------------------+------------------+--------------|
    | [2010-01-01]--[2010-02-05]  | us__seattle__wa  | *** Seattle  |
    | [2010-02-01]--[2010-03-01]  | us__portland__or | *** Portland |
    | [2010-03-01]--[2010-04-01]  | obvious_typo     | *** Victoria |

After learning the keyboard commands (par for the course), you can
edit this table like a spreadsheet, and it updates the underlying
headlines and properties.


### 4. Org-mode API

But editing tables is tedious. I wanted the Emacs to find issues for
me. To do this, I wrote some Elisp. I often want to go through an Org
file's headlines. I used the Org-mode APIs:

 * [`org-map-entries`](http://orgmode.org/manual/Using-the-mapping-API.html),
which iterates through the headlines and
 * [`org-entry-get`](http://orgmode.org/manual/Using-the-property-API.html#Using-the-property-API),
which extracts properties.

I frequently end up with something like

    (defun get-cities ()
         (delq nil
          (org-map-entries
           '(lambda ()
              (when (= (org-outline-level) 3)
                (org-entry-get nil "City"))))))


With Emacs Lisp, part of the state of the program is where the Emacs
cursor is. org-map-entries secretly moves a cursor around, and
org-entry-get accesses information about the header the cursor is on.

### 5. View the open source

In writing these functions, I found myself reading Elisp documentation
and digging through source code.

You can open documentation for an Elisp function with `C-h f` and `C-h v`.

Sometimes it's useful to dive into the source code itself. It's linked
to from the top of each entry, like "org.el" in

    org-map-entries is a Lisp function in `org.el'.

### 6. Org-mode's Dynamic blocks

Back to Org-mode. I also used
[dynamic blocks](http://orgmode.org/manual/Dynamic-blocks.html) like
this:

I wrote a function to check for overlapping and missing dates, and
wrap it in a specially-named function.

    (defun org-dblock-write:block-jessicas-check-date-overlap (params)
        (insert (check-for-overlapping-dates)))

Then I added a dynamic block to `travel.org`.

    #+BEGIN: block-jessicas-check-date-overlap

    #+END:

I enter `C-c C-c` to updated the block. Emacs executes the function
and replaces the block contents with the output of my program.

    #+BEGIN: block-jessicas-check-date-overlap
    2010-03-01
    #+END:


### 7. Using Python

I had a text file that mapped the world's cities to coordinates, and I
needed to lookup my travel city's coordinates. As much fun as I was
having in Emacs, I didn't want to write it in Elisp, so I wrote this
component in Python, and wrote Elisp to call Python


    (defun lookup-lat-lngs ()
      (shell-command-to-string
       (format "python travel/lookup_id.py --data '%s'"
               (join-strings-with-newlines
                (get-cities)))))


[Ergoemacs](http://ergoemacs.org/emacs/elisp_perl_wrapper.html)
provides more examples of how to do this.


## Oh no, what have I done

So that was my "application". The database was an Org file, which I
had created from a text file using keyboard macros.  The interface for
validating the input was also the Org file, using column view, and
dynamic blocks, which called into Elisp functions that used the
Org-mode API. I also used code to call Emacs `solar-sunrise-sunset` to
compute the number of minutes in a day.

As I hacked this together, it became less of an "I'm going to throw
this on GitHub" and more of an "I'm going to have to settle for a
blog post."

But it was fun, mind-bending, and helped me learn my tools.
