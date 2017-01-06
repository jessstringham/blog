---
title: xmllint --shell
tags: [lil-post]
layout: post
---

I wanted to share a tool I recently found that I thought has an
inventive interface and seems useful. I've been playing with a few
public APIs, like
[Merriam-Webster's dictionaryapi.com](http://www.dictionaryapi.com). I
find with projects like these, I inevitably spend time digging through
a mysteriously formatted response. In this case, the responses are in
XML, so my usual tools are useless (❤︎
[jq](https://stedolan.github.io/jq/)).


    <?xml version="1.0" encoding="utf-8" ?>
    <entry_list version="1.0">
    	<entry id="pettifogger"><ew>pettifogger</ew><subj>LW-1</subj><hw>pet*ti*fog*ger</hw><sound><wav>pettif01.wav</wav><wpr>!pe-tE-+fo-gur</wpr></sound><pr>ˈpe-tē-ˌfȯ-gər, -ˌfä-</pr><fl>noun</fl><et>probably from <it>petty</it> + obsolete English <it>fogger</it> pettifogger</et><def><date>1576</date> <sn>1</sn> <dt>:a lawyer whose methods are <fw>petty</fw>, underhanded, or disreputable :<sx>shyster</sx></dt> <sn>2</sn> <dt>:one given to quibbling over trifles</dt></def><uro><ure>pet*ti*fog*ging</ure><sound><wav>pettif02.wav</wav><wpr>!pe-tE-+fo-giN</wpr></sound> <pr>-giŋ</pr> <fl>adjective or noun</fl></uro><uro><ure>pet*ti*fog*gery</ure><sound><wav>pettif03.wav</wav><wpr>!pe-tE-+fo-g(u-)rE</wpr></sound> <pr>-g(ə-)rē</pr> <fl>noun</fl></uro></entry>
    </entry_list>


_Yay free access to data! Uh, all I really need is that `def` field._


### xmllint \-\-shell

`xmllint` is a tool that lints and parses XML files.

One way it exposes the parsed XML is through the `--shell`
option. `xmllint --shell` lets you explore the XML document on a
REPL. Nice!  Reusing the shell metaphor. Woah, I don't think I've
seen that before.

When I run it on my sample file, it opens the doc at the root of the
XML, or `/` in the metaphoric directory tree. Then I can `cd` and `ls`
around the document.

    $ xmllint --shell some_file.xml
    / > ls
    -a-        3 entry_list
    / > cd entry_list
    entry_list > ls
    ta-        2
    -a-       12 entry
    ta-        1


Or I can `cat` to see the XML again

    def > cat
    <def><date>1576</date> <sn>1</sn> <dt>:a lawyer whose methods are <fw>petty</fw>, underhanded, or disreputable :<sx>shyster</sx></dt> <sn>2</sn> <dt>:one given to quibbling over trifles</dt></def>

There's also `dir`, and `du`. When I've found the data I want, I can
`pwd` to see where I am, and then use the result with `xmllint`'s
`--xpath` option to use in my script

     xmllint --xpath /entry_list/entry/def/dt some_file.xml



Using a REPL to quickly dig through mysteriously structured data
is convenient. The shell and directory structure metaphor is uncannily
natural. And it was already installed on my computer. It seems like a
useful tool to muddle through XML responses with.
