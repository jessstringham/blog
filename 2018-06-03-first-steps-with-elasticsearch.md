---
title: First steps with Elasticsearch, searching a book catalog
tags: []
layout: post
display_image: 2018-06-03-ex.png
---

I hacked together a little project to start finding out how much I don't know about Elasticsearch and Lucene.


[Elasticsearch](https://www.elastic.co) is an open-source search engine that is used
for full-text search and log analytics. It's built on top of Lucene,
an information retrieval library. (Heads up, I'll probably accidentally attribute things to Elasticsearch instead of Lucene.)

Here is my understanding of Elasticsearch:
 - It works with data, such as log lines and JSON blobs.
 - It deals with large amounts of data, possibly so much that I would need to distribute it across
multiple machines.
 - It can be used to search the data using a bunch of smart information retrieval things.


In this post, I'll mostly be treating Elasticsearch/Lucene as a black box that stores data and does search.
I'll first spin up an Elasticsearch server. I then send the server documents.
I can tell it some fields are text and how to process them for searching. Then I'll run searches on
the data.

### Project

I want to search for a book title and get the YAML I need for my [reading list]({% link reading.html %}) YAML file.
I decided to use the Seattle library catalog I also used in [Plotting Library Catalog Subjects]({% post_url 2018-05-16-library-catalog-subject %}) because I had it available and it would probably work in most cases.
I'll use [Alfred](https://www.alfredapp.com) for the UI like I did for [note-taking in Jupyter notebook]({% post_url 2018-05-06-notebook-tour %}).

The result is I can type an Alfred command like "*author* one hundred years" and it will return a list of most-likely books.

## Docker-ELK

I decided to use [Docker](http://docker.com) to install Elasticsearch. Searching around I found [docker-elk](https://github.com/deviantony/docker-elk). ELK stands for Elasticsearch, Kibana (dashboard for searching and metrics), and Logstash (which I won't use in this project).

I cloned [docker-elk](https://github.com/deviantony/docker-elk) and ran `docker-compose up` in the directory.
Then I can `curl` my Elasticsearch server!

```
$ curl http://localhost:9200
{
  "name" : "7cL4AdP",
  "cluster_name" : "docker-cluster",
  "cluster_uuid" : "1BfJjKdHQh-iZPmzhR2ktQ",
  "version" : {
    "number" : "6.2.4",
    "build_hash" : "ccec39f",
    "build_date" : "2018-04-12T20:37:28.497551Z",
    "build_snapshot" : false,
    "lucene_version" : "7.2.1",
    "minimum_wire_compatibility_version" : "5.6.0",
    "minimum_index_compatibility_version" : "5.0.0"
  },
  "tagline" : "You Know, for Search"
}
```

Exciting! That was easy!

In order to keep data around after container removal, [I'd need to update the docker-compose.yml file](https://github.com/deviantony/docker-elk#how-can-i-persist-elasticsearch-data).

## `elasticsearch-dsl`

I could interact with Elasticsearch through `curl` commands, but instead, I used the high-level Python API, [`elasticsearch-dsl`](https://elasticsearch-dsl.readthedocs.io/en/latest/).

To experiment with it, I tried

```python
from elasticsearch_dsl.connections import connections

connections.create_connection(hosts=['localhost'])

connections.get_connection().cluster.health()
```

(The `create_connection` is important or else I'd get `KeyError: "There is no connection with alias 'default'."`)


This gives me stats about "shards" and "nodes" and "tasks." More Elasticsearch terms to learn later!

```
{'cluster_name': 'docker-cluster',
 'status': 'green',
 'timed_out': False,
 'number_of_nodes': 1,
 'number_of_data_nodes': 1,
 'active_primary_shards': 1,
 'active_shards': 1,
 'relocating_shards': 0,
 'initializing_shards': 0,
 'unassigned_shards': 0,
 'delayed_unassigned_shards': 0,
 'number_of_pending_tasks': 0,
 'number_of_in_flight_fetch': 0,
 'task_max_waiting_in_queue_millis': 0,
 'active_shards_percent_as_number': 100.0}
```


### Defining the DocType

Heads up, I'm going even deeper into "I don't know exactly what I'm doing" territory. I'm mostly following
the examples from [`elasticsearch-dsl`](https://elasticsearch-dsl.readthedocs.io/en/latest/).

While Elasticsearch can automatically generate schemas (called a "mapping" in Elasticsearch land),
I'll define one for the books.
To do this with `elasticsearch-dsl`, I'll subclass the `DocType` class and define the titles and authors fields and index name.

```python
from elasticsearch_dsl import DocType, Text

class Book(DocType):
    title = Text(analyzer='snowball')
    author = Text(analyzer='snowball')

    class Meta:
        index = 'books'

Book.init()  # tell Elastic about this mapping
```

To add an item, I can do

```python
Book(title='Some book name', author='Some author').save()
```

#### Analyzers and stemming

I'm using `analyzer='snowball'` just because that's what the examples use.
It looks like [analysis](https://www.elastic.co/guide/en/elasticsearch/reference/6.2/analysis.html) handles things like tokenization (figuring out which things in a string are words), removing stop words (such as "the" or "a", which in some cases make search perform poorly), and stemming (converting words like "modelling" to "model"). It looks like "snowball" is referring to the [Snowball stemmer](http://snowballstem.org).

I built my little search index for a [previous project]({% post_url 2018-05-06-notebook-tour %}) using Python's [nltk](https://www.nltk.org), which used the [PorterStemmer](http://www.nltk.org/howto/stem.html). It worked like this: if my documents had the word "modeling" in it, the search index was actually storing the word "model." And if I later searched for "model*s*", it would really search for "model", and find documents that contained "modeling" as well.

Skipping ahead, I think I noticed this in this project. When using a `match` search, if I
searched for "Trainspot," it returned "Trainspotting", but "Trainspott" returned worse results. This is awkward when the UI shows results
while I type in words. I should probably use a different combination of query or analyzer!

### Big Data

That seemed to work for one example, and in the spirit of just seeing if this works, I opted to call `Book(title='Some book name', author='Some author').save()` a ton of times for loading the rest of my data.

I don't think this was the right way to do it. The [Bulk API](https://www.elastic.co/guide/en/elasticsearch/reference/current/docs-bulk.html) sounds like it would have been a better choice.

#### Making big data smaller

I think one of the strengths of ELK is that it can handle data without much preprocessing.
However, I was a little nervous about passing 800MB of data into a system without knowing
how much it was going to increase the size. So I tried to reduce the size of the Seattle catalog data.
Specifically,  I removed columns other than title and author, lightly preprocessed the title, and removed duplicate titles and authors. This decreased the data from 800MB to 40MB.
Then I kicked off the for-loop and went off to do dishes. It probably took around a half an hour to load the 40M of data.

### Searching in Kibana

I have an excuse to use Kibana to try searching the newly added data!

![](/assets/2018-06-03-kibana-search.png)

Nice!

### Searching in code

I can also search using `elasticsearch-dsl`. Using the `Book` mapping defined above, I can try to match the title

```python
results = Book.search()\
    .query('match', title=term)\
    .execute()
```

I don't think `match` is the right option here, but it works for now.

### Gluing it together and using Alfred as the UI

I have the Alfred Powerpack, which lets me make
 [Script Filters](https://www.alfredapp.com/help/workflows/inputs/script-filter/).
 All I need to do is write a script that takes in the search term and outputs
[the special JSON format](https://www.alfredapp.com/help/workflows/inputs/script-filter/json/), and I get a nice little
interface.

### Finished

And it kind of works!

![](/assets/2018-06-03-alfred-search.png)

## Final thoughts

I now have a very convenient way of looking up books' authors which uses the same tools huge tech companies use to deal with big data.

### Cool things

 - `docker-compose` made it easy to set up Elasticsearch for development.
 - I liked using Kibana's "Dev tools" page that let me run the REST calls. This is especially nice because the Elasticsearch documentation gives examples using the REST API.
 - The search results aren't terrible.

### Mistakes

 - As described in the "Analyzers and stemming" section above, searching on every character does not work well with my search and analyzer.

 - With the dataset, my naive title preprocessing didn't trim out media type, so my search looks more like

```python
results = Book.search()\
    .exclude('match', title='[videorecording]')\
    .exclude('match', title='[sound recording]')\
    .query('match', title=term)\
    .execute()
```


### An incomplete list of what I don't know

 - Elasticsearch/Lucene has a lot of options that affect how well search works. For example, I don't know when to use a "fuzzy" instead of a "match", or when different analyzers work better.
   - I imagine that understanding how [relevance](https://www.elastic.co/guide/en/elasticsearch/guide/current/relevance-intro.html) is computed would be useful. It looks like there are neat [explain](https://www.elastic.co/guide/en/elasticsearch/reference/current/search-explain.html) commands for debugging.
   - On the other hand, a lot of the words are familiar, like TF-IDF and stemmers. Maybe I know a little more about Information Retrieval than I thought!
 - Setting this up for production would be very different. I also didn't use the distributed systems features.
