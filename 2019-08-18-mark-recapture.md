---
title: "Population Estimation with Mark-Recapture"
tags: [projects, interactive]
layout: post
mathjax: true
display_image: 2019-08-18-ex.png
---

How would you estimate the number of seagulls that live in Stockholm in the summer? Because of the difficulty of conducting a formal seagull census,
techniques such as mark-recapture experiments can be used to estimate the population by marking random samples.

Estimating the number of seagulls in Stockholm using a set of mark-recapture experiments could work like this: scientists start by "capturing" a random sample of seagulls, "marking" the seagulls with metal bird-anklets, and releasing the seagulls. Later, the scientists capture another random sample of seagulls and count the number of "recaptured" seagulls that already have bird-anklets. From the information gathered from that process alone, scientists can estimate the seagull population!

In this post, I'll go through the math behind one method to estimate the population from mark-capture experiments. The math required is a nice reminder of statistics, with appearances by Bayes' theorem and the Binomial distribution. This post will focus on the math, and I'll gloss over many important and interesting questions, such as how to take a representative sample of things that move around, lose their mark, migrate, reproduce, and die.


## Capture, Mark, Release, repeat!

Before looking at the math, I'll illustrate the example above. The process is:

* Capture a random sample
  * Note how many were captured and how many were recaptured (i.e. have a mark)
  * Update the population estimate
* Mark the sample
* Release the sample
* Repeat!

Below is an illustration. Click on the box below to go to the next step. The large circles are captured things. Light beige circles represent those we have not seen before. Dark blue are those that are recaptured.

<div id="intro"></div>

## Math

One method to do population estimation using mark-recapture experiments is explained in the 1986 paper "Population Estimation from Mark-Recapture Experiments using a Sequential Bayes Algorithm" by W. J. Gazey and M. J. Stanley.

The method produces a discrete probability distribution over a fixed set of possible population sizes \\(N\\).
What this means is that I need to provide a guess of the range of the population and choose the granularity of estimates.
For example, if I think the population is between 20 and 1000 and I need the granularity of 20, this method can give a distribution for the population \\(N = [20, 40, ..., 980, 1000]\\). It will give me values such as \\( P(N_i = 100) = .32 \\), which says that there is a 32% probability of the population being is 100 as opposed to the other possible values.

In addition, this method incorporates the results of multiple mark-recapture experiments. I'll annotate the experiments with the superscript \\((t)\\), as in \\(C^{(t)}\\).

To update \\(P(N)\\), the method starts with \\(P(R^{(t)} \mid N_i)\\), or the probability that we would recapture in experiments \\(t\\) a total of \\(R^{(t)}\\) things given the true population \\(N_i\\). \\(P(R^{(t)} \mid N_i)\\) is a function of how many things are already marked \\(M_t\\) (which we know), how many we captured \\(C_t\\) (which we know), and the population \\(N_i\\) (which I'll get back to).
Remembering combinatorics, we get

\\(
P(R^{(t)} \mid N_i) =
  \binom{C^{(t)}}{R^{(t)}}
  \left(
    \frac{M^{(t)}}{N_i}
  \right)^{R^{(t)}}
  \left(
    1 - \frac{M^{(t)}}{N_i}
  \right)^{C^{(t)} - R^{(t)}}
\\)

<small>Equation 1c from the Gazey and Stanley paper.</small>

However, what we really want is \\(P(N | \mathcal{D} )\\), or the distribution over the population given what we know about the values of \\(M\\), \\(C\\), and \\(R\\) from each experiment.
To do that, we can apply Bayes' Theorem:

\\(
P(N_i^{(t)}|\mathcal{D}^{(t)}) = \frac{
  P(N_i^{(t - 1)}) P(R^{(t)} \mid N_i)
}{
  \sum_{i} P(N_i^{(t - 1)}) P(R^{(t)} \mid N_i)
}
\\)

Each experiments \\(t\\), we'll use the current estimation \\(P(N^{t-1})\\) from the previous experiment. The equation requires a prior over the population \\(P(N^{(t=0)})\\), for which we'll use a uniform distribution over the population range.
To turn this into an estimated population range, I try to find the 95% highest posterior density interval of \\(P(N)\\).

<small>*There's also a problem where if \\(N_i < M\\) we get undefined values. To hack around this, I zero out those probabilities before computing the updated distribution.*</small>

## How the distribution changes by experiment

And that's it! Below is a simulation of the distribution \\(P(N)\\) updating with each new capture-mark-release study.

The animation has three components: the numbers being updated, the current distribution \\(P(N)\\) given the data we know, and what it would look like if we could see the entire population. After slowly walking through a few steps, the animation will speed up and go through around 20 mark-recapture experiments.


<div id="graph">&nbsp;</div>

## In code

The simulation for the animation was generated using <a href="https://gist.github.com/jessstringham/c1a9f90ef62672597b07713ce68fd439">this code</a>. When I don't need to illustrate the values, I can try more realistic populations. For example:

```
true_N = 8000
fs = FieldStudy(true_N, 1000, 100000, 100)

for _ in range(21):
    fs.sample(difficulty_of_catch=0.005)

print("Mode", fs.N[np.argmax(fs.P_k)])
```

In this example, the true population is \\( 8000 \\). I start with a guess that the population is between \\( 1000 \\) and \\( 100000 \\). After 20 experiments (excluding the initial marking experiment), this gives me a mode of \\( 8000 \\) and a range of \\(7000 - 12000\\). Not bad!

<small>*`difficulty_of_catch` is the probability of capturing each thing. I needed that for the demo and probably wouldn't know that value if I did a real experiment.*</small>

## Citations and other links

 - Gazey, W. J., and M. J. Staley. "Population estimation from mark‐recapture experiments using a sequential Bayes algorithm." Ecology 67.4 (1986): 941-951.
 - <a href="https://gist.github.com/jessstringham/c1a9f90ef62672597b07713ce68fd439">Code that produces simulations</a>

<script src="/assets/js/d3.v5.min.js"></script>


<script src="/assets/js/d3.v5.min.js"></script>

<style type="text/css">

.hiddenDotsTitle {
  color: #CCC;
  padding: 5px;
  margin: 2px;
  font-family: Courier;
  height: 1em;
  display: inline-block;

  -moz-user-select: none;
  -webkit-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.highlightedDotsTitle {
  color: #000;
  background-color: #dbd5b5;
  padding: 5px;
  margin: 2px;
  font-family: Courier;
  height: 1em;
  display: inline-block;

  -moz-user-select: none;
  -webkit-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.timestepDots {
  color: #000;
  border-right: 1px solid #CCC;
  padding: 5px;
  margin: 2px;
  font-family: Courier;
}

.guessDot {
  padding: 5px;
  margin: 2px;
  margin-right: 40px;
}

.continueBtn {
  font-family: Helvetica;
  border: 1px solid #000;
  background-color: #0f8b8d;
  color: #FFF;
  padding: 5px;
  margin: 2px;
  text-align: center;
  display:table-cell;

  //ugh
  -moz-user-select: none;
  -webkit-user-select: none;
  -ms-user-select: none;
  user-select: none;
}

.labelContainer {
  display: flex;
  flex-direction: row;
  flex-wrap: wrap;
}

.labelTinyBox {
  flex: initial;
}

</style>


<script>

var width = window.innerWidth;
var height = 300;

var boldColor = '#0f8b8d';
var dotColor = '#dbd5b5'

var big_size_num = 13
var base_size = '2px';
var big_size = big_size_num + 'px';
this.styles = {
  'normal': {
    'fill': dotColor,
    'stroke_width': 0,
    'r': base_size,
  },
  'marked': {
    'fill': boldColor,
    'stroke_width': 0,
    'r': base_size,
  },
  'recaptured': {
    'fill': boldColor,
    'stroke_width': 3,
    'color': '#000',
    'r': big_size,
  },
  'captured': {
    'fill': dotColor,
    'stroke_width': 3,
    'color': '#000',
    'r': big_size,
  }
}

var quickWait = 100;
var normalWait = 500;
var longWait = 1000;


class ValuesAnimation {
  constructor(selection, width, height, trueN) {
    this.selection = selection;
    this.width = width;
    this.height = height;
    this.trueN = trueN;
  }

  makeLabel(div, name) {
    return div.append('td')
      .html(name + "<sub>0</sub> = ")
      .style('font-family', 'Courier')
      .style('text-align', 'right')
      .style('color', '#c2b8b2');
  }

  makeValue(div) {
    return div.append('td')
      .html("&nbsp;")
      .style('font-family', 'Courier')
      .style('color', '#c2b8b2');
  }

  draw() {
    var table = this.selection
      .style('height', this.height)
      .style('width', this.width)
      .append('div')
      .style('border', '0.5px solid #eee')
      .append('table');

    var marked = table.append('tr');
    this.markedLabel = this.makeLabel(marked, 'marked');
    this.markedValue = this.makeValue(marked);

    var captured = table.append('tr');
    this.capturedLabel = this.makeLabel(captured, 'captured');
    this.capturedValue = this.makeValue(captured);

    var recaptured = table.append('tr');
    this.recapturedLabel = this.makeLabel(recaptured, 'recaptured');
    this.recapturedValue = this.makeValue(recaptured);

    var guess = table.append('tr');
    this.guessLabel = this.makeLabel(guess, 'guess');
    this.guessValue = this.makeValue(guess);
  }

  updateThing(labelSelector, valueSelector, name, timestamp, value) {
    labelSelector.transition().duration(quickWait).style('color', "#000");
    labelSelector.html(name + "<sub>" + timestamp + "</sub> = ");
    valueSelector.html(value);
    valueSelector
      .transition()
      .duration(quickWait)
      .style('background-color', '#0f8b8d')
      .style('color', "#000")
      .on('start', function () {})
      .transition()
      .delay(quickWait)
      .duration(quickWait)
      .style('background-color', '#FFF')
      .style('color', "#000");
  }

  updateMarked(timestamp, value) {
    this.updateThing(this.markedLabel, this.markedValue, "marked", timestamp, value);
  }

  updateCaptured(timestamp, value) {
    this.updateThing(this.capturedLabel, this.capturedValue, "captured", timestamp, value);
  }

  updateRecaptured(timestamp, value) {
    this.updateThing(this.recapturedLabel, this.recapturedValue, "recaptured", timestamp, value);
  }

  updateGuess(timestamp, value) {
    this.updateThing(this.guessLabel, this.guessValue, "guess", timestamp, value);
  }

  mute() {
    this.markedLabel.transition().duration(quickWait).style('color', '#c2b8b2');
    this.markedValue.transition().duration(quickWait).style('color', '#c2b8b2');
    this.capturedLabel.transition().duration(quickWait).style('color', '#c2b8b2');
    this.capturedValue.transition().duration(quickWait).style('color', '#c2b8b2');
    this.recapturedLabel.transition().duration(quickWait).style('color', '#c2b8b2');
    this.recapturedValue.transition().duration(quickWait).style('color', '#c2b8b2');
    this.guessLabel.transition().duration(quickWait).style('color', '#c2b8b2');
    this.guessValue.transition().duration(quickWait).style('color', '#c2b8b2');
  }
}

class HistogramAnimation {
  constructor(selection, width, height, xs, trueN, range) {
    this.selection = selection;
    this.width = width;
    this.height = height;

    this.barWidth = xs[1] - xs[0];

    this.labelHeight = 30;
    this.plotHeight = this.height - this.labelHeight;

    // prepare the scales
    this.xScale = d3.scaleLinear()
      .domain([0, d3.max(xs) + this.barWidth])
      .range([0, this.width]);

    this.yScale = d3.scaleLinear()
      .domain(range)
      .range([this.plotHeight, 0]);

    this.xs = xs;

    this.attrFunc = function (x, y, hpd) {
      return {
        'fill': x == trueN ? boldColor : dotColor,
        'opacity': x >= hpd[0] && x <= hpd[1] ? 1 : 0.5
      }
    };
  }

  draw(data) {
    var xScale = this.xScale;
    var yScale = this.yScale;
    var xs = this.xs;
    var attrFunc = this.attrFunc;
    var barWidth = this.barWidth;
    var width = this.width;
    var height = this.height;
    var plotHeight = this.plotHeight;
    var labelHeight = this.labelHeight;

    this.plotSelection = this.selection.append('svg')
        .attr("width", width)
        .attr("height", plotHeight)
      .append("g")
        .attr("transform", "translate(" + 0 + "," + -labelHeight + ")");

    this.label = this.selection.append('div')
      .html("<center>P(N|D<sub>0</sub>)</center>")
      .style("font-family", "Courier")
      .attr("transform", "translate(0, " + this.labelHeight - 5 + ")")

    this.bars = this.plotSelection.selectAll("bar")
      .data(data.dist)
      .enter().append("g")
      .attr("class", "bar")
      .attr("transform", function(d, i) {
        return "translate(" + xScale(xs[i] - barWidth/2) + "," + yScale(d) + ")";
      });

    this.bars.append("rect")
      .attr("width", xScale(barWidth) - 1) // assumes equally spaced domain
      .attr("height", function(d) {
        return plotHeight - yScale(d);
      })
      .attr('fill', function(d, i) {return attrFunc(xs[i], d, data.hpd)['fill']})
      .attr('opacity', function(d, i) {return attrFunc(xs[i], d, data.hpd)['opacity']})

    this.plotSelection.append("g")
      .attr("class", "x axis")
      .attr("transform", "translate(0," + plotHeight + ")")
      .call(d3.axisBottom(xScale).tickValues(xs.filter(function(d, i) { return i % 3 == 0})));
  }

  updateDist(timestep, data, speedy) {
    var xScale = this.xScale;
    var yScale = this.yScale;
    var xs = this.xs;
    var attrFunc = this.attrFunc;
    var barWidth = this.barWidth;
    var width = this.width;
    var height = this.height;
    var label = this.label;
    var plotHeight = this.plotHeight;

    var durationLong = longWait;
    var durationNormal = normalWait;

    if (speedy) {
      durationLong /= 2
      durationNormal /=2
    }

    var animation = this.bars.data(data.dist)

    if (!speedy) {
      animation = animation
        .transition()
        .duration(durationNormal)
        .on('start', function () {
          d3.active(this).select('rect')
          .attr('fill', boldColor)
        })
    }

    animation
      .transition()
      .duration(durationLong)
      // move down
      .attr("transform", function (d, i) {
        return "translate(" + xScale(xs[i] - barWidth/2) + "," + yScale(d) + ")";
      })
      .on('start', function (d, i) {
        // update rectangle size
        d3.active(this).select('rect')
        .attr("height", function (d) {
          return plotHeight - yScale(d);
        });
        // update label
        label.html("<center>P(N|D<sub>" + timestep + "</sub>)</center>")
      })
      .transition()
      .duration(durationNormal)
      .on('start', function (d, i) {
        d3.active(this).select('rect')
        // oof be careful, this uses the i from the outside
        .attr('fill', function(d) {return attrFunc(xs[i], d, data.hpd)['fill']})
        .attr('opacity', function(d) {return attrFunc(xs[i], d, data.hpd)['opacity']})
      })
  }
}


class ScatterAnimation {
  constructor(selector, width, height) {
    this.selector = selector;
    this.width = width;
    this.height = height;

    var margin = big_size_num + 3;

    this.xScale = d3.scaleLinear()
        .domain([0, 1])
        .range([margin, width - margin]);

    this.yScale = d3.scaleLinear()
        .domain([0, 1])
        .range([height - margin, margin]);
  }

  draw(data) {
    var xScale = this.xScale;
    var yScale = this.yScale;
    var width = this.width;
    var height = this.height;
    var margin = this.margin;

    this.points = this.selector
      .append('svg')
      .attr("width", width)
      .attr("height", height)
    .append("g")
    .selectAll('.points')
      .data(data)
      .enter().append('circle')
        .attr("cx", function(d) { return xScale(d["x"]) })
        .attr("cy", function(d) { return yScale(d["y"]) })
        .attr("r", 0)
        .attr("fill", styles['normal'].fill)
        .attr("r", base_size)
        .attr('stroke', '#000')
        .attr("stroke-width", styles['normal'].stroke_width);
  }

  capture(timestep, speed) {
    this.points
      .transition()
      .duration(speed)
      .attr("r", function (d, i) { return styles[d.statuses[timestep]].r })
      .attr("stroke-width", function (d, i) { return styles[d.statuses[timestep]].stroke_width })
  }

  mark(timestep, speed) {
    this.points
      .transition()
      .duration(speed)
      .attr("fill", function (d, i) { return styles[d.statuses[timestep] == 'captured' ? 'marked' : d.statuses[timestep]].fill })
  }

  continue(speed) {
    this.points
      .transition()
      .duration(speed)
      .attr("r", function (d, i) { return base_size })
      .attr("stroke-width", styles['normal'].stroke_width)
  }
}


class DotAnimation {
  constructor(selector, width, height, maxCount, trueN) {
    this.selector = selector

    this.height = height;
    this.width = width;

    this.trueN = trueN;

    this.xScale = d3.scaleLinear()
      .domain([0, maxCount])
      .range([20, width]);
  }

  draw(data) {
    this.data = data;

    this.labels = this.selector.append('div').attr('class', 'labelContainer')

    var timestepLabel = this.labels.append('div').text("Timestep: " + 0).attr('class', 'timestepDots labelTinyBox');
    var guess = this.labels.append('div').style('font-family', 'Courier').attr('class', 'labelTinyBox').style('width', 300);

    this.graph = this.selector.append('svg')
      .attr('height', this.height)
      .attr('width', this.width);

    this.title = this.selector.append('div').attr('class', 'labelTinyBox');
    var capture_title = this.title.append("span").text("Capture");
    var mark_title = this.title.append("span").text("Mark");
    var release_title = this.title.append("span").text("Release");

    this.labels.append('div').attr('class', 'continueBtn labelTinyBox').html("Continue &#x25B6;");


    this.highlightTitle = function (title) {
      this.title
        .selectAll('span')
        .attr('class', 'hiddenDotsTitle');

      var mapping = {
        "capture": capture_title,
        "mark": mark_title,
        "release": release_title,
      }

      mapping[title].attr('class', 'highlightedDotsTitle')
    }

    this.setTimestep = function (timestep) {
      timestepLabel.text("Timestep: " + timestep)
    }

    this.setGuess = function (timestep, highlight) {
      guess.html("guess<sub>" + timestep + "</sub>: " + this.data[timestep].guess[0] + " - " + this.data[timestep].guess[1] + " (actual: " + this.trueN + ")")
        .attr("class", "guessDot")

      if (highlight) {
        guess
          .transition()
          .style('background-color', boldColor)
          .transition()
          .style('background-color', '#fff')
      }
    }

    this.setGuess(0, false);

    var capture = this.capture.bind(this);
    this.selector.on('click', function () { capture(0) })
  }

  capture(timestep) {
    var xScale = this.xScale;
    var height = this.height;
    var setGuess = this.setGuess.bind(this);

    var numDots = this.data[timestep].labels.length;

    if (timestep >= this.data.length) {
      timestep = 0
    }

    this.setTimestep(timestep)
    this.highlightTitle('capture')

    var datagraph = this.graph.selectAll('.points')
      .data(this.data[timestep].labels)
      .enter().append('circle')
        .attr("cx", function(d, i) { return xScale(i) })
        .attr("cy", height/2)
        .attr("r", 0)
        .attr("fill", function (d) { return styles[d].fill })
        .attr('stroke', '#000')
        .attr("stroke-width", styles['normal'].stroke_width)
        .transition()
        .duration(500)
        .attr('r', big_size_num + 'px')
        .attr("stroke-width", function (d, i) { return styles[d].stroke_width })
      .on('end', function() { setGuess(timestep, timestep != 0)});

    this.points = this.graph.selectAll('circle');

    var mark = this.mark.bind(this);
    this.selector.on('click', function () { mark(timestep) })
  }

  mark(timestep) {
    this.points
      .transition()
      .duration(500)
      .attr("fill", styles['marked'].fill )

    this.highlightTitle('mark')

    var continueFunc = this.continueFunc.bind(this);

    this.selector.on('click', function () { continueFunc(timestep) })
  }

  continueFunc(timestep) {
    var graph = this.graph;

    this.highlightTitle('release')

    this.graph.selectAll('circle')
      .transition()
      .duration(500)
      .attr("stroke-width", 0)
      .attr('r', base_size)
      .attr('fill', '#fff')
      .on('end', function() { graph.selectAll('circle').remove();})

      var capture = this.capture.bind(this);

      this.selector.on('click', function () { capture(timestep + 1) })
  }
}


function expandSequenceData(data) {
  // Come up with locations for points
  var expandedData = d3.range(data.trueN).map(function() {
    return {
      "x": Math.random(),
      "y": Math.random(),
      "statuses": d3.range(data.sequence.length).map(function() {return "normal"})
    }
  });

  // iterate through sequence and check status, either "recaptured", "captured", "marked", or "normal"
  data.sequence.forEach(function(sample, time_i) {
      sample.captured.forEach(function(recaptured_i) {
          expandedData[recaptured_i]['statuses'][time_i] = 'captured'
      })

      sample.captured.forEach(function(captured_i) {
          for (var i = time_i + 1; i < data.sequence.length; i++) {
              expandedData[captured_i]['statuses'][i] = 'marked'
          }
      })
      sample.recaptured.forEach(function(recaptured_i) {
          expandedData[recaptured_i]['statuses'][time_i] = 'recaptured'
      })
  })

  return expandedData;
}

function makeCirclesSequenceData(data) {
  return data.sequence.map(function(sample, time_i) {
    return {
      'guess': sample.hpd,
      'labels': sample.captured.map(function(captured_i) {
        return sample.recaptured.indexOf(captured_i) == -1 ? 'captured' : 'recaptured'
      })
    }
  })
}

function introPlot(data) {
  var graph = d3.select("#intro");

  var trueN = data.trueN;
  var data =  makeCirclesSequenceData(data);
  var animator = new DotAnimation(graph, 720, 60, d3.max(data.map(function(d) { return d.labels.length })), trueN);

  animator.draw(data);
  animator.capture(0);  // begin interactive animation
}

function fancyPlot(data) {

  var graph = d3.select("#graph")
    .style('display', 'flex')
    .style('flex-wrap', 'wrap')

  var leftWidth = 320;
  var rightWidth = 320

  startBtn = graph.append('div')
    .attr('class', 'continueBtn')
    .style('width', width)
    .style('height', height)
    .style('opacity', 1)
  startBtn
    .append('p')
    .style('margin-top', '20px')
    .style('font-size', '24px')
    .text('start!');

  var leftPanel = graph.append("div").style("width", leftWidth + 'px');
  var rightPanel = graph.append("div").style("width", rightWidth + 'px');

  var numbers = leftPanel.append("div");
  var distPlot = leftPanel.append('div');
  var scatter = rightPanel.append("div");

  var numDrawer = new ValuesAnimation(
    numbers,
    leftWidth - 10,
    height/3,
    data.trueN,
  )

  var distDrawer = new HistogramAnimation(
    distPlot,
    rightWidth,
    height * 2/3,
    data.distDomain,
    data.trueN,
    [0, d3.max(data.sequence.map(function(x) { return d3.max(x.dist) }))],
  )

  var scatterDrawer = new ScatterAnimation(
    scatter,
    leftWidth - 10,
    height,
  )

  var expandedData = expandSequenceData(data);

  var steps = data.sequence.length;
  function next(timestep, speedy) {

    var durationLong = longWait;
    var durationNormal = normalWait;
    var durationQuick = quickWait;

    var animation = graph
      .transition()
      .duration(durationLong)
      .delay(1000)
      .on("start", function() {
        scatterDrawer.capture(timestep, normalWait)
      })

    if (speedy) {
      animation = animation.transition()
        .duration(durationNormal * 2)
        .on("start", function() {
          numDrawer.updateMarked(timestep, data.sequence[timestep].M)
          numDrawer.updateCaptured(timestep, data.sequence[timestep].C)
          numDrawer.updateRecaptured(timestep, data.sequence[timestep].R)
          distDrawer.updateDist(timestep, data.sequence[timestep], speedy)
          numDrawer.updateGuess(timestep, "" + data.sequence[timestep].hpd[0] + '-' + data.sequence[timestep].hpd[1])
        })
    } else {
      animation = animation
        .transition()
        .duration(durationNormal)
        .on("start", function() { numDrawer.updateMarked(timestep, data.sequence[timestep].M) })
        .transition()
        .duration(durationNormal)
        .on("start", function() { numDrawer.updateCaptured(timestep, data.sequence[timestep].C) })
        .transition()
        .duration(durationNormal)
        .on("start", function() {numDrawer.updateRecaptured(timestep, data.sequence[timestep].R) })
        .transition()
        .duration(durationLong + durationNormal * 3)
        .on('start', function () {
          distDrawer.updateDist(timestep, data.sequence[timestep], speedy)
        })
        .transition()
        .duration(durationNormal)
        .on('start', function () {
          numDrawer.updateGuess(timestep, "" + data.sequence[timestep].hpd[0] + '-' + data.sequence[timestep].hpd[1])
        })
    }

    animation = animation
      .transition()
      .duration(durationNormal)
      .on('start', function() { scatterDrawer.mark(timestep, durationNormal); })

    if (speedy) {
      animation
        .transition()
        .duration(durationNormal)
        .on('start', function() {
          scatterDrawer.continue(durationNormal);
        })
    } else {
      animation = animation.transition()
        .duration(durationNormal)
        .on('start', function() { scatterDrawer.continue(durationNormal); })
        .transition()
        .delay(durationLong)
        .duration(durationNormal)
        .on('start', function() { numDrawer.mute(); })
    }

    animation.on('end', function () {
      if (timestep <= 2) {
        next(timestep + 1, false);
      }
      else if (timestep + 1 < steps) {
        next(timestep + 1, true);
      }
    })
  }

  // start!
  graph.on('click', function() {
    graph.transition()
      .style('opacity', 0)
      .on('end', function() {
        startBtn.remove();

        scatterDrawer.draw(expandedData)
        distDrawer.draw(data.sequence[0])
        numDrawer.draw();
        graph.transition()
          .duration(longWait)
          .style('opacity', 1)
          .style('height', (50 + height) + 'px')
          .transition()
          .delay(normalWait)
          .duration(normalWait)
          .on("start", function() { scatterDrawer.capture(0, normalWait); })
          .transition()
          .duration(quickWait)
          .on('start', function() { scatterDrawer.mark(0, normalWait); })
          .transition()
          .delay(normalWait)
          .duration(quickWait)
          .on('start', function() { scatterDrawer.continue(normalWait); })
          .on('end', function () {
            next(1, false);
          });

        graph.on('click', function () {})
      });
  })
}

function plot(data) {
  introPlot(data);
  fancyPlot(data);
}

var simulation_data = d3.json("/data/20190810_population_estimate.json");
simulation_data.then(function (data) {plot(data)});

</script>



