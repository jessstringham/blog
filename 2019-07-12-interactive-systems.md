---
title: "Interactive Graph: Renewable Resource System Model output"
tags: [projects, interactive, quick]
layout: post
display_image: 2019-07-10-ex.png
---

I want to start trying to make interactive plots for my posts.
Learning how to produce usable interactive graphics at that quality is going to take me a while, so here's my first clumsy step!


One of the [systems modeling]({% post_url 2019-07-01-systems-modeling-from-scratch %}) examples
was a system with a renewable resource. Holding everything else in the system, one function, `yield_per_unit_capital_given_resource`, could determine whether the resource stayed stable, oscillated, or dies out.
Adjust the slider to see the effect.


<div id="graph"></div>

<script src="/assets/js/d3.v5.min.js"></script>


I produced the underlying data using [this script](https://github.com/jessstringham/notebooks/blob/master/scripts/run_simulation_for_d3.py).


<style type="text/css">
.lineyield {
    fill: none;
    stroke: #eeaa00;
    stroke-width: 3;
    stroke-linejoin: round;
}

.lineres {
    fill: none;
    stroke: #336699;
    stroke-width: 3;
    stroke-linejoin: round;
}

.linecap {
    fill: none;
    stroke: #aacc99;
    stroke-width: 3;
    stroke-linejoin: round;
}

#param_selector {
  width: 300px;
  text-align: center;
}

</style>


<script>

// This code based (or erm, mostly copied) on https://bl.ocks.org/gordlea/27370d1eea8464b04538e6d8ced39e89
var margin = {top: 50, right: 50, bottom: 50, left: 50}
  , fullwidth = window.innerWidth - margin.left - margin.right // Use the window's width
  , fullheight = 400;

var width = fullwidth / 6;
var height = fullheight / 3;


function draw_graph(graph, identifier, title) {

    var xScale = d3.scaleLinear()
        .domain(graph.x_domain)
        .range([0, width]);

    var yScale = d3.scaleLinear()
        .domain(graph.y_domain)
        .range([height, 0]);

    var svg = d3.select("#graph").append("svg")
      .attr("width", width + margin.left + margin.right)
      .attr("height", height + margin.top + margin.bottom)
    .append("g")
      .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

    // draw title
    svg.append("text")
      .attr("x", width / 2 )
      .attr("y", -10)
      .style("text-anchor", "middle")
      .style("font-family", 'helvetica')
      .style('font-size', '11pt')
      .text(title);

    // draw axes
    svg.append("g")
        .attr("class", "x axis")
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(xScale)); // Create an axis component with d3.axisBottom

    svg.append("g")
        .attr("class", "y axis")
        .call(d3.axisLeft(yScale)); // Create an axis component with d3.axisLeft

    function draw_line(measure) {
      var dataset = graph.ys_by_yield_parameter[measure]

      d3.select("." + identifier).remove();

      var line = d3.line()
          .x(function(d, i) { return xScale(graph.xs[i]); })
          .y(function(d) { return yScale(d); })

      svg.append("path")
          .datum(dataset)
          .attr("class", identifier)
          .attr("d", line);
    }

    return draw_line;
}


function make_all_graphs(data) {
  var draw_line_yield = draw_graph(
    data.yield_graph,
    'lineyield',
    'Yield per unit capital given resource'
  );
  var draw_line_res = draw_graph(
    data.yield_simulated_capital,
    'linecap',
    'Capital over time'
    );
  var draw_line_cap = draw_graph(
    data.yield_simulated_resource,
    'lineres',
    'Resource over time'
  );

  function draw_all(measure) {
    draw_line_res(measure);
    draw_line_cap(measure);
    draw_line_yield(measure);
  }

  var slider_bar = d3.select("#graph").insert("div", ":first-child")

  slider_bar.append('input')
    .attr('type', 'range')
    .attr('id', 'param_selector')
    .attr('name', 'yield_parameter')
    .attr('min', 0)
    .attr('max', data.yield_parameters.length - 1)
    .attr('value', 10)
    .on('click', function() { draw_all(data.yield_parameters[this.value]) });

  slider_bar.append('label')
    .attr('for', 'yield_parameter')
    .text('Technological efficiency');

  draw_all(0.4473684210526315);
}

var simulation_data = d3.json("/data/20190710_simulation.json");
simulation_data.then(function (data) {make_all_graphs(data)});

</script>

