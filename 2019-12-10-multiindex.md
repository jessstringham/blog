---
title: 'Pandas MultiIndex cheatsheet'
tags: [pandas]
layout: post
display_image: 2019-12-10-dataframe.png
---

This is a quick cheatsheet on creating and indexing with [Pandas MultiIndexes](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html).

## Guide

I'm representing values in the DataFrame using symbols. For a DataFrame like the one to the left, the representation I'll use is the one on the right.

<table>
<tr>
<td><img src="/assets/2019-12-10-pandas-dataframe.png"></td>
<td><img src="/assets/2019-12-10-dataframe.png"></td>
</tr>
</table>

For example, `A` in the sample `DataFrame` is represented by the blue rectangle that look like <img style='width:18px' src="/assets/2019-12-10-A.png"> in the images. As additional examples, `B` in the sample `DataFrame` is represented by <img style='width:18px' src="/assets/2019-12-10-B.png">, the row `one` is <img style='width:18px' src="/assets/2019-12-10-one.png"> and `two` is <img style='width:18px' src="/assets/2019-12-10-two.png">.

## Creating MultiIndexes

Four ways to create `MultiIndex`es are `pd.MultiIndex.from_tuples`, `pd.MultiIndex.from_product`, `pd.MultiIndex.from_arrays`, and `pd.MultiIndex.from_frame`.

<table>
<tr>
 <td><pre>pd.MultiIndex.from_tuples</pre></td>
 <td><img style="width:320px" src="/assets/2019-12-10-from-tuples.png"></td>
</tr>
<tr>
 <td><pre>pd.MultiIndex.from_arrays</pre></td>
 <td><img style="width:320px" src="/assets/2019-12-10-from-arrays.png"></td>
</tr>
<tr>
 <td><pre>pd.MultiIndex.from_product</pre></td>
 <td><img style="width:320px" src="/assets/2019-12-10-from-product.png"></td>
</tr>
<tr>
 <td><pre>pd.MultiIndex.from_frame</pre> <small>The column names become the names of the levels.</small></td>
 <td><img style="width:320px" src="/assets/2019-12-10-from-frame.png"></td>
</tr>
</table>


## Creating a sample DataFrame with MultiIndex

You can create a `DataFrame` with `MultiIndex`es as both the index and the columns.

<img style="width:540px" src="/assets/2019-12-10-multiindex-input.png">

<img src="/assets/2019-12-10-df-func.png">

## MultiIndex from a group by

You can create up with a `DataFrame` with a `MultiIndex` by using a `pivot` or `groupby` using multiple columns.

<img src="/assets/2019-12-10-groupby.png">


## Accessing columns

This section describes how to select different columns and rows using the `MultiIndex`es.
This section uses the `DataFrame` below, which has a MultiIndex as the columns and rows.

<img style="width:320px" src="/assets/2019-12-10-df-equals.png">


### Columns

<table>
<tr>
<td><img src="/assets/2019-12-10-cols-1.png"></td>
</tr>
<tr>
<td><img src="/assets/2019-12-10-cols-2.png"></td>
</tr>
<tr>
<td><img src="/assets/2019-12-10-cols-3.png"></td>
</tr>
<tr>
<td><img src="/assets/2019-12-10-cols-4.png"></td>
</tr>
</table>

### Rows

<table>
<tr>
<td><img src="/assets/2019-12-10-rows-1.png"></td>
</tr>
<tr>
<td><img src="/assets/2019-12-10-rows-2.png"></td>
</tr>
<tr>
<td><img src="/assets/2019-12-10-rows-3.png"></td>
</tr>
<tr>
<td><img src="/assets/2019-12-10-rows-4.png"></td>
</tr>
</table>
