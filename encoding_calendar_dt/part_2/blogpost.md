# Encoding Temporal Features (Part 2)
## How to teach seasonality to Deep Neural Networks (DNN)
---

In my last blog post I presented a method to encode *calendar_dt* in such a way that DNNs are capable of learning public holiday effects. Moreover, it was even possible learn short term seasonality like weekly pattern. In this blog post I'd like to switch focus from this short term seasonality to long term patterns. Long term seasonality is quite common in real world data. To name some examples: The average temperature per day (unless you live in the equatorial area), solar eruption or even people behavior which can be quite seasonal, e.g. the number of cinema visitors decreases in summer time.

But how can we teach a DNN seasonal effects, such that its predictions show them as well? 
# Common Approaches
There are basically two types of methods to handle long-term seasonality in time series; decomposition and encoding. Encoding methods create features based on the input date, while decomposition methods focus on filter out the seasonality part of the signal. In this post we will focus on encoding methods.

A widely used encoding technique splits the *calendar_dt* into several date parts like year, month and day_of_year. A DNN can then learn the seasonality effect using these features.

The advantage of this approach is that it doesn't require any domain knowledge, since the DNN will try to extract the information directly from the signal. Like everything in life this advantage comes with a price. In this case it's the shir amount of data needed for training especially when dealing with yearly periodic seasonality. This becomes clear when imagine a time series having similar values in January and December. To learn such a pattern, the DNN need to combine totally different encoded input dates.

At Adtrac, we found a way to cope with such problems by helping the DNN with more informative features. Let's have a look how this works.
# Features based on Similar Days
The base idea of our encoding is that we introduce the similarity of different dates. This similarity is defined across several dimensions. Thus, days next to each others should have similar values across all dimensions, while days split by months should share less feature values. So how can we achieve this?
## Set Anchors based on Prior
In a first step, we use domain knowledge to define relevant time ranges. So for example if our time series shows monthly pattern, we will use months as relevant time zones. On the other hand, if we think our data correlates closer with seasons we will define our time ranges as spring, summer, autumn and winter.

Unlike the common approaches this split requires some domain knowledge or some time to find an appropriate split by running some experiments.

Having set relevant time ranges we set an anchor for each range. Usually we use the first day of the time range as anchor, but it could be an other date within this range.

But how does this help?

## Calculate Values for Each Anchor
Using our prior anchors, we will encode *calendar_dt* by calculating a particular value for each of these date ranges. Let's make an example for a seasonal prior. Let's assume we want to encode the calendar_dt *2021-05-10*, hence we need to calculate how spring-ish, how summer-ish, how autumn-ish and how winter-ish this particular date is. 

In order to do so, we will use a mixture of three normal distributions, where we use the corresponding day_of_year of the input as the mean (and shifted by period length). Moreover, the standard deviation is just set as the length of the date range. 

Picture

In order to get a value for each date range we just take the sum over all days within this range. 

## Achievement
When we apply this method to some sample inputs we see that close days have almost the same features while far days look differently across the dimensions. Therefore, this encoding helps a DNN to distinguish days across this dimensions and hence help learning it seasonality effects.

Some words about the costs of this encoding, because one could argue its computational intensive calculating *period_lenght*-values of a Gaussian Mixture for each input. That's true to some extent. It's indeed more expensive than the common approaches, although it's still O(n). 
One trick to speed up the encoding process is to pre-calculate all possible calendar_dts. So, at Adtrac, we calculated all values for the next 10 years and just join the corresponding values whenever needed. 

---
# Conclusion
Seasonality can be problem for DNNs, especially if you haven't
In this blog post we have seen different encoding methods for the input *calendar_dt*. Common approaches focuses on splitting the date into its parts like year and month. Unfortunately, this approach requires an enormous of data for training a DNN.
Therefore, we at Adtrac came up with an other method, which requires fewer data for the price of some domain knowledge. The main idea of this encoding is to encode close days similar across several dimensions. This allows a DNN to easily learn e.g. how spring looks like.
