# Report

# What we have done
## Revenue Formula Derivation
We tried several formulas for the revenue calculation. First, we tried `'Non refund' x reservation_status_date` and `ADR x (week nights + weekend nights)` to include the bookings that has their deposit already paid upon reservation. However, we get the following chart after matching it with the labeled revenue data. The red line is our output while the blue line is the labeled data.

![](./img/rev_formula1.png)

Then, we obtained a better result when using `ADR x (week nights + weekend nights)` as follows:

![](./img/rev_formula2.png)

It seems that the outcome of this formula perfectly fits the data. We also write a program to find the actual boundary. The result shows that the numerical boundaries of the labels are 10000, 20000 , and so on. Therefore, we can transform the original ordinal classification problem to a regression problem without loss of generality. This will make the task more easier since there are many algorithms and models designed for regression problems.

## Detection/Dropping of Outlier
### Abnormal ADR
When observing the ADR of each order, we found that there are some data with abnormal ADR values. First, all the values of ADR are within the range from -200 to 500, except one data with an ADR value of nearly 5400, which is very likely to be an outlier that may influence the prediction of ADR. Therefore, we decided to remove this data. On the other hand, we note that some orders have extremely negative ADR, which may make the daily revenue drop dramatically. To confirm this, we plot the daily revenue diagram and mark the dates with this property:

![](./img/abnormal_single_cost_plot_2016.png)

As we can see from the diagram above, the revenue in 2016/1/26 decreased to nearly zero just because of the single abnormal order. Moreover, we also marked the date of the orders with extremely high revenue (more than 3000), see the orange dots in the figure above. As a result, these orders don't cause any obvious jump. Therefore, we decide to remove the order with extremely negative ADR from the training data.

## Model Building and Evaluation
After discussion, we came up with two main ideas for the framework of model. First, an intuitive way to train the models and make prediction is to treat each order as single input data. In this way, we can convert the original ordinal ranking problem to two subproblems: a regression problem (i.e., predict the correct ADR value for each order) and a traditional binary classification problem (i.e., predict if an order is canceled or not). These two problems are what we are familiar to in this course; therefore, we can utilize the algorithms such as perceptron learning algorithm, support vector machine, etc. Moreover, since both of these two problems are well-known machine learning problem, there are many existing well-developed algorithms that solve two problems for us to explore. Based on the above reasons, it's a possibly method to be implemented and analyzed.
However, the framework we mentioned above has a possible drawback: it focuses on predicting every order's ADR and whether it's canceled without considering other orders in the same day as a whole. When using the ADR and `is_canceled` predicted by model to compute the total daily revenue, we just summing up each single revenue. This means that we may lose some information of the whole day. For this reason, we considered another approach as a possible solution: predict the daily revenue using the information of the whole day. There are several methods to do handle this problem. For example, we use seq2seq, which is a architecture of recurrent neural network, to predict the daily revenue based on previous daily revenue and some other information.
### Seq2Seq
### Random Forest
### Neural Network 
