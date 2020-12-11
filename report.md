# Report

# What we have done
## Revenue Formula Derivation
1. We tried several formulas for the revenue calculation. First, we tried `'Non refund' x reservation_status_date` and `ADR x (week nights + weekend nights)` to include the bookings that has their deposit already paid upon reservation. However, we get the following chart after matching it with the labeled revenue data. The red line is our output while the blue line is the labeled data.

![](./img/rev_formula1.png)

2. Then, we obtained a better result when using `ADR x (week nights + weekend nights)` as follows:

![](./img/rev_formula2.png)
