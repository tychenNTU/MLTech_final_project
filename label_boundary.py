import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from datetime import date

raw_train_data = pd.read_csv("./train.csv")
raw_test_data = pd.read_csv("./test.csv")
train_data = raw_train_data.copy()

# extract some informations that are useful to compute revenue
adr_df = raw_train_data[['arrival_date_year', 
                         'arrival_date_month',
                         'arrival_date_day_of_month',
                         'adr', 
                         'stays_in_weekend_nights',
                         'stays_in_week_nights',
                         'is_canceled',
                         'deposit_type',
                         'reservation_status_date']].copy()
# month names converted to the numbers
adr_df['arrival_date_month'].replace({'January' : 1,
        'February' : 2,
        'March' : 3,
        'April' : 4,
        'May' : 5,
        'June' : 6,
        'July' : 7,
        'August' : 8,
        'September' : 9, 
        'October' : 10,
        'November' : 11,
        'December' : 12}, inplace=True)
adr_df['arrival_date'] = "" # format: y-m-d

def combine_to_date(row):
    # year, month, day -> y-m-d
    year = str(row['arrival_date_year'])
    month = str(row['arrival_date_month']) if row['arrival_date_month'] >= 10 \
        else '0' + str(row['arrival_date_month'])
    day = str(row['arrival_date_day_of_month']) if row['arrival_date_day_of_month'] >= 10\
        else '0' + str(row['arrival_date_day_of_month'])
    return year + '-' + month + '-' + day

def stay_nights_sum(row):
    return row['stays_in_weekend_nights'] + row['stays_in_week_nights']

# additional informations
adr_df['arrival_date'] = adr_df.apply(combine_to_date, axis=1)
adr_df['tm_yday'] = adr_df['arrival_date'].apply(lambda input_date: datetime.strptime(input_date,"%Y-%m-%d").timetuple().tm_yday)
adr_df['tm_wday'] = adr_df['arrival_date'].apply(lambda input_date: datetime.strptime(input_date,"%Y-%m-%d").timetuple().tm_wday)
adr_df['stay_nights'] = adr_df.apply(stay_nights_sum, axis=1)


# compute the revenue
# this dataframe is used to store the revenue for each day
revenue_df = pd.DataFrame(columns=['date', 'revenue', 'year', 'month', 'month_day'])

def compute_single_venue(row):
    adr = row['adr']
    nights = row['stay_nights']
    is_canceled = row['is_canceled']
    if not is_canceled:
        return adr * nights
    else:
        return 0
revenue_df = pd.DataFrame(columns=['arrival_date', 'revenue', 'year', 'month', 'month_day', 'tm_yday'])
revenue_df.set_index = revenue_df['arrival_date']
for index, row in adr_df.iterrows():
    if row['arrival_date'] not in revenue_df.index.values:
        # create the new row
        new_row = pd.DataFrame()
        new_row['arrival_date'] = [row['arrival_date']]
        new_row['revenue'] = [compute_single_venue(row)]
        new_row['year'] = [row['arrival_date_year']]
        new_row['month'] = [row['arrival_date_month']]
        new_row['month_day'] = [row['arrival_date_day_of_month']]
        new_row['tm_yday'] = new_row['arrival_date'].apply(lambda input_date: datetime.strptime(input_date,"%Y-%m-%d").timetuple().tm_yday)
        # append the row into the revenue dataframe
        new_row.index = new_row['arrival_date']
        revenue_df = pd.concat([revenue_df, new_row])
    else:
        revenue_df.loc[row['arrival_date'], 'revenue'] += [compute_single_venue(row)]

# insert the label into the table
label_data = pd.read_csv("train_label.csv")
revenue_df['label'] = np.nan
for idx, row in label_data.iterrows():
    revenue_df.loc[row['arrival_date'], 'label'] = row['label']

# sort the revenue_df by the value of revenue
sorted_revenue_df = revenue_df.sort_values(by='revenue')

# find the boundaries of the label
cur_label = sorted_revenue_df.iloc[0]['label']
cur_revenue = sorted_revenue_df.iloc[0]['revenue']
range_boundaries = []
for idx, row in sorted_revenue_df.iterrows():
    if cur_label != row['label']:
        range_boundaries.append((row['revenue'] + cur_revenue) / 2)
    cur_revenue = row['revenue']
    cur_label = row['label']
print("label boundaries:", range_boundaries)