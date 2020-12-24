# common:
import pandas as pd
import numpy as np
from datetime import datetime
from datetime import date
# for ML:
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

# default numerical columns
num_features = ["lead_time","arrival_date_week_number","arrival_date_day_of_month",
                "stays_in_weekend_nights","stays_in_week_nights","adults","children",
                "babies","is_repeated_guest", "previous_cancellations",
                "previous_bookings_not_canceled","agent","company",
                "required_car_parking_spaces", "total_of_special_requests"]

cat_features = ["hotel","arrival_date_month","meal","market_segment",
                "distribution_channel","reserved_room_type","deposit_type","customer_type"]

# this function will generate a data preprocessor corresponding to the given training features
def get_the_data_preprocessor(num_features=num_features, cat_features=cat_features):

    # indicate the training features contained in the input data
    features_spec = num_features + cat_features

    # preprocess numerical feats:
    # for most num cols, except the dates, 0 is the most logical choice as fill value
    # and here no dates are missing.
    num_transformer = SimpleImputer(strategy="constant")
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant")),
        ("std_scaler", StandardScaler()),
    ])

    # Preprocessing for categorical features:
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))])

    # Bundle preprocessing for numerical and categorical features:
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                                ("cat", cat_transformer, cat_features)])

    return preprocessor, features_spec


# functions to compute the daily revenue
def get_revenue_df(input_data):
    def combine_to_date(row):
    # year, month, day -> y-m-d
        date = row['arrival_date_day_of_month']
        date = '0' + str(date) if int(date) < 10 else str(date)
        return str(row['arrival_date_year']) + "-" \
        + str(row['arrival_date_month']) + "-" \
        + date

    def stay_nights_sum(row):
        return row['stays_in_weekend_nights'] + row['stays_in_week_nights']
    
    # extract some informations that are useful to compute revenue
    adr_df = input_data[['arrival_date_year', 
                            'arrival_date_month',
                            'arrival_date_day_of_month',
                            'adr', 
                            'stays_in_weekend_nights',
                            'stays_in_week_nights',
                            'is_canceled'
                            ]].copy()

    # month names converted to the numbers
    adr_df['arrival_date_month'].replace({'January' : '1',
            'February' : '02',
            'March' : '03',
            'April' : '04',
            'May' : '05',
            'June' : '06',
            'July' : '07',
            'August' : '08',
            'September' : '09', 
            'October' : '10',
            'November' : '11',
            'December' : '12'}, inplace=True)
    adr_df['arrival_date'] = "" # format: y-m-d

    # additional informations
    adr_df['arrival_date'] = adr_df.apply(combine_to_date, axis=1)
    adr_df['tm_yday'] = adr_df['arrival_date'].apply(lambda input_date: datetime.strptime(input_date,"%Y-%m-%d").timetuple().tm_yday)
    adr_df['tm_wday'] = adr_df['arrival_date'].apply(lambda input_date: datetime.strptime(input_date,"%Y-%m-%d").timetuple().tm_wday)
    adr_df['stay_nights'] = adr_df.apply(stay_nights_sum, axis=1)

    revenue_df = pd.DataFrame(columns=['arrival_date', 'revenue', 'year', 'month', 'month_day', 'tm_yday'])
    def compute_single_venue(row):
        adr = row['adr']
        nights = row['stay_nights']
        is_canceled = row['is_canceled']
        if not is_canceled:
            return adr * nights
        else:
            return 0
    revenue_df.set_index = revenue_df['arrival_date']
    for _, row in adr_df.iterrows():
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
    return revenue_df