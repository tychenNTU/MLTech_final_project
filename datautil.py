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
def get_the_data_preprocessor(num_features=num_features, cat_features=cat_features, std_num_feature=True):
    """
    This function will return return a scikit-learn ColumnTransformer that process the data and
    the spec of the data should contain to be processed by the returned transformer.
    You can denote the spec of the preprocessed data by passing num_features (numerical data) 
    or cat_features (categorical data).
    By default, the preprocessor will standardize the numerical data when using transform(), you can 
    remove this process by setting std_num_feature=False.
    """
    # indicate the training features contained in the input data
    features_spec = num_features + cat_features

    # preprocess numerical feats:
    # for most num cols, except the dates, 0 is the most logical choice as fill value
    # and here no dates are missing.
    num_transformer = SimpleImputer(strategy="constant")
    num_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant")),
        ("std_scaler", StandardScaler()),
    ]) if std_num_feature else Pipeline(steps=[("imputer", SimpleImputer(strategy="constant"))])

    # Preprocessing for categorical features:
    cat_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("onehot", OneHotEncoder(handle_unknown='ignore'))])

    # Bundle preprocessing for numerical and categorical features:
    preprocessor = ColumnTransformer(transformers=[("num", num_transformer, num_features),
                                                ("cat", cat_transformer, cat_features)])

    return preprocessor, features_spec


# return a copy of the input dataframe that contains new columns "arrival_date"
def get_arrival_date_column(input_data, month_to_num=False):
    """
    This function will return a dataframe with an additional column "arrival_date".
    The format of arrival_date: %y-%m-%d.
    Warning: The input_data must have the following columns:
        1. "arrival_date_day_of_month": str or int
        2. "arrival_date_year": str or int
        3. "arrival_date_month": str of the name of the month (e.g. July)
    """

    month_dict = {'January' : '1',
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
            'December' : '12'}
    def combine_to_date(row):
    # year, month, day -> y-m-d
        date = row['arrival_date_day_of_month']
        date = '0' + str(int(date)) if int(date) < 10 else str(date)
        return str(row['arrival_date_year']) + "-" \
        + month_dict[row['arrival_date_month']] + "-" \
        + date
    # extract some informations that are useful to compute revenue
    new_df = input_data.copy()

    # month names converted to the numbers
    new_df['arrival_date'] = "" # format: y-m-d

    # additional informations
    new_df['arrival_date'] = new_df.apply(combine_to_date, axis=1)
    if month_to_num:
        new_df['arrival_date_month'].replace(month_dict, inplace=True)
    return new_df


def get_date_difference_col(input_data):
    """
    This function add a new column called date_difference.
    The input data must contain "arrival_date" and "reservation_status_date"
    column value = date1 - date2 (unit: days)
    date1: arrival_date
    date2: reservation_status_date
    """
    def compute_date_difference(row):
        arrival_date = datetime.strptime(row["arrival_date"], "%Y-%m-%d")
        reservation_status_date = datetime.strptime(row["reservation_status_date"], "%Y-%m-%d")
        date_diff = (arrival_date - reservation_status_date).days
        return date_diff
    result_data = input_data.copy()
    result_data["date_diff"] = result_data.apply(compute_date_difference, axis=1)
    for _, row in result_data.iterrows():
        row["date_diff"] = compute_date_difference(row)
    return result_data


def get_weekday_col(input_data):
    """
    This functino add a new "weekday" (day of the week) column from arrival_date column.
    """
    result_data = input_data.copy()
    result_data["weekday"] = result_data["arrival_date"].apply(lambda date: datetime.strptime(date, "%Y-%m-%d").weekday())
    return result_data


def add_group_by_mean_col(input_data, by_col, target_col, group_na=False, avg_target_val_dict=None, na_val=-1):
    """
    This function will add a new row using .groupby() function.
    It will return two objcets:
    1. result_data: the dataframe with the new compute column
    2. the dictionary that maps by_col column value to the target_col column value
    """
    result_data = input_data.copy()
    if avg_target_val_dict is None:
        if group_na:
            result_data = result_data.fillna(na_val)
            group_by_object = result_data.groupby(by_col)
        else: group_by_object = result_data.groupby(by_col)
        avg_target_val_dict = dict(group_by_object[target_col].mean())
    result_data[by_col + "_avg_" + target_col] = result_data[by_col].map(avg_target_val_dict)
    return result_data, avg_target_val_dict


# functions to compute the daily revenue
def get_revenue_df(input_data):
    """
    This function will compute the daily revenue of the input dataframe.
    The input dataframe must contain the following data columns:
        * 'arrival_date_year'
        * 'arrival_date_month'
        * 'arrival_date_day_of_month'
        * 'adr' 
        * 'stays_in_weekend_nights'
        * 'stays_in_week_nights'
        * 'is_canceled'
    """
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
    adr_df = get_arrival_date_column(adr_df, month_to_num=True)

    # add some additional columns
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


def split_data_by_date(input_data, val_ratio, rand_seed=42, add_arrival_date=False):
    """
    This function will choose the orders of specific days (ramdomly selected) as validation set.
    The remaining orders will be left as training data.
    """
    # preprocess
    data = input_data.copy()
    arrival_date_in_df = True
    if 'arrival_date' not in data.columns:
        data = get_arrival_date_column(data)
        arrival_date_in_df = False
    date_list = data['arrival_date'].unique()

    # split the data
    np.random.seed(42)
    n_data = len(date_list)
    shuffled_indice = np.random.permutation(n_data)
    n_valid = int(n_data * val_ratio)
    val_indice = shuffled_indice[:n_valid]
    val_date = set(date_list[val_indice])
    val_data = data[data['arrival_date'].isin(val_date)]
    train_data = data[~data['arrival_date'].isin(val_date)]
    if not arrival_date_in_df and not add_arrival_date:
        val_data = val_data.drop(['arrival_date'], axis=1)
        train_data = train_data.drop(['arrival_date'], axis=1)
    return train_data, val_data

def get_preprocessed_xy():
    full_data = pd.read_csv('../train.csv')
    full_data = full_data[(full_data['adr'] < 1000) & (full_data['adr'] > -100)] # remove outliers
    full_data = full_data[full_data['is_canceled'] == 0] # only use the uncanceled orders to train

    # get the preprocessor and the default training features
    preprocessor, features_spec = get_the_data_preprocessor()

    # split data into input and label
    X_train_full_raw = full_data[features_spec]
    y_train_full = np.array(full_data['adr'])
    X_transformed = preprocessor.fit_transform(X_train_full_raw)
    return X_transformed, y_train_full
