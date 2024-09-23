import numpy as np
from datetime import datetime

def reduce_mem_usage(df):
    '''
        This code is use to down cast the dtype of different columns of the dataframe.
        Ref: https://mikulskibartosz.name/how-to-reduce-memory-usage-in-pandas
    
    '''
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype
        
        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.uint8).min and c_max < np.iinfo(np.uint8).max:
                    df[col] = df[col].astype(np.uint8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.uint16).min and c_max < np.iinfo(np.uint16).max:
                    df[col] = df[col].astype(np.uint16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.uint32).min and c_max < np.iinfo(np.uint32).max:
                    df[col] = df[col].astype(np.uint32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
                elif c_min > np.iinfo(np.uint64).min and c_max < np.iinfo(np.uint64).max:
                    df[col] = df[col].astype(np.uint64)
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))
    return df


def extract_date_feature(date):
    date = datetime.strptime(date, "%Y-%m-%d")
    try:
        date = pd.to_datetime(date)
    except Exception as e:
        raise e

    # Extract features from the date
    month = date.month
    weekday = date.weekday()

    return month, weekday
    
    

def extract_item_feature(item_id):
    category_encoding = {
        'FOODS': 0,
        'HOBBIES': 1,
        'HOUSEHOLDS':2
    }
    category, department, item = item_id.split('_')

    return int(department), category_encoding[category], int(item) 

def extract_store_feature(store_id):
    state_encoding = {
        'CA' : 0,
        'TX' : 1,
        'WI' : 2
    }
    state, store_num = store_id.split("_")
    return state_encoding[state], int(store_num)

def extract_features(date, store_id, item_id):
    month, weekday = extract_date_feature(date)
    state, store_num = extract_store_feature(store_id)
    department, category, item = extract_item_feature(item_id)
    return np.array([[month, weekday, state, store_num, category, department, item]])