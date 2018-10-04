import pandas as pd
import numpy as np
from IPython.display import display, Markdown

def make_df_we_wanted(df, dummie_col = "DepartmentDescription", is_use_positive_scancount_only = True, is_test_df = False):
    result = df.copy()
    
    # 요일을 숫자로 변경
    result["Weekday"] = __change_weekday_to_number(result)

    # 반환여부를 1(반환한 경우), 0로 표현
    result["Return"] = result["ScanCount"].apply(lambda a: 1 if a < 0 else 0)
    
    # Scan
    if is_use_positive_scancount_only:
        result["ScanCount"] = result["ScanCount"].apply(lambda a : a if a >=0 else 0)
    
    # 원하는 컬럼(dummie_col에 넣어준 컬럼명)을 Dummie로 변경
    result = __make_dummy_columns(result, dummie_col)
    
    # VisitNumber를 이용해서 groupby하여 Row수를 VisitNumber Unique한 숫자만큼 축소
    result = __make_df_groupby_visit_number(result, is_test_df)
    
    return __make_weekday_as_dummies(result)

def __change_weekday_to_number(df):
    weekday_dict = {
        "Monday" : 1,
        "Tuesday" : 2,
        "Wednesday" : 3,
        "Thursday" : 4,
        "Friday" : 5,
        "Saturday" : 6,
        "Sunday" : 7
    }
    return df["Weekday"].map(weekday_dict)

def __make_dummy_columns(df, dummie_col):
    ItemNumber = df["ScanCount"]
    dummies_desc = pd.get_dummies(df[dummie_col])
    desc_cols = dummies_desc.columns
    df[desc_cols] = dummies_desc.apply(lambda x: x * ItemNumber)
    should_remove_cols = ["Upc", "DepartmentDescription", "FinelineNumber", "ScanCount"]
    new_cols = [col for col in df.columns if col not in should_remove_cols]
    return df[new_cols]

def __make_df_groupby_visit_number(df, is_test_df):
    cols = [col for col in list(df.columns) if col != "VisitNumber"]
    np_max_cols = cols[:2] if is_test_df else cols[:3]
    values = [np.max if col in np_max_cols else np.sum for col in cols]
    dict_ = dict(zip(cols, values))
    if is_test_df:
        result_df = df.groupby(by='VisitNumber').agg(dict_).reset_index()
        result_df["HEALTH AND BEAUTY AIDS"] = np.zeros(len(result_df))
        return result_df
    return df.groupby(by='VisitNumber').agg(dict_).reset_index()

def __make_weekday_as_dummies(df):
    dummies_desc = pd.get_dummies(df["Weekday"])
    desc_cols = dummies_desc.columns
    df[desc_cols] = dummies_desc
    return df.drop("Weekday", axis = 1)






