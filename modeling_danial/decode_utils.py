
import numpy as np
import pandas as pd

def checksum(x):
    try:
        odd = map(int, ','.join(x[-1::-2]).split(','))
        even = map(int, ','.join(x[-2::-2]).split(','))
        sum_odd3 = sum(odd) * 3
        total = sum_odd3 + sum(even)
        rem = total % 10
        if rem == 0:
            return rem
        return 10 - rem
    except:
        return 0

def full_upc(x):
    try:
        if len(x) < 12:
            missing = 11 - len(x)
            zeros = ['0'] * missing
            xx = zeros + ','.join(x).split(',') + [str(checksum(x))]
            xx = ''.join(xx)
            return xx
    except:
        return "-9999"

def company(x):
    try:
        p = x[:6]
        if p == '000000':
            if x == "000000-99990":
                return "000000"
            return x[7:-3]
        return p
    except:
        return "-9999"
    
def item(x):
    try:
        p = x[:6]
        if p == '000000':
            if x == "000000-99990":
                return "000000"
            return x[10:-1]
        return x[7:-1]
    except:
        return "-9999"
    
def decodeStuffNeedsToBeDecoded(df_):
    df = df_.copy()
    col_list = ["FinelineNumber", "DepartmentDescription", "Upc"]
    replace_nan_list = ["1.1", "NULL", "-9999"]
    tmp = df[df["Upc"].isnull()]
    vn_li = tmp[~tmp["DepartmentDescription"].isnull()].VisitNumber.unique()
    
    for col in col_list:
        col_idx = col_list.index(col)
        df[col] = df[col].astype(str)
        df[col] = df[col].apply(lambda a : a.split(".")[0] \
                                if a != "nan" else replace_nan_list[col_idx])
        if col_idx == 2:
            df[col] = df[col].apply(full_upc)
            df["Company"] = df[col].apply(company)
            df["Item_nbr"] = df[col].apply(item)
    
#     301691837020, 4822
    df.set_index("VisitNumber", inplace=True)
    df.at[vn_li, "Upc"] = "301691837020"
    df.at[vn_li, "FinelineNumber"] = "4822"
    df.at[vn_li, "Company"] = "301691"
    df.at[vn_li, "Item_nbr"] = "3702"
    df = df.reset_index()

    return df

def getMostFrequentFeatureAsDf(df_, col):
    """
        df_ : df_train or df_test
        col : Most frequent item per visitnumber를 구하고자하는 컬럼명을 넣어준다. 
    """
    df = df_.copy()
    tmp_df = df.groupby(["VisitNumber", col]).agg({"ScanCount" : np.sum}).reset_index()
    mf_feature_li = []
    scancount_li = []
    for i, vn in enumerate(tmp_df.VisitNumber.unique()):
        tmp = tmp_df[tmp_df.VisitNumber == vn]
        max_count = tmp["ScanCount"].max()
        mf_feature = tmp[tmp["ScanCount"] == max_count][col].values[0]
        mf_feature_li.append(mf_feature)
        scancount_li.append(max_count)
        if i % 5000 == 0:
            print(str(i) + "까지 진행됨. " + str(95674 - i) + "개 남음.")
    key_value = "MF_" + col
    result_df = pd.DataFrame({key_value : mf_feature_li, "ScanCount" : scancount_li})
    return result_df

def concatDf(df_1, df_2):
    dd_cols = df_1.columns
    if ("MENS WEAR" in dd_cols) and ("MENSWEAR" in dd_cols):
        df_1["MENS WEAR"] = df_1["MENSWEAR"] + df_1["MENS WEAR"]
        df_1.drop("MENSWEAR", axis=1, inplace = True)
    return pd.concat([df_1, df_2], axis = 1)

def concatMenswear(df):
    df["MENS WEAR"] = df["MENSWEAR"] + df["MENS WEAR"]
    return df.drop("MENSWEAR", axis=1)

def getSpecifiedVisitNumberData(df_train, vn):
    display(df_train[df_train.VisitNumber == vn])
#     display(df_train_dd[df_train_dd.VisitNumber == vn])
    
def getDummiesDf(df):
    dummie_col = df.columns[0]
    ItemNumber = df["ScanCount"]
    dummies_desc = pd.get_dummies(df[dummie_col])
    desc_cols = dummies_desc.columns
    result_df = dummies_desc.apply(lambda x: x * ItemNumber)
    result_df = pd.DataFrame(np.where(result_df < 0, 0, result_df), columns=result_df.columns)
    return result_df