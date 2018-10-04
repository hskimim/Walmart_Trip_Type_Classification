
import pandas as pd
import numpy as np

def get_pivor_df(df, col):
    return pd.pivot_table(data= df, index="VisitNumber", fill_value=0,\
                          values="ScanCount", columns=col, aggfunc=np.sum)

def get_num_of_categories_per_users(df_, col = "DepartmentDescription"):
    """
        예:) vn_dd_more_than_one, vn_dd_one = get_flat_type_user(df_decoded) 반드시 df는 decoding 함수로 decoded된 것을 넣을 것.
    """
    df = df_.copy()
    df_count = df.groupby(["VisitNumber", col]).sum()["ScanCount"].reset_index(name="Sc_sum")
    df_count["Sc_sum"] = np.where(df_count["Sc_sum"] < 0, 0, df_count["Sc_sum"])
    df_count = df_count.dropna()
    df_count_total = df_count.groupby("VisitNumber").sum()["Sc_sum"].reset_index(name="Total")

    df_merged = pd.merge(df_count, df_count_total, on="VisitNumber")
    
    zero_counts_vn_li = df_merged[df_merged["Total"] == 0].VisitNumber.unique()
    df_merged = df_merged.groupby("VisitNumber").count()[col].reset_index(name="Count")
    df_merged[df_merged.VisitNumber.isin(zero_counts_vn_li)] = 0
    
    return df_merged["Count"].values

def get_count_from_col(df_decoded, col):
    tmp = df_decoded.groupby([col, "DepartmentDescription"]).sum()["ScanCount"].reset_index()
    tmp = tmp.groupby(col).size().reset_index(name="Count")
    if col == "Item_nbr":
        li = tmp[tmp.Count > 1][col].unique()
        return li
    li = tmp[tmp.Count == 1][col].unique()
    return li

def getRelevantListWeNeed(df_decoded):
    tmp = df_decoded.groupby(["VisitNumber", "DepartmentDescription"]).agg({"ScanCount" : np.sum}).reset_index()
    tmp = tmp.groupby("VisitNumber").sum()["ScanCount"].reset_index(name='ScanCount')
    sc_li = tmp.ScanCount.values # 95674
    cp_li = get_count_from_col(df_decoded, "Company") # 4806
    fl_li = get_count_from_col(df_decoded, "FinelineNumber") # 2695
    fl_li = fl_li.astype(float)
    fl_li = fl_li.astype(str)
    item_li = get_count_from_col(df_decoded, "Item_nbr")

    return sc_li, cp_li, fl_li, item_li

def dummies_regularize_by_visit_number(df, df_col, cols=[], col = "DD"):
    cols_do_not_need = ['VisitNumber', 'Return', 'TripType', 'Monday', 'Tuesday', 'Wednesday', 'Thursday',
       'Friday', 'Saturday', 'Sunday', 'MENS WEAR']
    if col == "DD":
        cols = [col for col in cols if col not in cols_do_not_need]
        
    return df[cols].div(df_col, axis = 0)

def get_filttered_list_by_cols(dfs):
    col_fl_filttered_li = []
    col_cp_filttered_li = []
    col_item_filttered_li = []
    cols = [col_fl_filttered_li, col_cp_filttered_li, col_item_filttered_li]
    threshold = [14, 11.01, 73]
    for index, df in enumerate(dfs):
        for idx, col in enumerate(df.columns):
            sum_of_col = df[col].sum()
            if sum_of_col >= threshold[index]:
                cols[index].append(col)
    return cols