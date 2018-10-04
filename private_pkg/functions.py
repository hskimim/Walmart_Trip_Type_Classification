
def getNullDataInfo(df):
    df = df.isnull().sum().reset_index()
    df.columns = ["Feature_name", "Sum of null data"]
    return df

def __getNullDataDistributionByColumn(df, col, see_list_detail):
    if col == "VisitNumber":
        return
    try:
        unique_li = sorted(df[col].unique())
    except:
        unique_li = df[col].unique()
    data_name_li = []
    len_li = []
    for unique in list(unique_li):
        data_name_li.append(unique)
        if type(unique) is not str:
            unique = float(unique)
        if type(unique) is float:
            if np.isnan(unique):
                len_li.append(len(df[df[col].isnull()]))
            else:
                len_li.append(len(df[df[col] == unique]))                
        else:        
            len_li.append(len(df[df[col] == unique]))
    if see_list_detail:
        print(dict(zip(data_name_li, len_li)))
    if len(unique_li) > 1:
        data_name_li = [ 'Null' if type(d) is float and np.isnan(d) else d for d in data_name_li]
        plt.figure(figsize=(10, 5))
        sns.barplot(x = data_name_li, y = len_li)
        plt.xticks(rotation = 90)
        plt.show()
        if col == "TripType":
            print("DepartmentDescription이 Null인 경우의 분포")
            __drawDictionaryOfTripTypeOnBarplot(df, False, see_list_detail)
            print("DepartmentDescription이 PHARMACY RX인 경우의 분포")
            __drawDictionaryOfTripTypeOnBarplot(df, True, see_list_detail)
            
def __drawDictionaryOfTripTypeOnBarplot(df, isNull, see_list_detail):
    df_ = df.copy()
    unique_li = sorted(df_["TripType"].unique())
    if isNull:
        df_ = df_[df_["DepartmentDescription"] == "PHARMACY RX"]
    else:
        df_ = df_[df_["DepartmentDescription"].isnull()]
    data_name_li = []
    len_li = []
    for unique in list(unique_li):
        data_name_li.append(unique)
        len_li.append(len(df_[df_["TripType"] == unique]))
    if see_list_detail:
        print("분포")
        print(dict(zip(data_name_li, len_li)))
    plt.figure(figsize=(10, 5))
    sns.barplot(x = data_name_li, y = len_li)
    plt.xticks(rotation = 90)
    plt.show()
            
def getNullDataDetailInfo(df, specify_col = [], see_list_detail = False):
    df_ = df[df["Upc"].isnull()]
    if specify_col == []:
        cols = list(df_.columns)
    else:
        cols = specify_col
    for col in cols:
        unique = df_[col].unique()
        print()
        print(str(col) + " 컬럼의 유니크 데이터 정보(" + str(len(unique)) + "종류)")
        if see_list_detail:
            print(unique)
        print()
        print("데이터별 분포")
        __getNullDataDistributionByColumn(df_, col, see_list_detail)        
        print()
        if len(cols) > 1:
            print("================================================================================")