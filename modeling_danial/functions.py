
import numpy as np
import pandas as pd
import seaborn as sns
import requests # to send slack msg
import json # to send slack msg
import matplotlib.pylab as plt
from datetime import datetime
from IPython.display import display, Markdown
from sklearn.metrics import confusion_matrix

"""
    함수종류
    getNullDataInfo : 컬럼별로 np.nan의 개수를 알고싶을 때 사용. 패러미터로 df를 넣어준다.
    getNullDataDetailInfo : 컬럼별로 좀 더 디테일한 Null데이터 관련 정보를 보고싶을 때 사용.
    displayDataByVisitNumber : 지정해준 VisitNumber에 대한 Row만을 Display한다. (반환하지 않는다!)
    makeDfWeWanted : 컬럼은 Dd나 Fl을 지정해줄 경우 그 컬럼의 카테고리수에 해당하는 컬럼을 가진 Df을 반환한다.(말이 어렵다. 간단하게는 우리가 원하는 Df를 반환)
    makeSubmissionDf : Y_predicted 값과 Submission df으로 Kaggle 제출용 데이터프레임을 만들어 반환한다.
    getColsToMakeFeatureMatrix : X_feature로 만들기위해 TripType과 VisitNumber 컬럼들을 제외하기 위한 Mask용 List를 반환한다.
    compareClassificationReport : classification_report를 두개넣어주면 비교하기 쉽도록 만든 DF를 반환한다.
    getAccuracy : y_true, y_pred, datalength를 넣어주면 Accuracy를 Display한다.
"""

def getNullDataInfo(df):
    """
        패러미터 종류 : df
        df : 컬럼별로 np.nan데이터의 총 개수를 DataFrame형식으로 반환한다.
    """
    df = df.isnull().sum().reset_index()
    df.columns = ["Feature_name", "Sum of null data"]
    return df

def getNullDataDetailInfo(df, specify_col = [], see_list_detail = False, visit_number_threshold = 2):
    """
        패러미터 종류 : df, specify_col, see_list_detail
        df : 확인하고자하는 데이터프레임을 넣어준다.
        specify_col : 보고싶은 컬럼들의 이름을 리스트형식으로 넣어준다. default는 []이며, 이 경우에는 모든 컬럼에대한 정보를 다 보여준다. 
        see_list_detail : 분포를 리스트형식 파일로도 보고싶은 경우에 True를 넣어준다. default는 False
        visit_number_threshold : 0보다 큰 정수를 넣어줘야한다. default는 2
    """
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
        __getNullDataDistributionByColumn(df_, col, see_list_detail, visit_number_threshold, df)        
        print()
        if len(cols) > 1:
            print("================================================================================")
            
def displayDataByVisitNumber(df, vn):
    """
        패러미터 정보 :
        확인하고싶은 df와 vn : VisitNumber를 넣어준다(int형식으로).
    """
    display(df[df["VisitNumber"] == vn])

def makeDfWeWanted(df, df_train, df_test, group_by_col_name = "DepartmentDescription", is_use_negative_scan_count = False, is_model_for_multinomial = False):
    """
        패러미터 정보 : 
        df에는 변환하고자하는 df를 넣어준다.
        df_train에는 train데이터 프레임을 반드시 넣어준다.
        df_test에는 test데이터 프레임을 반드시 넣어준다.(FinelineNumber)때문.
        is_use_negative_scan_count는 False가 default이다.
    """
    cols = df.columns
    df_ = df.copy()
    dd_cols = []
    if group_by_col_name == "DepartmentDescription":
        df_[group_by_col_name] = df_[group_by_col_name].apply(lambda a : "Null" if type(a) is float else a)
        dd_cols = [uq if not type(uq) is float else "Null" for uq in df_train[group_by_col_name].unique()]
    else:
        df_[group_by_col_name] = df_[group_by_col_name].apply(lambda a : -1.0 if np.isnan(a) else a)
        train_fl_li = [fl if not np.isnan(fl) else -1  for fl in df_train[group_by_col_name].unique()]
        test_fl_li = [fl if not np.isnan(fl) else -1  for fl in df_test[group_by_col_name].unique()]
        dd_cols = [li if not np.isnan(li) else -1.0 for li in list(set(list(test_fl_li) + list(train_fl_li)))]
    dd_cols.insert(0, "VisitNumber")
    dd_cols.append("TripType")
    result_df = pd.DataFrame(columns = dd_cols)
    dict_ = dict(zip(dd_cols, np.zeros(len(dd_cols)).astype(int)))
    is_test_df = False
    if "TripType" in cols:
        df_ = pd.DataFrame(df_.groupby(["VisitNumber", "TripType", group_by_col_name]).sum()["ScanCount"]).reset_index()
    else:
        is_test_df = True
        dd_cols = dd_cols[:-1]
        df_ = pd.DataFrame(df_.groupby(["VisitNumber", group_by_col_name]).sum()["ScanCount"]).reset_index()
    return __makeDf(df_, is_use_negative_scan_count, dd_cols, is_test_df, group_by_col_name, is_model_for_multinomial)

def makeSubmissionDf(df, y_pred):
    """
        Submission용 df를 만드는 함수
        df : walmart에서 제공한 submission df를 넣어준다.
        y_pred : df의 visit_number와 같은 순서로 나열된 predict한 y값으로 이루어진 List를 넣어준다.
        간단하게 말하자면 그냥 model.predict(x)해서 나온 y 리스트 넣어주면 된다.
    """
#     for i, df_row in enumerate(df.index):
#         col_name = "TripType_" + str(y_pred[i])
#         df.loc[df_row][col_name] = 1
#         if (i % 5000) == 0:    
#             print(str(i) + "명 진행됨. 아직 " + str(len(df) - i) + "명 데이터 남음.")

    y_pred_proba_xgb_df = pd.DataFrame(y_pred, columns = df.columns[1:])
    result_df = pd.concat([df["VisitNumber"], y_pred_proba_xgb_df], axis=1)
    
    return result_df

def getColsToMakeFeatureMatrix(df_test, df_train, is_fl = False):
    """
       df_test : test dataframe
       df_train : train dataframe
       is_fl : FinelineNumber가 들어간 컬럼명을 원하는지 알려준다. default는 False로 DepartmentDescription으로 이뤄진 컬럼리스트를 반환한다.
    """
    df_ = df_train.copy()
    dd_cols = []
    if not is_fl:
        group_by_col_name = "DepartmentDescription"
        df_[group_by_col_name] = df_[group_by_col_name].apply(lambda a : "Null" if type(a) is float else a)
        dd_cols = [uq if not type(uq) is float else "Null" for uq in df_train[group_by_col_name].unique()]
    else:
        group_by_col_name = "FinelineNumber"
        df_[group_by_col_name] = df_[group_by_col_name].apply(lambda a : -1.0 if np.isnan(a) else a)
        train_fl_li = [fl if not np.isnan(fl) else -1  for fl in df_train[group_by_col_name].unique()]
        test_fl_li = [fl if not np.isnan(fl) else -1  for fl in df_test[group_by_col_name].unique()]
        dd_cols = [str(li) if not np.isnan(li) else "-1.0" for li in list(set(list(test_fl_li) + list(train_fl_li)))]
    return dd_cols

def compareClassificationReport(report1, report2):
    """
        classification_report 두개를 비교 분석하기 용이하게 DF을 만들어서 반환한다.
        report1 : classification_report
        report2 : classification_report
        set_trip_type_as_index : triptype을 인덱스로한 df를 원하는지 넣어준다. default는 True
    """
    report1_df, cols = __preproccessToMakeDf(report1)
    report2_df, cols = __preproccessToMakeDf(report2)
    li = np.zeros(39 * 2 * 6).astype(str).reshape(39 * 2, 6)
    cols.append("model")
    for idx in range(len(li)):
        if idx % 2 == 0:
            tmp = list(report_fl_np[idx//2])
            tmp.append("fl")
            li[idx] = np.array(tmp)
        else:
            tmp = list(report_dd_np[idx//2])
            tmp.append("dd")
            li[idx] = np.array(tmp)
    df_report = pd.DataFrame(li, columns=cols)
    for tt in df_report["TripType"].unique():
        display(df_report[df_report["TripType"] == tt])
    return df_report

def getAccuracy(y_true, y_pred, data_length):
    """
        y_true : 원래 타겟 컬럼의 데이터를 넣어준다.
        y_pred : 예측한 값을 넣어준다.
        data_length : 예측한 데이터의 총 개수를 넣어준다.
    """
    display(Markdown("##### Accuracy : " + str(round(np.trace(confusion_matrix(y_true, y_pred))/data_length, 4))))

def __preproccessToMakeDf(report):
    result = [li for li in report.strip().split(" ") if li != ""]
    cols = [rlt.split("\n")[0] for rlt in result[:4]]
    cols.insert(0, "TripType")
    data = [rlt.split("\n")[0] for rlt in result[4:] if rlt != "/"]
    data = np.array(data).reshape(39, 5)
    return data, cols

def __makeDf(df, is_use_negative_scan_count, dd_cols, is_test_df, group_by_col_name, is_model_for_multinomial):
    display(Markdown("##### makeDfWeWanted함수는 !! 데이터프레임을 반환하는 \
    함수이므로 변수로 받아주셔야합니다!! 오래 걸리는 작업(맥북 프로 기준 3분)이므로 지금 실수하셨다면 빨리 커널 종료하고 다시 시도해주세요."))
    vn_uq_li = df["VisitNumber"].unique()
    li = []
    scan_count_min = 0
    if is_use_negative_scan_count:
        scan_count_min = -df["ScanCount"].min()
    for i, vn in enumerate(vn_uq_li):
        tmp_df = df[df["VisitNumber"] == vn]
        space = []
        if is_test_df:
            space = (np.zeros(len(dd_cols)).astype(int) + scan_count_min) if not is_test_df else (np.zeros(len(dd_cols)).astype(int) + scan_count_min)
        else:
            space = (np.zeros(len(dd_cols)).astype(int) + scan_count_min) if not is_test_df else (np.zeros(len(dd_cols) - 1).astype(int) + scan_count_min)
        space[0] = vn
        if not is_test_df:
            tripType = tmp_df["TripType"].unique()[0]
            space[len(space) - 1] = tripType
        for row_nbr in tmp_df.index:
            dd = tmp_df.loc[row_nbr][group_by_col_name]
            scan_cnt = tmp_df.loc[row_nbr]["ScanCount"] + scan_count_min
            idx = dd_cols.index(dd)
            if is_use_negative_scan_count:
                space[idx] = scan_cnt
            else:
                if is_model_for_multinomial:
                    if scan_cnt > 0:
                        space[idx] = scan_cnt
                else:
                    space[idx] = scan_cnt
        li.append(space)
        if (i % 5000) == 0:    
            print(str(i) + "명 진행됨. 아직 " + str(len(vn_uq_li) - i) + "명 데이터 남음.")
    return pd.DataFrame(li, columns=dd_cols) 
    
def __getNullDataDistributionByColumn(df, col, see_list_detail, visit_number_threshold, original_df):
    try:
        unique_li = sorted(df[col].unique())
    except:
        unique_li = df[col].unique()
    data_name_li = []
    len_li = []
    if col == "VisitNumber":
        data_name_li, len_li = __getMoreAnalyzedVisitNumber(df, unique_li, col, visit_number_threshold, original_df)
    else:
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
    else:
        if len(unique_li) == 1:
            print(dict(zip(data_name_li, len_li)))
    if len(unique_li) > 1 and col != "VisitNumber":
        data_name_li = [ 'Null' if type(d) is float and np.isnan(d) else d for d in data_name_li]
        plt.figure(figsize=(10, 5))
        sns.barplot(x = data_name_li, y = len_li)
        plt.xticks(rotation = 90)
        plt.show()

def __getMoreAnalyzedVisitNumber(df, unique_li, col, visit_number_threshold, original_df):
    df_null_mask = []
    df_pharmacy_mask = []
    data_name_li = []
    len_li = []
    for unique in list(unique_li):
        length = len(df[df[col] == unique])
        if length >= visit_number_threshold:            
            data_name_li.append(unique)
            len_li.append(length)
    
    display(Markdown("## 여기서 세가지 모두 Null인 아이템이란 Upc, DepartmentDescription, FinelineNumber가 다 NaN인 경우를 말한다.\
    또한 두가지가 Null인 아이템이란 Upc, FinelineNumber가 NaN이면서 DepartmentDescription은 PHARMACY RX인 경우를 말한다."))
    
    df_dd_null = df[df["DepartmentDescription"].isnull()]
    df_null_mask = df_dd_null["VisitNumber"].isin(data_name_li)
    df_dd_null_visit_number_above_threshold = df_dd_null[df_null_mask]
    ppl_nbr = len(df_dd_null_visit_number_above_threshold["VisitNumber"].unique())
    print("세가지 컬럼이 모두 Null인 " + str(visit_number_threshold) + "가지 이상 종류의 아이템을 산 사람은 총 " + str(ppl_nbr) + "명이다.")
    __drawTripTypeBarPlot(df_dd_null_visit_number_above_threshold, original_df, df, False, visit_number_threshold)
    
    df_dd_pharmacy = df[df["DepartmentDescription"] == "PHARMACY RX"]
    df_pharmacy_mask = df_dd_pharmacy["VisitNumber"].isin(data_name_li)
    df_dd_pharmacy_visit_number_above_threshold = df_dd_pharmacy[df_pharmacy_mask]
    ppl_nbr = len(df_dd_pharmacy_visit_number_above_threshold["VisitNumber"].unique())
    print("두가지 컬럼이 모두 Null인 " + str(visit_number_threshold) + "가지 이상 종류의 아이템을 산 사람은 총 " + str(ppl_nbr) + "명이다.")
    __drawTripTypeBarPlot(df_dd_pharmacy_visit_number_above_threshold, original_df, df, True, visit_number_threshold)
    
    print("위 두가지 경우의 합이 VisitNumber컬럼의 유니크한 데이터 총 개수와 다른 이유는 23757, 50745")
    print("VisitNumber에 해당하는 두 명이 세가지 다 Null인 아이템과 Pharmacy RX 제품 두개다 구매했기 때문이다.")
    display(df[df["VisitNumber"] == 23757])
    display(df[df["VisitNumber"] == 50745])
    
    __getDetailInfoAboutVisitNumberBoughtNotNullProduct(data_name_li, original_df, visit_number_threshold)
    
    return data_name_li, len_li

def __drawTripTypeBarPlot(df, original_df, original_df_with_only_null, isNull, visit_number_threshold):
    col = "TripType"
    unique_li = sorted(original_df[col].unique())
    data_name_li = []
    len_li = []
    for unique in list(unique_li):
        length = len(df[df[col] == unique])         
        data_name_li.append(unique)
        len_li.append(length)
    plt.figure(figsize=(10, 5))
    sns.barplot(x = data_name_li, y = len_li)
    plt.xticks(rotation = 90)
    plt.show()
    if isNull:
        percentage = str(round(np.array(len_li).max()/sum(len_li), 2) * 100) + "%"
        display(Markdown('## **Pharmacy rx인 경우에는 {}가 TripType 5에 해당한다.**'.format(percentage)))
        display(Markdown('## **두가지 컬럼이 Null인 {}가지 이상 종류의 아이템을 산사람들의 분포이다.**'.format(visit_number_threshold)))        
    else:
        percentage = str(round(np.array(len_li).max()/sum(len_li), 2) * 100) + "%"
        display(Markdown('## **Upc, DepartmentDescription, FinelineNumber 모두 Null인 경우에는 999 type이 {}를 차지한다.**'.format(percentage)))        
        display(Markdown('## **세가지 컬럼이 모두 Null인 {}가지 이상 종류의 아이템을 산사람들의 분포이다.**'.format(visit_number_threshold)))        
        
def __getDetailInfoAboutVisitNumberBoughtNotNullProduct(data_name_li, original_df_, visit_number_threshold):
    col = "VisitNumber"
    original_df = original_df_.copy()
    cut_off_df = original_df_[original_df_["Upc"].isnull()]
    df_null_mask = original_df["VisitNumber"].isin(data_name_li)
    null_df = original_df[df_null_mask]
    null_li = null_df[col].unique()
    uq_li = []
    uq_li_who_bought_only_null_products = []
    for li in null_li:
        if len(null_df[null_df[col] == li]["Upc"].unique()) > 1:
            uq_li.append(li)
        else:
            uq_li_who_bought_only_null_products.append(li)
            
    null_df = original_df_[original_df_[col].isin(uq_li)]    
    cut_off_null_df = cut_off_df[cut_off_df[col].isin(uq_li)]

    display(Markdown("#### Null이 포함된 " + str(visit_number_threshold) + "가지 이상의 아이템을 산 사람들의 null 아이템 정보(df.tail)"))
    display(cut_off_null_df.tail())     
    display(Markdown("#### Null이 포함된 " + str(visit_number_threshold) + "가지 이상의 아이템을 산 사람들의 모든 아이템 정보(df.tail)"))
    display(null_df.tail())     
#     display(null_df.groupby("VisitNumber"))
    
    display(Markdown("## 총 " + __setColoredText(len(null_li)) + "명의 사람들 중에 " + __setColoredText(len(uq_li)) + "(" + __setColoredText(round(len(uq_li)/len(null_li), 2) * 100) + "%)명에 해당하는 사람들은 Null 데이터가 아닌 아이템도 샀다."))
    display(Markdown("## 즉, " + __setColoredText(len(null_li) - len(uq_li)) + "명의 사람만이 Null 데이터만 가진 아이템을 샀다."))
    
#     display(len(uq_li_who_bought_only_null_products), uq_li_who_bought_only_null_products[:5])
    df_only_bought_null_products = cut_off_df[cut_off_df["VisitNumber"].isin(uq_li_who_bought_only_null_products)]
    
    df_only_bought_null_pharmacy_products = df_only_bought_null_products[df_only_bought_null_products["DepartmentDescription"] == "PHARMACY RX"]
    df_only_bought_null_not_pharmacy_products = df_only_bought_null_products[df_only_bought_null_products["DepartmentDescription"] != "PHARMACY RX"]
    
    display(df_only_bought_null_pharmacy_products.tail())
    unique_nbr = len(df_only_bought_null_pharmacy_products["VisitNumber"].unique())
    display(Markdown("## 그 " +  str(len(null_li) - len(uq_li)) + "명 중에 " + str(unique_nbr) + " 명이 Pharmacy 제품만을 산사람이다."))
    
    display(df_only_bought_null_not_pharmacy_products.tail())
    uq_li_999 = df_only_bought_null_not_pharmacy_products["VisitNumber"].unique()
    unique_nbr_ = len(uq_li_999)
    display(Markdown("## 그 " +  str(len(null_li) - len(uq_li)) + "명 중에 " + str(unique_nbr_) + " 명이 Null 제품만을 산사람이다."))
#     display(df_only_bought_null_not_pharmacy_products["TripType"].unique())
    display(Markdown("## 이 사람들의 TripType은 모두 "+ __setColoredText(999) + "이다."))
    
    special_vn_li = []
    for nbr in uq_li_999:
        sumOfScanCount = df_only_bought_null_not_pharmacy_products[df_only_bought_null_not_pharmacy_products["VisitNumber"] == nbr]["ScanCount"].sum()
        if sumOfScanCount > 0:
            special_vn_li.append(nbr)
    display(Markdown("## 그 중에서 VisitNumber별로 ScanCount의 Sum이 0이상인 경우(실제로 NaN아이템을 산사람)의 VisitNumber list다."))            
    print(special_vn_li)
    
def __setColoredText(txt, color_rgb = "red"):
    return "<span style='color:" + color_rgb +"'>" + str(txt) + "</span>"

def saveDataFrameToCsv(df, fileName, idx = False):
    """
        넘겨준 df를 filename + 년월일시간분 의 format으로 이루어진 이름의 파일로 생성해준다.
        index를 True로 넘겨주면 저장할 때 아규먼트로 index=True를 넣어주게 된다.
    """
    fileName += "_" + datetime.now().strftime("%Y%m%d%H%M") + ".csv"
    return df.to_csv(fileName, index = idx)

def sendSlackDm(url, text):
    """
        Parameter :
            각자 받은 url을 넣어준다.
            text에는 보낼 글 내용
    """
    webhook_url = url
    slack_data = {'text': text}
    response = requests.post(
        webhook_url,
        data=json.dumps(slack_data),
        headers={'Content-Type': 'application/json'}
    )
    if response.status_code != 200:
        raise ValueError(
            'Request to slack returned an error %s, the response is:\n%s'%(response.status_code, response.text)
    )