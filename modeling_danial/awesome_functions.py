
import pandas as pd
import numpy as np
import requests # to send slack msg
import json # to send slack msg
from datetime import datetime
from IPython.display import display, Markdown
from sklearn.metrics import confusion_matrix
from sklearn.externals import joblib

"""
    make_submission_df : Walmart에서 제공한 Submission df 원본과 predict_proba 리스트를 넣어주면 Submission할 수 있는 형태의 Df를 반환한다.
    get_df_to_fit : make_df_we_wanted로 만든 매트릭스를 바로 머신러닝 모델에 Fit할 수 있는 상태로 만들어 반환한다.
    make_df_we_wanted : 우리가 원하는 모형의 Df를 반환한다. (DD, FL)두가지 컬럼만 지원한다.
    compareClassificationReport : 독립된 두개의 Classification_report를 패러미터로 넣어주면 각 TripType별로 확인할 수 있는 Df를 반환한다.
    getAccuracy : y_true, y_pred, length of data를 넣어주면 Accuracy를 반환한다.
    saveDataFrameToCsv : 데이터 프레임을 저장해준다. 
    sendSlackDm : 슬랙 url을 이용해서 메시지를 발송한다.
    saveModelObjectAsPickle : 모델 객체를 저장하는 함수.
"""

def make_submission_df(df, y_pred):
    """
        Submission용 df를 만드는 함수
        df : walmart에서 제공한 submission df를 넣어준다.
        y_pred : df의 visit_number와 같은 순서로 나열된 predict한 y값으로 이루어진 List를 넣어준다.
        간단하게 말하자면 그냥 model.predict(x)해서 나온 y 리스트 넣어주면 된다.
    """
    y_pred_proba_xgb_df = pd.DataFrame(y_pred, columns = df.columns[1:])
    result_df = pd.concat([df["VisitNumber"], y_pred_proba_xgb_df], axis=1)
    
    return result_df

def get_df_to_fit(df, is_test_df = False):
    
    # test 모델은 TripType이 없으므로, X만 반환한다.
    if is_test_df:
        return df.drop(["VisitNumber"], axis = 1)
    
    # model에 fit할 때 사용할 X, y를 반환한다. (X, y로 받아줘야 한다.)
    return df.drop(["TripType", "VisitNumber"], axis = 1), df["TripType"]

def make_df_we_wanted(df, df_train, df_test, dummie_col = "DepartmentDescription", 
                      is_use_positive_scancount_only = True, is_test_df = False, is_need_null_column = False):
    """
        df : 전처리 목표 dataframe
        df_train : walmart에서 제공한 train dataframe을 넣어준다. (FinelineNumber를 이용해서 만들 때 필요하다.)
        df_test : walmart에서 제공한 test dataframe을 넣어준다. (FinelineNumber를 이용해서 만들 때 필요하다.)
        dummie_col : default는 DepartmentDescription이지만, FinelineNumber가 필요할 때는 문자열로 넣어주면된다.
        is_use_positive_scancount_only : False를 할 경우엔 음수도 같이 누적된다.
        is_test_df : default는 False. model.predict에 넣을 test df를 만들 때 True로 넣어준다.
        is_need_null_column : default는 False. True를 넣어주면 NaN인 아이템을 산 데이터 정보는 Null이라는 컬럼을 만들어 넣어준다.
    """
    
    # 넣어준 패러미터들 정보에대해 Display한다.
    __display_parameter_detail(dummie_col, is_use_positive_scancount_only, is_test_df, is_need_null_column)
    
    result = df.copy()
    
    # Null데이터 개수도 필요한 경우에 사용하면 컬럼으로 dd인경우 Null이 추가되고 fl인 경우엔 -1컬럼이 추가된다.
    if is_need_null_column: 
        if dummie_col == "DepartmentDescription":
            result[dummie_col] = result[dummie_col].apply(lambda a : "Null" if type(a) is float else a)
        else:
            result[dummie_col] = result[dummie_col].apply(lambda a : -1.0 if np.isnan(a) else a)
    
    # 요일을 숫자로 변경
    result["Weekday"] = __change_weekday_to_number(result)

    # 반환여부를 1(반환한 경우), 0로 표현
    result["Return"] = result["ScanCount"].apply(lambda a: 1 if a < 0 else 0)
    
    if dummie_col == "DepartmentDescription":
        # 원하는 컬럼(dummie_col에 넣어준 컬럼명)을 Dummie로 변경
        result = __make_dummy_columns(result, dummie_col)
    else:
        return __make_fl_df_using_for_sentence(result, df_train, df_test, dummie_col,\
                                               is_use_positive_scancount_only, is_test_df, is_need_null_column)
    
    # VisitNumber를 이용해서 groupby하여 Row수를 VisitNumber Unique한 숫자만큼 축소
    result = __make_df_groupby_visit_number(result, is_test_df, is_use_positive_scancount_only)
    
    return __make_weekday_as_dummies(result)

def __display_parameter_detail(dummie_col, is_use_positive_scancount_only, is_test_df, is_need_null_column):
    display(Markdown("##### Dummy타입으로 만든 컬럼 명 : " + dummie_col))
    display(Markdown("##### ScanCount는 양수만 사용")) if is_use_positive_scancount_only else display(Markdown("##### ScanCount는 음수만 사용"))
    display(Markdown("##### Test df 만드는 중")) if is_test_df else display(Markdown("##### Train df 만드는 중"))
    display(Markdown("##### Null 컬럼을 만듬")) if is_need_null_column else display(Markdown("##### Null 컬럼 없는 모델"))
    print()
    if not is_test_df:
        display(Markdown("> 위 정보들을 Display하는 이유는 이번 FeatureMatrix를 사용한 모델에 Fit할 Test 모델 만들 때 같은 전처리를 하기 위해서다."))

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
#     should_remove_cols = ["Upc", "DepartmentDescription", "FinelineNumber"]
    new_cols = [col for col in df.columns if col not in should_remove_cols]
    return df[new_cols]

def __make_fineline_number_dummy_columns(df, df_train, df_test, dummie_col):
    train_fl_li = [fl if not np.isnan(fl) else -1  for fl in df_train[dummie_col].unique()]
    test_fl_li = [fl if not np.isnan(fl) else -1  for fl in df_test[dummie_col].unique()]
    fl_cols = [li if not np.isnan(li) else -1.0 for li in list(set(list(test_fl_li) + list(train_fl_li)))]
    df = __make_dummy_columns(df, dummie_col)
    return df, fl_cols

def __make_df_groupby_visit_number(df, is_test_df, is_use_positive_scancount_only):        
    cols = [col for col in list(df.columns) if col != "VisitNumber"]
    np_max_cols = cols[:2] if is_test_df else cols[:3]
    values = [np.max if col in np_max_cols else np.sum for col in cols]
    dict_ = dict(zip(cols, values))
    result_df = df.groupby(by='VisitNumber').agg(dict_).reset_index()
    
    # is_use_positive_scancount_only가 True인 경우에는 0이하의 ScanCount는 모두 0으로 만들어준다.
    if is_use_positive_scancount_only:
        result_df = pd.DataFrame(np.where(result_df < 0, 0, result_df), columns=result_df.columns)
    if is_test_df:
        # test경우에는 없는 컬럼이므로 0으로 채워서 추가해준다. (Model을 만들 때 사용한 feature_matrix와 shape이 같아야하므로)
        result_df["HEALTH AND BEAUTY AIDS"] = np.zeros(len(result_df))
        return result_df
    return result_df

def __make_weekday_as_dummies(df):
    # finelinenumber를 column으로 만든경우에는 1,2,3,4,5,6,7 중에 몇개가 중복될 위험이 있어서 다시 명시적으로 바꿔준다.
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    dummies_desc = pd.get_dummies(df["Weekday"])
    dummies_desc.columns = weekdays
    desc_cols = dummies_desc.columns
    df[desc_cols] = dummies_desc
    return df.drop("Weekday", axis = 1)

def __make_fl_df_using_for_sentence(df, df_train, df_test, dummie_col, is_use_positive_scancount_only, is_test_df, is_need_null_column):
    df[dummie_col] = df[dummie_col].apply(lambda a : -1.0 if np.isnan(a) else a)
    train_fl_li = [fl if not np.isnan(fl) else -1  for fl in df_train[dummie_col].unique()]
    test_fl_li = [fl if not np.isnan(fl) else -1  for fl in df_test[dummie_col].unique()]
    dd_cols = [li if not np.isnan(li) else -1.0 for li in list(set(list(test_fl_li) + list(train_fl_li)))]
    dd_cols.insert(0, "Return")
    dd_cols.insert(0, "VisitNumber")
    if not is_test_df:
        dd_cols.insert(2, "TripType")
        df = pd.DataFrame(df.groupby(["VisitNumber", "TripType", "Weekday", dummie_col]).sum()["ScanCount"]).reset_index()
    else:
        df = pd.DataFrame(df.groupby(["VisitNumber", "Weekday", dummie_col]).sum()["ScanCount"]).reset_index()
    return __makeDf(df, dd_cols, is_test_df, dummie_col, is_use_positive_scancount_only)

def __makeDf(df, dd_cols, is_test_df, dummie_col, is_use_positive_scancount_only):
    display(Markdown("##### 이 작업은 데이터프레임을 반환하는 \
    함수이므로 변수로 받아주셔야합니다!! 오래 걸리는 작업이므로 지금 실수하셨다면 빨리 커널 종료하고 다시 시도해주세요."))
    vn_uq_li = df["VisitNumber"].unique()
    
    # 아래 for문에서는 FinelineNumber에 대해서만 일처리를 하기때문에 마지막에 Weekday부분은 Concat해야된다.
    df = __make_weekday_as_dummies(df)
    weekdays = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    values = [np.max for col in weekdays]
    dict_ = dict(zip(weekdays, values))
    df_to_concat_later = df.groupby(by="VisitNumber").agg(dict_).reset_index().drop("VisitNumber", axis = 1)
    
    li = []
    for i, vn in enumerate(vn_uq_li):
        tmp_df = df[df["VisitNumber"] == vn]
        space = np.zeros(len(dd_cols)).astype(int) if not is_test_df else np.zeros(len(dd_cols)).astype(int)
        # 위에서 VisitNumber는 0으로 설정했으므로 0번 인덱스에 넣어준다.
        space[0] = vn
        # 위에서 Return 여부는 1으로 설정했으므로 1번 인덱스에 넣어준다. (Return했다면 1)
        space[1] = 1 if True in (0 > tmp_df["ScanCount"].unique()) else 0

        # Test_df가 아닌 경우에는 TripType도 넣어줘야한다. Fit이 가능하도록.
        if not is_test_df:
            tripType = tmp_df["TripType"].unique()[0]
            # 위에서 TripType은 2번으로 설정했으므로 2번 인덱스에 넣어준다.
            space[2] = tripType
            
        for row_nbr in tmp_df.index:
            dd = tmp_df.loc[row_nbr][dummie_col]
            scan_cnt = tmp_df.loc[row_nbr]["ScanCount"]
            idx = dd_cols.index(dd)
            if not is_use_positive_scancount_only:
                space[idx] = scan_cnt
            else:
                if scan_cnt > 0:
                    space[idx] = scan_cnt
        li.append(space)
        if (i % 5000) == 0:    
            print(str(i) + "명 진행됨. 아직 " + str(len(vn_uq_li) - i) + "명 데이터 남음.")
    
    df_fl = pd.DataFrame(li, columns=dd_cols) 
    
    return pd.concat([df_fl, df_to_concat_later], axis = 1)


#################################################################################################################################

from sklearn.naive_bayes import MultinomialNB
import xgboost

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

def fitNaiveBayesModel(X, y):
    return MultinomialNB().fit(X, y)

def fitXGBClassifier(X, y, n_estimators=100, max_depth=2):
    return xgboost.XGBClassifier(n_estimators = n_estimators, max_depth = max_depth)

def saveDataFrameToCsv(df, fileName, is_submission_df = False, idx = False):
    # fl을 이용해 만든 데이터프레임은 용량이 1Gb정도 되므로 저장시간이 8~9분 걸린다.
    """
        넘겨준 df를 filename + 년월일시간분 의 format으로 이루어진 이름의 파일로 생성해준다.
        index를 True로 넘겨주면 저장할 때 아규먼트로 index=True를 넣어주게 된다.
        is_submission_df 를 통해서 submission은 다른 폴더로 저장시킨다.
    """
    fileName += "_" + datetime.now().strftime("%Y%m%d%H%M") + ".csv"
    if is_submission_df:
        fileName = "Submission_models/" + fileName
    else:
        fileName = "Feature_matrix/" + fileName
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
        
def saveModelObjectAsPickle(model, fileName):
    filename = "Model_pkl/" + fileName
    joblib.dump(model, fileName)
    display(Markdown("##### Done!"))