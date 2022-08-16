import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder,MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_validate
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier,GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings


warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_rows', 20)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


df = pd.read_csv("dataset/train.csv")

def DataFrame_Check(df):
    print("###################################### DATAFRAME CHECK ######################################\n\n")
    print("###################################### HEAD ######################################")
    print(df.head())
    print("###################################### TAIL ######################################")
    print(df.tail())
    print("###################################### COLUMNS ######################################")
    print(df.columns)
    print("###################################### INFO ######################################")
    df.info()
    print("###################################### DESCRIBE ######################################")
    print(df.describe().T)
    print("###################################### NULL VALUES ######################################")
    print(df.isnull().sum())
    print("###################################### SHAPE ######################################")
    print(df.shape)


#TARGET VALUES:
TARGET = "Survived"


def grab_columns(df,cat_th = 10, car_th = 20):
    """

    :param df: DataFrame
    :param cat_th: int:
    :param car_th: int:
    :return: cat_col, num_col, car_col

    Notes:
    cat_col: Categorical Columns
    num_col: Numerical Columns
    car_col: Cardinal Columns
    """

    cat_col = [col for col in df.columns if df[col].dtypes == "O"]
    num_but_cat = [col for col in df.columns if (df[col].dtypes != "O") and (df[col].nunique() < car_th)]
    cat_but_car = [col for col in df.columns if (df[col].dtypes == "O") and (df[col].nunique() > car_th)]

    cat_col = cat_col + num_but_cat
    cat_col = [col for col in cat_col if col not in cat_but_car]

    num_col = [col for col in df.columns if (df[col].dtypes != "O") and (col not in num_but_cat)]


    return cat_col,num_col,cat_but_car


cat_col, num_col, car_col = grab_columns(df)

###################################### ANALYSİS ######################################

#Numerical Analysis
def numerical_analysis(df,col,plot = False):
    """

    :param df: DataFrame
    :param col: Numerical Column
    :param plot: If you want ploting about your numerical columns
    """
    quantile = [0.05,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95]

    print(df[col].describe(quantile).T)
    print("############################")

    if plot:
        sns.histplot(df[col],color = "red")
        plt.xlabel(col)
        plt.title(col)
        plt.show(block = True)

#We should drop PassengerId because its not effective for our models.
num_col = [col for col in num_col if col != "PassengerId"]


for col in num_col:
    numerical_analysis(df,col,plot=True)

#Categorical Analysis

def categorical_analysis(df,col,plot = False):
    """
    :param df: DataFrame
    :param col: Categorical Column
    :param plot:If you want ploting about your categorical columns
    """

    print(pd.DataFrame({col:df[col].value_counts(),
                        "Ratio":100*df[col].value_counts()/len(df)}))
    print("######################################################")

    if plot:
        sns.countplot(x=df[col],data=df,color = "red")
        plt.title(col)
        plt.xticks(rotation = 30)
        plt.show(block = True)

for col in cat_col:
    categorical_analysis(df,col,plot=True)

cat_col = [col for col in cat_col if col != "Survived"]
def target_cat_summary(df,col,TARGET):
    print(col)
    print(pd.DataFrame({TARGET:df.groupby(col)[TARGET].mean(),
                        "Count":df[col].value_counts(),
                        "Ratio":100*df[col].value_counts()/len(df)}))

    print("############################################")

for col in cat_col:
    target_cat_summary(df,col,TARGET)

#Comment
"""

According to the target_cat_summary function, we can interpret that gender and passenger class affect survival.
"""

###############################
#OUTLIER VALUES
###############################

def outlier_thresholds(df,col, qu1 = 0.25, qu3 = 0.75, plot = False):
    """
    :param df: DataFrame
    :param col: Numerical Columns
    :param qu1: Fixed Value = 0.25
    :param qu3: Fixed Value = 0.75
    :return: low, up
    """
    q1 = df[col].quantile(qu1)
    q3 = df[col].quantile(qu3)

    ıqr = q3 - q1

    low = q1 - 1.5*ıqr
    up = q3 + 1.5*ıqr

    if plot:
        txt =col,low,up
        sns.boxplot(df[col])
        plt.title(txt)
        plt.show(block = True)

    return low,up


def check_outlier(df,col):
    low,up = outlier_thresholds(df,col)

    if df[(df[col] < low) | df[col] > up].any(axis = None):
        return True
    else:
        return False

for col in num_col:
    print(col,check_outlier(df,col))

"""
Outlier Check:
Age False
Fare False
"""


##############################
#NAN VALUES
##############################

df.isnull().sum()
"""
NAN VALUES
PassengerId      0
Survived         0
Pclass           0
Name             0
Sex              0
Age            177
SibSp            0
Parch            0
Ticket           0
Fare             0
Cabin          687
Embarked         2
"""

############################################
#Feature Extraction And Feature Interaction
############################################

#Categorical Cabin
df.loc[df["Cabin"].isnull(),"Cat_Cabin"] = 0
df["Cat_Cabin"] = df["Cat_Cabin"].fillna(1)
df = df.drop(["Cabin"],axis=1)


#Do you have any relatives
df.loc[(df["SibSp"] + df["Parch"] > 0),"Relative"] = 1
df.loc[(df["SibSp"] + df["Parch"] <= 0),"Relative"] = 0
df["SUM_Relative"] = df["SibSp"] + df["Parch"]


#Categorical Person Class
df.loc[(df["Pclass"] == 1),"CAT_Pclass"] = "First_Class"
df.loc[(df["Pclass"] == 2),"CAT_Pclass"] = "Second_Class"
df.loc[(df["Pclass"] == 3),"CAT_Pclass"] = "Third_Class"


#Person title
df["Person_Title"] = df["Name"].apply(lambda x: x.split(",")[1].split(".")[0].strip())
df["Person_Title"].unique()

df.dropna(inplace = True,axis=0)
df.isnull().sum()

cat_col,num_col,car_col = grab_columns(df,col)
num_col = [col for col in num_col if col != "PassengerId"]
#######################
#ENCODİNG
#######################


#LABEL ENCODİNG
def label_encoder(dataframe, binary_col):
    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])
    return dataframe

binary_cols = [col for col in df.columns if df[col].dtypes == "O" and df[col].nunique() == 2]
binary_cols


for col in binary_cols:
    df =label_encoder(df,col)

#ONE HOT ENCODİNG
onehot = [col for col in cat_col if col not in binary_cols and col not in ["Survived"]]

def one_hot_encoder(dataframe, categorical_cols, drop_first=False):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

df = one_hot_encoder(df, onehot, drop_first=True)


###########################
#SCALİNG
###########################

scaler = MinMaxScaler()

df[num_col] = scaler.fit_transform(df[num_col])



###################
#MODEL
###################

def MODEL(df):

    y = df["Survived"]
    X = df.drop(["PassengerId", "Survived", "Name", "Ticket"], axis=1)

    #Logistic Regression
    logistic_reg_model = LogisticRegression().fit(X,y)

    cv_result_l_r = cross_validate(logistic_reg_model,X,y,cv=5,scoring=["accuracy","f1","roc_auc","precision","recall"])

    print("-------LOGISTIC REGRESSION-------")
    print("\n-Cross Validate-")
    for i in cv_result_l_r:
        print(i,": ",cv_result_l_r[i].mean())

    """
fit_time :  0.03073606491088867
score_time :  0.013050651550292969
test_accuracy :  0.8160346695557964
test_f1 :  0.7682344280824702
test_roc_auc :  0.8694113325911127
test_precision :  0.7872405443767221
test_recall :  0.7565638233514822
"""

    print("\n-Holdout-")
    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.3,random_state=123)
    logistic_reg_model_hol = LogisticRegression().fit(X_train,y_train)
    y_pred_logistic = logistic_reg_model_hol.predict(X_test)
    print(classification_report(y_test,y_pred_logistic))
    """
                  precision    recall  f1-score   support
           0       0.81      0.89      0.85       125
           1       0.82      0.71      0.76        89
    accuracy                           0.81       214
   macro avg       0.81      0.80      0.80       214
weighted avg       0.81      0.81      0.81       214
    """

    print("------ Random Forest -------")
    rf_model = RandomForestClassifier().fit(X,y)
    cv_result_rf_model = cross_validate(rf_model,X,y,cv=5,scoring=["accuracy","f1","recall","precision","roc_auc"])
    print("Cross Validate")
    for i in cv_result_rf_model:
        print(i,": ",cv_result_rf_model[i].mean())

    """
fit_time :  0.2654531002044678
score_time :  0.04468932151794434
test_accuracy :  0.7907810499359795
test_f1 :  0.7366725997684294
test_recall :  0.7327283726557774
test_precision :  0.7471544327819963
test_roc_auc :  0.8549538908122232

    """

    print("Holdout")
    rf_model_hol = RandomForestClassifier().fit(X_train,y_train)
    y_pred_rf_model = rf_model_hol.predict(X_test)
    print(classification_report(y_test,y_pred_rf_model))

    """
                  precision    recall  f1-score   support
           0       0.83      0.88      0.85       125
           1       0.81      0.74      0.78        89
    accuracy                           0.82       214
   macro avg       0.82      0.81      0.81       214
weighted avg       0.82      0.82      0.82       214
    """

    print("Grid Search CV")
    rf_params = {"max_depth":[None,3,5,7,9],
                   "min_samples_split":range(2,10),
                   "max_features":["sqrt","log2",4],#auto removed 1.1 version
                   "min_samples_leaf":[2,4,6,8],
                   "max_samples":range(1,10)
                   }
    grid_search = GridSearchCV(rf_model,
                               rf_params,
                               cv=5,
                               verbose=1,
                               n_jobs=-1).fit(X,y)
    # scoring=["accuracy","recall","f1","roc_auc","precision"] is not necessary

    print(grid_search.best_params_)

    rf_grid_model = rf_model.set_params(**grid_search.best_params_)


    cv_result_grid_search_cv = cross_validate(rf_grid_model,
                                              X,y,
                                              cv=5,
                                              scoring=["accuracy","f1","recall","precision","roc_auc"])
    try:
      for i in cv_result_grid_search_cv:
          print(i,": ",cv_result_grid_search_cv[i].mean())

    except RuntimeError:
        print("Runtime Error Ignored")

    """
    {'max_depth': 3, 'max_features': 'sqrt', 'max_samples': 9, 'min_samples_leaf': 2, 'min_samples_split': 3}
fit_time :  0.21257624626159669
score_time :  0.04698853492736817
test_accuracy :  0.7499556781246922
test_f1 :  0.5625720163360959
test_recall :  0.41609195402298854
test_precision :  0.9391116567961866
test_roc_auc :  0.8372477593578266
    """


    print("------- KNN -------")
    print("Cross Validate")

    knn_cv = KNeighborsClassifier(n_neighbors=5).fit(X,y)
    cv_knn = cross_validate(knn_cv,
                            X,y,
                            cv=5,
                            n_jobs=-1,
                            scoring=["accuracy","f1","recall","precision","roc_auc"])
    for i in cv_knn:
        print(i,": ",cv_knn[i].mean())

    """
fit_time :  0.012503814697265626
score_time :  0.22011985778808593
test_accuracy :  0.7711513838274402
test_f1 :  0.6941217808789967
test_recall :  0.6493647912885663
test_precision :  0.7572379015170776
test_roc_auc :  0.8309158037080531
    """

    print("Holdout")
    knn_hol = KNeighborsClassifier(n_neighbors=5).fit(X_train,y_train)
    y_pred_knn_hol = knn_hol.predict(X_test)
    print(classification_report(y_test,y_pred_knn_hol))

    """
                  precision    recall  f1-score   support
           0       0.77      0.91      0.83       125
           1       0.83      0.61      0.70        89
    accuracy                           0.79       214
   macro avg       0.80      0.76      0.77       214
weighted avg       0.79      0.79      0.78       214

    """

    print("Grid Search CV")
    knn_params = {"n_neighbors":range(5,12),
                  "algorithm":["ball_tree","kd_tree","brute"],
                  "p":[1]}

    grid_search_knn = GridSearchCV(knn_cv,
                                   knn_params,
                                   cv=5,
                                   n_jobs=-1).fit(X,y)
    print(grid_search_knn.best_params_)
    knn_final = knn_cv.set_params(**grid_search_knn.best_params_)

    cv_result_knn_grid = cross_validate(knn_final,
                                        X,y,
                                        cv=5,
                                        n_jobs=-1,
                                        scoring=["accuracy","f1","recall","precision","roc_auc"])
    for i in cv_result_knn_grid:
        print(i,": ",cv_result_knn_grid[i].mean())

    """
{'algorithm': 'ball_tree', 'n_neighbors': 9, 'p': 1}
fit_time :  0.008517169952392578
score_time :  0.09408841133117676
test_accuracy :  0.7949867034374076
test_f1 :  0.7263076356189092
test_recall :  0.6803992740471869
test_precision :  0.7905974025974026
test_roc_auc :  0.8565945289186455
    """

    print("------- CART -------")
    print("\nCross Validate")
    cart = DecisionTreeClassifier().fit(X,y)
    cv_result_cart = cross_validate(cart,
                                    X,y,
                                    cv=5,
                                    n_jobs=-1,
                                    scoring=["accuracy","f1","recall","precision","roc_auc"]
                                    )
    for i in cv_result_cart:
        print(i,": ",cv_result_cart[i].mean())

    """
fit_time :  0.015633010864257814
score_time :  0.03349757194519043
test_accuracy :  0.7570767260908106
test_f1 :  0.7006024155744804
test_recall :  0.7047791893526922
test_precision :  0.6987808045087013
test_roc_auc :  0.7505275740060089

    """


    print("\nHoldout")
    cart_hol = DecisionTreeClassifier().fit(X_train,y_train)
    y_pred_cart = cart_hol.predict(X_test)
    print(classification_report(y_test,y_pred_cart))

    """
                  precision    recall  f1-score   support
           0       0.76      0.82      0.79       125
           1       0.72      0.63      0.67        89
    accuracy                           0.74       214
   macro avg       0.74      0.73      0.73       214
weighted avg       0.74      0.74      0.74       214
    """

    print("\nGrid Search CV")
    cart_params = {"criterion":["gini","entropy","log_loss"],
                   "min_samples_split":range(2,5),
                   "max_features":["sqrt","log2"],}

    grid_search_cart = GridSearchCV(cart,cart_params,cv=5,n_jobs=-1).fit(X,y)
    print(grid_search_cart.best_params_)
    cart_final = cart.set_params(**grid_search_cart.best_params_)
    cv_result_cart_grid = cross_validate(cart_final,
                                         X,y,
                                         n_jobs=-1,
                                         scoring=["accuracy", "f1", "recall", "precision", "roc_auc"]
                                         )
    for i in cv_result_cart_grid:
        print(i,": ",cv_result_cart_grid[i].mean())

    """
{'criterion': 'entropy', 'max_features': 'sqrt', 'min_samples_split': 4}
fit_time :  0.0031346797943115233
score_time :  0.01562972068786621
test_accuracy :  0.7766768442824781
test_f1 :  0.7152844816529788
test_recall :  0.6976406533575318
test_precision :  0.7369099341876112
test_roc_auc :  0.787145500668507
    """

    print("\n------- Gradient Boosting Decesion Tree -------")
    print("\nCross Validate")
    gbm = GradientBoostingClassifier(random_state=17).fit(X,y)
    cv_result_gbm = cross_validate(gbm,
                                   X,y,
                                   cv=5,
                                   n_jobs=-1,
                                   scoring = ["accuracy","f1","recall","precision","roc_auc"])

    for i in cv_result_gbm:
        print(i,": ",cv_result_gbm[i].mean())

    """
fit_time :  0.8236753940582275
score_time :  0.01302180290222168
test_accuracy :  0.8174529695656456
test_f1 :  0.7540895998143329
test_recall :  0.7011494252873564
test_precision :  0.8210602664136678
test_roc_auc :  0.8747771050343912
    """

    print("\nHoldout")
    gbm_hol = GradientBoostingClassifier(random_state=17).fit(X_train,y_train)
    y_pred_gbm_hol = gbm_hol.predict(X_test)
    print(classification_report(y_test,y_pred_gbm_hol))

    """
                  precision    recall  f1-score   support
           0       0.80      0.88      0.84       125
           1       0.81      0.70      0.75        89
    accuracy                           0.80       214
   macro avg       0.80      0.79      0.79       214
weighted avg       0.80      0.80      0.80       214
    """

    print("\nGrid Search CV")
    gbm_params = {"loss":["log_loss","exponential"],
                  "learning_rate":[0.01,0.05,0.1,0.15],
                  "criterion":["friedman_mse","squared_error"],
                  "min_samples_split":range(2,10),
                  "random_state":range(50)
                  }
    #criterion -> mse deprecated in v1.0 and will be removed in v1.2
    #loss-> deviance depracted in v1.1
    gbm_grid = GridSearchCV(gbm,gbm_params,cv=5,n_jobs=-1,verbose=1).fit(X,y)
    print(gbm_grid.best_params_)
    gbm_final_model = gbm.set_params(**gbm_grid.best_params_).fit(X,y)
    cv_result_gbm_grid = cross_validate(gbm_final_model,
                                        X,y,
                                        cv=5,
                                        scoring=["accuracy","f1","recall","precision","roc_auc"])

    for i in cv_result_gbm_grid:
        print(i,": ",cv_result_gbm_grid[i].mean())

    """
    {'criterion': 'friedman_mse', 'learning_rate': 0.15, 'loss': 'log_loss', 'min_samples_split': 4, 'random_state': 0}
    fit_time :  0.2600370407104492
    score_time :  0.015630197525024415
    test_accuracy :  0.8329360780065006
    test_f1 :  0.7735970793880693
    test_recall :  0.7150635208711436
    test_precision :  0.8523576049812931
    test_roc_auc :  0.879351039871484
    """
    print("\n------- LightGBM -------")
    print("\nCross Validate")
    lgbm_model = LGBMClassifier(random_state=17).fit(X,y)
    cv_result_lgbm = cross_validate(lgbm_model,X,y,cv=5,scoring=["accuracy","f1","recall","precision","roc_auc"])
    for i in cv_result_lgbm:
        print(i,": ",cv_result_lgbm[i].mean())

    """
    fit_time :  0.10822114944458008
score_time :  0.017604923248291014
test_accuracy :  0.8202501723628484
test_f1 :  0.7673351623794101
test_recall :  0.7395039322444041
test_precision :  0.799998016168229
test_roc_auc :  0.8768167630028417

    """

    print("\nHoldout")
    lgbm_hol = LGBMClassifier(random_state=17).fit(X_train,y_train)
    y_pred_lgbm_hol = lgbm_hol.predict(X_test)
    print(classification_report(y_test,y_pred_lgbm_hol))

    """
                  precision    recall  f1-score   support
           0       0.80      0.89      0.84       125
           1       0.82      0.70      0.75        89
    accuracy                           0.81       214
   macro avg       0.81      0.79      0.80       214
weighted avg       0.81      0.81      0.81       214
    """

    print("\nGrid Search CV")
    lgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [100, 300, 500, 1000],
                   "colsample_bytree": [0.5, 0.7, 1]}

    lgbm_grid = GridSearchCV(lgbm_model,lgbm_params,cv=5,n_jobs=-1,verbose=1).fit(X,y)
    print(lgbm_grid.best_params_)
    lgbm_final_model = lgbm_model.set_params(**lgbm_grid.best_params_).fit(X,y)
    cv_result_lgbm_grid = cross_validate(lgbm_final_model,
                                         X,y,
                                         cv=5,
                                         scoring=["accuracy","f1","recall","precision","roc_auc"])

    for i in cv_result_lgbm_grid:
        print(i,": ",cv_result_lgbm_grid[i].mean())

    """
    {'colsample_bytree': 1, 'learning_rate': 0.01, 'n_estimators': 1000}
fit_time :  0.4224067687988281
score_time :  0.01894855499267578
test_accuracy :  0.8469319412981384
test_f1 :  0.8011293320615355
test_recall :  0.7705384150030247
test_precision :  0.8397594997594997
test_roc_auc :  0.8854118816310553

    """


    print("\n------- CatBoost -------")
    print("\nCross Validate")
    catboost_model = CatBoostClassifier(random_state=17,verbose=False)

    cv_result_catboost = cross_validate(catboost_model,X,y,cv=5,scoring=["accuracy","f1","roc_auc","recall","precision"])
    for i in cv_result_catboost:
        print(i,": ",cv_result_catboost[i].mean())

    """
    fit_time :  2.8319028854370116
score_time :  0.01250905990600586
test_accuracy :  0.8202600216684723
test_f1 :  0.7592662347781562
test_roc_auc :  0.8782122649422746
test_recall :  0.7082274652147611
test_precision :  0.8309116199645604

    """
    print("\nHoldout")
    catboost_hol = CatBoostClassifier(random_state=17,verbose=False).fit(X,y)
    y_pred_catboost_hol = catboost_hol.predict(X_test)
    print(classification_report(y_test,y_pred_catboost_hol))

    """
                  precision    recall  f1-score   support
           0       0.84      0.95      0.89       125
           1       0.92      0.75      0.83        89
    accuracy                           0.87       214
   macro avg       0.88      0.85      0.86       214
weighted avg       0.87      0.87      0.87       214

    """

    print("\nGrid Search CV")
    catboost_params = {"iterations": [200, 500],
                       "learning_rate": [0.01, 0.1],
                       "depth": [3, 6]}

    catboost_best_grid = GridSearchCV(catboost_model, catboost_params, cv=5, n_jobs=-1, verbose=True).fit(X, y)
    print(catboost_best_grid.best_params_)
    catboost_final = catboost_model.set_params(**catboost_best_grid.best_params_, random_state=17).fit(X, y)

    cv_results = cross_validate(catboost_final, X, y, cv=5, scoring=["accuracy", "f1", "roc_auc"])

    for i in cv_results:
        print(i, ": ", cv_results[i].mean())

    """
{'depth': 3, 'iterations': 200, 'learning_rate': 0.1}
fit_time :  0.292034387588501
score_time :  0.007940196990966797
test_accuracy :  0.8328966807840047
test_f1 :  0.7759777414232955
test_roc_auc :  0.8801986202829589

    """


