from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import pandas as pd

#parameters
data_name = 'biodeg_norm'

#preprocessing
dir = './dataset/'+data_name+'.csv'
kf = KFold(n_splits=5, shuffle=True, random_state=0)
df = pd.read_csv(dir, index_col=0)
score_list = []

#execution
for train, test in kf.split(df):
    df_train = df.iloc[train]
    df_test = df.iloc[test]
    train_X = df_train.iloc[:, :-1]
    train_y = df_train.iloc[:, -1]
    test_X = df_test.iloc[:, :-1]
    test_y = df_test.iloc[:, -1]
    model = RandomForestClassifier(n_estimators = 50)
    model.fit(train_X, train_y)
    predict_y = model.predict(test_X)
    score = f1_score(test_y, predict_y)
    score_list.append(score)
print(score_list)