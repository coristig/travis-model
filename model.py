import os
import pandas as pd

df_raw = pd.read_csv('./salary_data.csv', index_col=0)
col_names = df_raw.columns

### create some holdouts for unit testing
raw_test = pd.DataFrame(df_raw.sample(10))
raw_test = raw_test.drop(['salary'],axis=1)
raw_test.head()

####
df=df_raw
df.columns=col_names
#reordering our columns
df = df[['salary','gender','ranking','highest_deg','years_current_rank','years_since_deg']]

df=df.dropna()

############ SPLIT OUR DATA ########################
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(df[['gender','ranking','highest_deg','years_current_rank','years_since_deg']], df['salary'].values, test_size=0.1, random_state=42)

############ START TRANSFORMATION ########################

dummies = {}
for col in X_train.columns:
    if X_train[col].dtype==object:
        dummies[col]=df[col].unique()
        df[col]=df[col].astype('category', categories=dummies[col])


def transform_dummies(data,raw):
    if len(data)==1:
        df = pd.DataFrame(data,index=[0])
    else:
        df = pd.DataFrame(data)

    if raw==True:
        if 'salary' in data.columns:
            df = df.drop(['salary'],axis=1)

    for col in df:
        if col in dummies.keys() and str(df[col].dtype)!='category':
            cats = dummies.get(col).tolist()
            df[col] = df[col].astype('category',categories=cats)

    df2 = pd.get_dummies(df,drop_first=False)
    df2 = df2.drop(['gender_male','ranking_assistant','highest_deg_masters'], axis=1)

    return df2

################## MODELIN' #########################

#train our model
from sklearn.ensemble import RandomForestRegressor

RFmodel = RandomForestRegressor()
RFmodel.fit(transform_dummies(X_train,False), y_train)


training_val = RFmodel.score(transform_dummies(X_train,False), y_train)
testing_val = RFmodel.score(transform_dummies(X_test,False), y_test)
print "training:", testing_val
print "testing: ", training_val

############ DEPLOYMENT ######################

from yhat import Yhat, YhatModel, preprocess

class TravisModel(YhatModel):
    def fit_val(self):
        testing_val = RFmodel.score(transform_dummies(X_test, False), y_test)
        return testing_val

    def execute(self,data):
        data = transform_dummies(data,False)
        output = RFmodel.predict(data)
        return output.tolist()

########## DEPLOY SET #####################

if __name__ == '__main__':
    yh = Yhat(
        os.environ['YHAT_USERNAME'],
        os.environ['YHAT_APIKEY'],
        os.environ['YHAT_URL'],
    )
    yh.deploy("TravisModel", TravisModel, globals(), True)
