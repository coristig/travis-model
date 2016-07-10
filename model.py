import os
import pandas as pd
from sklearn import linear_model
from sklearn import datasets

iris = datasets.load_iris()
X = pd.DataFrame(iris.data[:,0:3], columns=iris.feature_names[0:3])
y = pd.DataFrame(iris.data[:,3:4], columns=iris.feature_names[3:4])

regr = linear_model.LinearRegression()
regr.fit(X, y)

from yhat import Yhat, YhatModel, preprocess, df_to_json

class LinReg(YhatModel):
    REQUIREMENTS=["pandas","scikit-learn"]
    @preprocess(in_type=pd.DataFrame, out_type=dict)
    def execute(self, data):
       prediction = regr.predict(pd.DataFrame(data)).tolist()
       return {"prediction":prediction}

if __name__ == '__main__':
    yh = Yhat(
        os.environ['YHAT_USERNAME'],
        os.environ['YHAT_APIKEY'],
        os.environ['YHAT_URL'],
    )

yh.deploy("LinearRegression", LinReg, globals(), sure=True, autodetect=False)

