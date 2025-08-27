import joblib
import pandas as pd

ourmodel = joblib.load('mymodel.model')
new_df = pd.DataFrame({'Income' :[44,44,44], 'Advertising':[6,3,11], 'Price':[105,88,77], 'Age':[33,55,22]})
new_pred = ourmodel.predict(new_df)
print('Sales 예측결과 :\n', new_pred)
