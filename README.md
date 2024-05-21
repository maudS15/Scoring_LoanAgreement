# Decision support model for granting credit to individuals with limited credit history.

## User interface
Streamlit Dashboard  
Home_Page.py is the main page. Two other pages are in the pages repository  
Available [here](https://scoringloanagreement-mauds.streamlit.app/)

## API
deployed on Azure
```
#request predict_proba, method POST
url_predict = https://scoringloanagreement.azurewebsites.net/predit/
#request shap values, method POST
url_shap = https://scoringloanagreement.azurewebsites.net/shap/
```

## Model pipeline

Feature engineering
```
data['INCOME_CREDIT_PERC'] = data['AMT_INCOME_TOTAL'] / data['AMT_CREDIT']
data['ANNUITY_INCOME_PERC'] = data['AMT_ANNUITY'] / data['AMT_INCOME_TOTAL']
data['PAYMENT_RATE'] = data['AMT_ANNUITY'] / data['AMT_CREDIT']
```

Feature selection: the following features present too high correlation with others
```
['APARTMENTS_MODE', 'BASEMENTAREA_MODE', 'YEARS_BEGINEXPLUATATION_MODE', 'YEARS_BUILD_MODE', 'COMMONAREA_MODE', 'ELEVATORS_MODE', 'ENTRANCES_MODE', 'FLOORSMAX_MODE',
 'FLOORSMIN_MODE', 'LANDAREA_MODE', 'LIVINGAPARTMENTS_MODE', 'LIVINGAREA_MODE', 'NONLIVINGAPARTMENTS_MODE', 'NONLIVINGAREA_MODE', 'APARTMENTS_MEDI', 'BASEMENTAREA_MEDI',
 'YEARS_BEGINEXPLUATATION_MEDI', 'YEARS_BUILD_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI', 'ENTRANCES_MEDI', 'FLOORSMAX_MEDI', 'FLOORSMIN_MEDI', 'LANDAREA_MEDI',
 'LIVINGAPARTMENTS_MEDI', 'LIVINGAREA_MEDI', 'NONLIVINGAPARTMENTS_MEDI', 'NONLIVINGAREA_MEDI', 'TOTALAREA_MODE', 'CNT_CHILDREN', 'AMT_CREDIT','AMT_ANNUITY',
'OBS_60_CNT_SOCIAL_CIRCLE','DEF_60_CNT_SOCIAL_CIRCLE','REGION_RATING_CLIENT_W_CITY']
```

Column transformer: SimpleImputer and scaling

SMOTENC:
balance the class distribution (from 1:11, to 1:2)

Column transformer: Apply OneHotEncoding to categorical features

Classifier: GradientBoostingClassifier from Scikit-learn
