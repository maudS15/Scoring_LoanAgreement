import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import joblib
import shap



#80 selected features
features = ['AMT_INCOME_TOTAL', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
        'REGION_POPULATION_RELATIVE', 'DAYS_BIRTH', 'OWN_CAR_AGE', 'FLAG_MOBIL',
        'FLAG_WORK_PHONE', 'FLAG_CONT_MOBILE', 'FLAG_PHONE',
        'YEARS_BEGINEXPLUATATION_MEDI', 'COMMONAREA_MEDI', 'ELEVATORS_MEDI',
        'DAYS_LAST_PHONE_CHANGE', 'FLAG_DOCUMENT_6', 'FLAG_DOCUMENT_7',
        'FLAG_DOCUMENT_8', 'FLAG_DOCUMENT_9', 'FLAG_DOCUMENT_13',
        'NAME_CONTRACT_TYPE_Revolving loans', 'NAME_TYPE_SUITE_Family',
        'NAME_TYPE_SUITE_Other_A', 'NAME_TYPE_SUITE_Spouse, partner',
        'OCCUPATION_TYPE_HR staff', 'OCCUPATION_TYPE_High skill tech staff',
        'OCCUPATION_TYPE_IT staff', 'ORGANIZATION_TYPE_Business Entity Type 3',
        'ORGANIZATION_TYPE_Construction', 'ORGANIZATION_TYPE_Culture',
        'ORGANIZATION_TYPE_Electricity', 'ORGANIZATION_TYPE_Government',
        'ORGANIZATION_TYPE_Industry: type 1', 'ORGANIZATION_TYPE_Medicine',
        'ORGANIZATION_TYPE_Transport: type 4', 'ORGANIZATION_TYPE_XNA',
        'FONDKAPREMONT_MODE_org spec account', 'BURO_MONTHS_BALANCE_SIZE_MEAN',
        'BURO_CREDIT_ACTIVE_Active_MEAN', 'BURO_CREDIT_CURRENCY_nan_MEAN',
        'BURO_CREDIT_TYPE_Cash loan (non-earmarked)_MEAN',
        'BURO_CREDIT_TYPE_Interbank credit_MEAN',
        'CLOSED_DAYS_CREDIT_ENDDATE_MAX', 'PREV_AMT_APPLICATION_MIN',
        'PREV_AMT_DOWN_PAYMENT_MAX', 'PREV_AMT_DOWN_PAYMENT_MEAN',
        'PREV_DAYS_DECISION_MEAN', 'PREV_CNT_PAYMENT_MEAN',
        'PREV_NAME_CONTRACT_TYPE_Cash loans_MEAN',
        'PREV_FLAG_LAST_APPL_PER_CONTRACT_N_MEAN',
        'PREV_NAME_CASH_LOAN_PURPOSE_Building a house or an annex_MEAN',
        'PREV_NAME_CASH_LOAN_PURPOSE_Buying a holiday home / land_MEAN',
        'PREV_NAME_CASH_LOAN_PURPOSE_Buying a new car_MEAN',
        'PREV_NAME_CASH_LOAN_PURPOSE_Car repairs_MEAN',
        'PREV_NAME_GOODS_CATEGORY_Clothing and Accessories_MEAN',
        'PREV_NAME_GOODS_CATEGORY_Other_MEAN', 'PREV_NAME_PORTFOLIO_Cars_MEAN',
        'PREV_NAME_PORTFOLIO_Cash_MEAN', 'PREV_CHANNEL_TYPE_Stone_MEAN',
        'PREV_NAME_YIELD_GROUP_XNA_MEAN', 'PREV_NAME_YIELD_GROUP_high_MEAN',
        'PREV_NAME_YIELD_GROUP_low_normal_MEAN',
        'PREV_PRODUCT_COMBINATION_POS household with interest_MEAN',
        'PREV_PRODUCT_COMBINATION_POS household without interest_MEAN',
        'PREV_PRODUCT_COMBINATION_POS mobile without interest_MEAN',
        'PREV_PRODUCT_COMBINATION_POS other with interest_MEAN',
        'APPROVED_AMT_ANNUITY_MAX', 'APPROVED_AMT_ANNUITY_MEAN',
        'APPROVED_AMT_CREDIT_MAX', 'APPROVED_APP_CREDIT_PERC_MEAN',
        'APPROVED_APP_CREDIT_PERC_VAR', 'APPROVED_AMT_DOWN_PAYMENT_MIN',
        'APPROVED_AMT_DOWN_PAYMENT_MAX',
        'APPROVED_HOUR_APPR_PROCESS_START_MEAN',
        'APPROVED_RATE_DOWN_PAYMENT_MIN', 'APPROVED_RATE_DOWN_PAYMENT_MAX',
        'REFUSED_APP_CREDIT_PERC_MEAN', 'REFUSED_RATE_DOWN_PAYMENT_MAX',
        'INSTAL_PAYMENT_PERC_MAX', 'INSTAL_PAYMENT_PERC_SUM',
        'INSTAL_PAYMENT_DIFF_MEAN']

#list of clients without target
X = pd.read_csv("Cleaned/allfeatures_test.csv",
                index_col='SK_ID_CURR',
                usecols=['SK_ID_CURR']+features,
                nrows=10,
                )

#load the model (already fitted)
model = joblib.load("GradientBoosting")

#prediction: threshold 0.36 for the model
y_proba = model.predict_proba(X)[:,1]
seuil = 0.36
y_pred = [0 if value < seuil else 1 for value in y_proba]

prediction = pd.DataFrame(index=X.index,
                          columns=['Proba','Prediction'])
prediction['Proba'] = y_proba
prediction['Prediction'] = y_pred

#visual streamlit
st.set_page_config(layout="wide")
st.set_option('deprecation.showPyplotGlobalUse', False)

st.markdown("<h1 style='text-align: center;'>Loan scoring</h1>",
            unsafe_allow_html=True)
st.write("### ")

col1, col2 = st.columns(2, gap='large')


with col1:
    
    
    
    client_list = st.selectbox(label="Client",
                               options= prediction.index)
    
    
    def message(client_id):
        pred = prediction.loc[client_id, 'Prediction']
        global color_bar
        if pred==0:
            color_bar = "green"
            return f"<p style='font-size:36px; color:green;'>Accepted</p>"
        else:
            color_bar = "red"
            return f"<p style='font-size:36px; color:red;'>Refused</p>"
    
    st.markdown(f"<p style='font-size:20px;'>Prediction class: {prediction.loc[client_list, 'Prediction']}</p>",
                unsafe_allow_html=True)
    st.markdown(message(client_list),
                unsafe_allow_html=True)

    #gauge
    jauge = go.Figure(go.Indicator(domain = {'x': [0.25,0.75], 'y': [0.5, 1]},
                                 value = prediction.loc[client_list, 'Proba'],
                                 mode = "gauge+number",
                                 title = {'text': "Probability of class prediction"},
                                 gauge = {'axis': {'range': [0, 1]},
                                          'bgcolor': "white",
                                          'bar': {'color': color_bar},
                                          'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': seuil}}))
    st.plotly_chart(jauge, use_container_width=True)

    #SHAP
    X_preprocessed = pd.DataFrame(model['preprocessing'].transform(X),
                                 index=X.index,
                                 columns=features)
    
    explainer = shap.TreeExplainer(model['classifier'])
    observation = X_preprocessed.loc[[client_list]]
    shap_value = explainer.shap_values(observation)
    
    most_important_feats = pd.Series(shap_value[0], index=features).abs().sort_values(ascending=False)[:10].index
    
    if st.checkbox("Show the most important features for the client"):
        data_client = pd.DataFrame(index=most_important_feats)
        data_client['Real'] = X.loc[client_list, most_important_feats]
        data_client['Processed'] = X_preprocessed.loc[client_list, most_important_feats]
        st.dataframe(data_client.T)



with col2:
    plt.title("Features with the most impact", y=1.1, fontsize=24)
    shap.decision_plot(explainer.expected_value,
                       shap_value,
                       feature_names=features
                       )
    st.pyplot(bbox_inches='tight')