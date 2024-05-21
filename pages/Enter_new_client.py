import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go
from datetime import datetime, date


example = pd.read_csv('Cleaned/app_train.csv', index_col='SK_ID_CURR', nrows=50000)
example.drop(columns='TARGET', inplace=True)
features = example.columns.tolist()


select = example.select_dtypes(include='object').columns.tolist()
flags = [c for c in example.select_dtypes(include='int').columns if example[c].nunique()==2 or 'FLAG' in c]
days = [c for c in features if 'DAYS' in c]
living_place = [c for c in example.select_dtypes(exclude='object').columns if c[-3:] in ['AVG', 'ODE', 'EDI']]
nums = [c for c in features if c not in select+flags+days+living_place]


@st.cache_data
def requete(df):
    """Request predic_proba to API"""
    data = df.fillna('').to_dict(orient='split', index=False)
    
    #URL locale
    url = 'http://127.0.0.1:5000/predict/'
    #URL API Azure
    #url = 'https://scoringloanagreement.azurewebsites.net/predict/'
    # Envoyer la requête POST à votre API
    response = requests.post(url, json=data)
    
    return response.json()['probability']

seuil = 0.31
def get_pred(proba):
    
    #prediction: threshold 0.31 for the model
    if proba < seuil:
        return 0
    else:
        return 1

def buil_df_client():
    
    values = {c:[st.session_state[c]] for c in features}
    for c in days:
        if st.session_state[c]:
            d = st.session_state[c]
            today = date.today()
            values[c] = (d-today).days
    df = pd.DataFrame(data=values, columns=features)
    st.session_state.df = df

if 'df' not in st.session_state:
    submit = False
else:
    submit=True



#Title
st.markdown("<h1 style='text-align: center;'>New Client</h1>",
            unsafe_allow_html=True)




if submit:
    st.dataframe(st.session_state.df)
    proba = requete(st.session_state.df)[0]
    #st.write(f"Proba : {round(proba,2)}")
    pred = get_pred(proba)
    #st.write(f"Class: {pred}")
    color_bar='green' if pred==0  else 'red'
    #gauge
    jauge = go.Figure(go.Indicator(domain = {'x': [0.25,0.75], 'y': [0.5, 1]},
                                 value = proba,
                                 mode = "gauge+number",
                                title = {'text': "Probability of class prediction"},
                                 gauge = {'axis': {'range': [0, 1]},
                                          'bgcolor': "white",
                                          'bar': {'color': color_bar},
                                          'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': seuil}}))
    st.plotly_chart(jauge, use_container_width=True)






with st.form("FORM :"):
    st.form_submit_button("Submit New Client", on_click=buil_df_client)
    

    
    #categorical features
    with st.expander("Basic information", expanded=False):
        for c in select:
            st.selectbox(label=c,
                         options=example[c].unique().tolist(),
                         index=None,
                         key=c)
    #flags
    with st.expander("Flags", expanded=False):
        for c in flags:
            st.toggle(label=c,
                      value=False,
                      key=c)
    
    
    #dates
    with st.expander("Dates", expanded=False):
        for c in days:
            st.date_input(label=c,
                          value=None,
                          key=c,
                          format="YYYY-MM-DD")
    
    
    #numerical
    with st.expander("Integers", expanded=False):
        for c in nums:
            mini = 0 if example[c].dtype=='int' else 0.0
            st.number_input(label=c,
                            value=None,
                            min_value=mini,
                            key=c)
    
    #living place
    with st.expander("Others", expanded=False):
        for c in living_place:
            st.number_input(label=c,
                            value=None,
                            min_value=0,
                            key=c)
    

