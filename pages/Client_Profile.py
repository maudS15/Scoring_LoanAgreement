import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import shap




#meilleur seuil
seuil = 0.31

@st.cache_data
def load_data(url, nrows):
    df = pd.read_csv(url,
                     index_col='SK_ID_CURR',
                     nrows=nrows)
    return df

#list of clients without target
new_clients = load_data("Cleaned/app_test.csv",
              nrows=10).round(2)

new_clients_ids = new_clients.index
features = new_clients.columns

#baseline
old_clients = load_data('Cleaned/app_train.csv',
                  nrows=10000)
old_clients_target = old_clients['TARGET']
old_clients.drop(columns='TARGET', inplace=True)



#################### using API
#request predict_proba for all new clients
@st.cache_data
def requete(df):
    """Request predic_proba to API"""
    data = df.fillna('').to_dict(orient='split', index=False)
    
    #URL locale
    #url = 'http://127.0.0.1:5000/predict/'
    #URL API Azure
    url = 'https://scoringloanagreement.azurewebsites.net/predict/'
    # Envoyer la requête POST à votre API
    response = requests.post(url, json=data)
    
    return response.json()['probability']

# Series with predict_proba for new clients
new_clients_proba = pd.Series(requete(new_clients), index=new_clients_ids)



def get_pred(proba):
    
    #prediction: threshold 0.31 for the model
    if proba < seuil:
        return 0
    else:
        return 1



def message(pred):
    global color_bar
    if pred==0:
        color_bar = "green"
        return f"<p style='font-size:36px; color:green;'>Accepted</p>"
    else:
        color_bar = "red"
        return f"<p style='font-size:36px; color:red;'>Refused</p>"




####################   SHAP
@st.cache_data
def get_shap_values():
    #URL locale
    #url = 'http://127.0.0.1:5000/shap/'
    #URL API Azure
    url = 'https://scoringloanagreement.azurewebsites.net/shap/'
    
    data = new_clients.fillna('').to_dict(orient='split', index=False)
    # Envoyer la requête POST à votre API
    response = requests.post(url, json=data)
    
    
    df_shap = pd.DataFrame(data = response.json()['shap'],
                           columns=response.json()['feature_names_out'],
                           index = new_clients_ids)
    
    return response.json()['expected'], df_shap


#SHAP values for the new client sample
expected_value, df_shap = get_shap_values()

def regroup_feats(list_names_out, allow_repeat=True):
    """Return the corresponding list of original features from the shap features (after onehotencoding and feature engineering)
    Allow repeated values in regroup"""
    regroup = []
    for name in list_names_out:
        if name in ['PAYMENT_RATE', 'ANNUITY_INCOME_PERC', 'INCOME_CREDIT_PERC']:
            regroup.append(name)
        else:
            for f in features:
                if f in name:
                    if allow_repeat or f not in regroup:
                        regroup.append(f)
                
    return regroup
        
def clients_original_values(idx, feats):
    ''''Return list of the client's original values (feature engineering included)'''
    val = []
    for f in feats:
        if f in features:
            val.append(new_clients.loc[idx, f])
        elif f=='PAYMENT_RATE':
            val.append(round(new_clients.loc[idx, 'AMT_ANNUITY']/new_clients.loc[idx, 'AMT_CREDIT'], 2))
        elif f=='ANNUITY_INCOME_PERC':
            val.append(round(new_clients.loc[idx, 'AMT_ANNUITY']/new_clients.loc[idx, 'AMT_INCOME_TOTAL'],2))
        elif f=='INCOME_CREDIT_PERC':
            val.append(round(new_clients.loc[idx, 'AMT_INCOME_TOTAL']/new_clients.loc[idx, 'AMT_CREDIT'],2))
    return val

def local_feat_importance(idx):
    """"Display individual shap importance against average shap importance among the sample"""
    global_feat_imp = df_shap.abs().mean()
    local_feat_imp = df_shap.loc[idx, :].abs().sort_values(ascending=False)[:10]
    main_feats = local_feat_imp.index
    
    corresponding_original_feats = regroup_feats(main_feats)
    fig = go.Figure()
    
    fig.add_trace(
        go.Bar(x=main_feats, y=global_feat_imp.loc[main_feats],
               name="Global importance",
               marker_color='grey')
        )
    
    fig.add_trace(
        go.Bar(x=main_feats, y=local_feat_imp.values,
               name=f"Client {idx}",
               text=clients_original_values(idx, corresponding_original_feats),
               textposition='outside',
               marker_color='crimson')
    )


    fig.update_layout(hovermode="x unified",
                      font = dict(size=12, color="black"),
                      uniformtext_minsize=12,uniformtext_mode='show',
                      height=700,
                      title_text='10 most important features for the client compared to the average shap score.',
                      title_font_size = 24
    )
    
    fig.update_xaxes(tickfont =dict(size=14, color='black'))
    fig.update_yaxes(tickfont =dict(size=14, color='black'))
    
    return fig



################### Stats page 2
def get_color(cats, cat):
    """"Dictionary of color for plotly bar, color a specific category cat in crimson"""
    colors = {c : 'grey' for c in cats}
    colors[cat] = 'crimson'
    return list(colors.values())

def univariate(idx, category, by_class):
    """Display barplot for categorical features and boxplot for numerical features
    and position of the individual
    idx: index of the client
    category: feature chosen for the distribution to show
    by_class: bool: filter data among the same predicted class"""
    
    person = new_clients.loc[idx, category]
    
    if by_class:
        group_data = old_clients.loc[old_clients_target.loc[old_clients_target==pred].index, category]
        group = 'class '+str(pred)
    else:
        group_data = old_clients[category]
        group = 'All'
   
    #bar plot
    if group_data.nunique()<20 or old_clients.dtypes[category] == 'object':
        count = group_data.value_counts(normalize=True).round(2)
        
        color_map = get_color(count.index, person)
        
        fig = go.Figure(data=[go.Bar(
            x=count.index,
            y=count.values,
            text=count.values,
            marker_color=color_map,
            )])
        fig.update_layout(title_text=f"Client's {category} = {person} ")
    
    #box plot
    else:
        fig = go.Figure()
        
        fig.add_trace(go.Box(
            x=group_data,
            name=group,
            marker_color='grey',
        ))
        # Ajouter un point rouge pour le salaire particulier
        fig.add_trace(go.Scatter(
            y=[group], 
            x=[person], 
            mode='markers', 
            marker=dict(color='crimson', size=10),
            name='Client'
        ))
        
        # Mise en forme du graphique
        fig.update_layout(title=category + ' distribution' "  -> Client's value " + str(person),
                          xaxis_title='',
                          yaxis_title='')
        
        fig.update_layout(
            height=400)
    
    return fig
























#visual streamlit


#Title
st.markdown("<h1 style='text-align: center;'>Loan scoring</h1>",
            unsafe_allow_html=True)


 #Selectbox with the ids of the clients   
client_ID = st.selectbox(label="Client",
                           options= new_clients_ids)

proba = new_clients_proba[client_ID]
pred = get_pred(proba)

#Accepted or Refused
st.markdown(message(pred),
            unsafe_allow_html=True)






####################   SIDEBAR
st.sidebar.title("NAVIGATION")

pages = ["Main information", "Details", "Change values"]
page = st.sidebar.radio("Go to ", pages)








####################   PAGE1
if page==pages[0]:


    col1, col2, col3, col4 = st.columns([1.1,0.8,1,0.9], gap="small")
    
    with col1:
        st.write("Click on buttons to choose the plot :")
    
    with col2:
       b_gauge = st.button("Gauge", key="gauge")
    
    with col3:
       b_details = st.button("Feature importance", key='Details')
    
    with col4:
       b_decision = st.button("Decision plot", key="Decision")
    
    #Feature importance
    if b_details:
        fig = local_feat_importance(client_ID)
        st.plotly_chart(fig, use_container_width=True)
    
    #Shap decision plot
    elif b_decision:
        
        shap.decision_plot(expected_value,
                           df_shap.loc[client_ID].values,
                           feature_names=df_shap.columns.tolist(),
                           link="logit",
                           highlight=0,
                           show=False,
                           #return_objects=True
                           )
        decision = plt.gcf()
        st.pyplot(decision)
    
    else:
        #gauge
        jauge = go.Figure(go.Indicator(domain = {'x': [0.25,0.75], 'y': [0.5, 1]},
                                     value = proba,
                                     mode = "gauge+number+delta",
                                     number={'font_color':color_bar},
                                     delta = {'reference': seuil, 'decreasing': {'color': color_bar}, 'increasing': {'color': color_bar}},
                                     gauge = {'axis': {'range': [0, 1]},
                                              'bgcolor': "white",
                                              'bar': {'color': color_bar},
                                              'threshold' : {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': seuil}}))
        jauge.update_layout(title_text=f'Probability of class prediction compared to threshold {seuil}',
                          title_font_size = 24, font_size=16,
                          font = {'color': 'black'}
        )
        st.plotly_chart(jauge, use_container_width=False)



####################   PAGE2


if page==pages[1]:
#Comparison with old clients    

    #Select the first variable to analyse (ordered by SHAP importance)
    shap_by_imp = df_shap.loc[client_ID].abs().sort_values(ascending=False).index
    original_feats = regroup_feats(shap_by_imp, allow_repeat=False)
    var = st.selectbox(label="Choose the feature to analyse :",
                           options=original_feats)
    

    by_class = st.toggle("Filter by class :",
                         value=False)
    

    if by_class:
        st.markdown(f"<p style='font-size:20px;'>Data filtered among class : {pred}</p>",
                    unsafe_allow_html=True)
            
    
    fig = univariate(client_ID, var, by_class)
    st.plotly_chart(fig, use_container_width=True)
    
    
    if st.toggle("Second analysis :", value=False):
        original_feats.remove(var)
        var2 = st.selectbox(label="Choose the 2nd feature to analyse :",
                               options=original_feats)
        
        fig2 = univariate(client_ID, var2, by_class)
        st.plotly_chart(fig2, use_container_width=True)
        
        
        #bivariate analysis
        
        #scatterplot with color scaled by proba
        old_clients_proba = pd.Series(requete(old_clients), index=old_clients.index)
        

        fig3, ax = plt.subplots(figsize=(7,4))
        plt.title(f"Distribution between {var} and {var2}")
        c = sns.blend_palette(["#005500", "#FFFF90", "#AA0000", "#220000"], n_colors=4, as_cmap=True)



        if old_clients[var].nunique()>10 and old_clients[var2].nunique()>10:
            sns.scatterplot(data=old_clients,x=var, y=var2,
                              hue=old_clients_proba,hue_norm=(0,1),
                              palette=c,
                              )
            plt.xticks(fontsize=7)
        
        else:
            sns.stripplot(data=old_clients,x=var, y=var2,
                              hue=old_clients_proba,hue_norm=(0,1),
                              palette=c,
                              )
            plt.xticks(fontsize=7, rotation=45)
   
        plt.yticks(fontsize=7)
        
        plt.scatter(new_clients.loc[client_ID, var], new_clients.loc[client_ID, var2],
                    marker="*", s=500, c='black', zorder=10)
        
        plt.legend(bbox_to_anchor=(1,1), title="Probability")
        st.pyplot(fig3.get_figure())





####################   PAGE3
if page == pages[2]:
    
    st.write("Client's original values")
    st.dataframe(new_clients.loc[[client_ID]])
    
    col_form, col_res = st.columns(2, gap="medium")
    
    with col_form:
        with st.form("new_values"):
            #NAME_CONTRACT_TYPE
            feat = "NAME_CONTRACT_TYPE"
            options_list = old_clients[feat].unique().tolist()
            name_contrat = st.selectbox(label=feat,
                                        options=options_list)
            #NAME_FAMILY_STATUS
            feat = "NAME_FAMILY_STATUS"
            options_list = old_clients[feat].unique().tolist()
            name_family = st.selectbox(label=feat,
                                        options=options_list)
           
            #CNT_FAM_MEMBERS
            feat = "CNT_FAM_MEMBERS"
            cnt_fam = st.number_input(label=feat,
                                      min_value=0,
                                      step=1)
            
            #AMT_GOODS_PRICE
            feat = "AMT_GOODS_PRICE"
            amt_goods = st.number_input(label=feat,
                                  min_value=old_clients[feat].min()*0.9,
                                  max_value=old_clients[feat].max()*1.1)
            
            #EXT_SOURCE_1
            feat = "EXT_SOURCE_1"
            ext1 = st.slider(label=feat,
                                  min_value=0.0,
                                  max_value=1.0)
            
            
            #submit button
            submit = st.form_submit_button("Submit changes")
        
        
    with col_res:
        if submit:
            #get new result
            new_values = new_clients.loc[[client_ID]]
            for f, v in zip(["NAME_CONTRACT_TYPE", "NAME_FAMILY_STATUS", "CNT_FAM_MEMBERS","AMT_GOODS_PRICE", "EXT_SOURCE_1"],
                            [name_contrat, name_family, cnt_fam, amt_goods, ext1]):
                new_values.loc[client_ID, f] = v
            
            new_proba = requete(new_values)[0]
            new_pred = get_pred(new_proba)
            
            st.write("Compare original prediction with new.")
            st.dataframe(pd.DataFrame({'proba':[proba, new_proba],
                                       'class':[pred, new_pred]},
                                      index=['original', 'new']))








