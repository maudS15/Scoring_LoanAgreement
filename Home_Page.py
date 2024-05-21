import streamlit as st


st.set_page_config(layout='wide', initial_sidebar_state='auto',page_title="Home Page")

st.markdown("# :rainbow[Home Page]")

st.markdown("This interface allows you to obtain the score assigned by the decision model for new clients based on the provided information.")

st.markdown('''Use the **sidebar** to navigate through the interface.''')

st.markdown("### Client Profile")
st.markdown('''Find information on clients awaiting a decision and obtain the score and graphs that show the data most influencing the model's decision.  
         Use the radio buttons to get more information with graphs that visualize the client's positioning among all previously recorded clients.  
         You can choose from the most relevant variables and also filter to obtain the distribution among records belonging to the same decision class.  
         You can also modify certain information and get an updated score.''')

st.markdown("### Enter a New Client")
st.markdown("This page allows you to enter the data of a new client and obtain the model's prediction score.")