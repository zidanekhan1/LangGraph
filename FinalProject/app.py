import streamlit as st
from agents.graph import app
from langchain_core.messages import HumanMessage
st.sidebar.title("Settings")
st.sidebar.text("COMING SOON⏱️")
st.set_page_config("langgraphquery")
st.title("GRAPH QUERY MODEL")

user_query=st.text_input("Ask something")
if user_query:
    response = app.invoke({"messages":[HumanMessage(content=user_query)]})
    final_message = response["messages"][-1].content
    st.write(final_message)