from langchain_groq import ChatGroq

def get_llm(api):
    return ChatGroq(
        model = "llama-3.3-70b-versatile",
        api_key=api
    )