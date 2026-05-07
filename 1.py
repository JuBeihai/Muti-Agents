from langchain_community.llms import tongyi


# sk-3379d8673c5b4c2d846e24f65afc35a8



key= 'sk-3379d8673c5b4c2d846e24f65afc35a8'

llm = tongyi.Tongyi(api_key = key)


# 调用

llm.invoke("夏天适合吃什么水果？")
