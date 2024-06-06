import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_core.runnables import RunnablePassthrough
from langchain.prompts import PromptTemplate


st.title('🦜🔗 중국 주요 법률 정보 검색 Bot')

openai_api_key = st.text_input('OpenAI API Key', type='password')



def generate_response(text):
    from langchain_chroma import Chroma
    from langchain_openai import OpenAIEmbeddings

    # Chroma 인스턴스 생성
    persist_directory = "/Users/parkhyunwoo/Desktop/dev/Blog_Posting/chromadb"

    # 디스크에서 문서를 로드합니다.
    db3 = Chroma(persist_directory=persist_directory, embedding_function=OpenAIEmbeddings(api_key=openai_api_key))

    # 질의합니다.
    query = text
    docs = db3.similarity_search(query)
    print(docs)

    retriever = db3.as_retriever()

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        max_tokens=1000,
        temperature=0.7,
        api_key=openai_api_key
    )

    template ="""
    너는 중국 법률에 대한 전문가야. {question}과 관련된 법률에 대해 너의 기존 지식 & 내용과 {context}를 참조해서 답변해줘.
    format
    해석 : 
    """

    prompt = PromptTemplate(template=template, input_variables=['question'])

    # def format_docs(docs):
    #     # 검색한 문서 결과를 하나의 문단으로 합쳐줍니다.
    #     return "\n\n".join(doc.page_content for doc in docs)


    chain = {"context": retriever , "question": RunnablePassthrough()} | prompt  | llm 

    response = chain.invoke(text)
    print(response.content)
    
    return response



with st.form('my_form'):
    text = st.text_area('입력:', '중국 법률과 관련된 질의를 주세요')
    submitted = st.form_submit_button('Submit')
    if not openai_api_key.startswith('sk-'):
        st.warning('Please enter your OpenAI API key!', icon='⚠')
    if submitted and openai_api_key.startswith('sk-'):
        response = generate_response(text)
        st.write(response.content)



