from fastapi import FastAPI
from langchain_upstage import ChatUpstage
from langchain_upstage import UpstageEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

"""
FastAPI 인스턴스 생성 => uvicorn 서버 실행 : [ uvicorn main:app --reload ]
만약 app 대신 다른 이름으로 인스턴스 생성 시 : [ uvicorn main:{인스턴스명} --reload ]
--reload : 코드 변경 시 서버가 자동으로 재시작되도록 설정하는 옵션
"""
app = FastAPI()


"""
@ : '데코레이터'라고 불림

[HttpGet("/")]와 대응됨 (경로 작동 데코레이터)
"""
@app.get("/")
def root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def test_items(item_id: int):
    return {"item_id": item_id }

def load_chroma_db():
    # DB를 저장할 때 사용했던 것과 동일한 모델을 사용해야 한다
    embeddings = UpstageEmbeddings(model="embedding-query")

    database = Chroma(
        persist_directory="/vectorstore/chroma",
        embedding_function=embeddings,
        collection_name="chroma-tax"
    )
    return database

def load_upstage_model():
    # .env UPSTAGE_API_KEY 설정 필요
    llm = ChatUpstage()
    return llm

@app.post("/test-chat")
def test_chat(query: str):
    lim = load_upstage_model()
    response = lim.invoke(query)
    return {"response": response}

@app.post("/chat")
def execute_chat_query(query: str):
    llm = load_upstage_model()
    database = load_chroma_db()

    # query와 관련된 상위 n개의 문서를 검색한다.
    retrieved_docs = database.similarity_search(query)

    prompt = f"""[Identity]
        - 당신은 최고의 한국 소득세 전문가입니다.
        - 질문자는 소득세에 관련해서 전혀 모르는 사람이라고 가정합니다.
        - [Context]를 참고해서 사용자의 [Question]에 답변해주세요.

        [Context]
        {retrieved_docs}

        [Question]
        {query}
    """

    # invoke 는 stateless : 각 요청마다 독립적으로 처리된다.
    # 따라서, llm을 싱글톤으로 관리하더라도 해당 사용자의 질문이 다른 사용자의 질문에 간섭하거나 하는 문제가 없음.
    # 그러므로 llm은 서버 시작 시 한 번만 로드해서 싱글톤으로 관리하는 것이 좋음.
    response = llm.invoke(prompt)
    return {"response": response}