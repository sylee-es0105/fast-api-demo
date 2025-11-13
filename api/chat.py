from fastapi import APIRouter
from services.llm_service import llm_service
from services.vectorstore_service import vector_store_service

router = APIRouter()

# 동기 방식으로 LLM 호출, 응답 완료까지 대기
def _invoke(query: str):
    llm = llm_service.get_llm()
    response = llm.invoke(query)
    return {"response": response}

# 비동기 방식으로 LLM 호출, 다른 요청 처리 가능
async def _ainvoke(query: str):
    llm = llm_service.get_llm()
    response = await llm.ainvoke(query)
    return {"response": response}

# 동기 방식으로 LLM 응답을 스트리밍(응답이 부분 완료될 때마다 chunk로 나눠져 실시간 출력)
def _stream(query: str):
    llm = llm_service.get_llm()
    response = llm.stream(query)
    return {"response": response}

# 비동기 방식으로 LLM 응답을 스트리밍
async def _astream(query: str):
    llm = llm_service.get_llm()
    response = await llm.astream(query)
    return {"response": response}

@router.post("/rag")
def chat_with_context(query: str):
    database = vector_store_service.get_vectorstore()

    # query와 관련된 상위 n개의 문서를 검색한다. (기본값 k = 4)
    retrieved_docs = database.similarity_search(query)

    # TODO : 템플릿 설정

    prompt = f"""[Identity]
        - 당신은 최고의 한국 소득세 전문가입니다.
        - 질문자는 소득세에 관련해서 전혀 모르는 사람이라고 가정합니다.
        - [Context]를 참고해서 사용자의 [Question]에 답변해주세요.

        [Context]
        {retrieved_docs}

        [Question]
        {query}
    """

    return _stream(prompt)