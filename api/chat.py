from fastapi import APIRouter
from services.llm_service import llm_service
from services.vectorstore_service import vector_store_service
from langchain_core.messages import SystemMessage, HumanMessage
from services.prompt_service import prompt_service

router = APIRouter()

# 응답 방식 4가지 - 테스트 해보면서 하나로 통일해서 쓰거나 옵션화

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

def _chat(prompt: list[str]):
    return _invoke(prompt)

@router.post("/direct")
def chat(prompt: str):
    return _invoke(prompt)

@router.post("/rag")
def chat_with_context(user_input: str):

    # query와 연관성이 높은 상위 k개의 데이터를 검색한다. (기본값 k = 4)
    retrieved_docs = vector_store_service.search(user_input)

    # 메모 (TODO 삭제) : AIMessage는 이전 응답 기록을 참고해서 답변시켜야 할 때, 이전 AI의 응답 내용을 담아 보내는 것.
    # 일회성 응답 시에는 생략한다.
    prompt = [
        SystemMessage(f"""
            [Identity]
            - 당신은 최고의 한국 소득세 전문가입니다.
            - 질문자는 소득세에 관련해서 전혀 모르는 사람이라고 가정합니다.
            - [Context]를 참고해서 사용자의 질문에 답변해주세요.
            - 반드시 존댓말로 답변해주세요.
            - 정확한 답변이 불가능하다면, 추측하지 말고 모른다고 응답하세요.

            [Context]
            {retrieved_docs}

            [Response Format (JSON)]
            {{
                "answer": "(질문에 대한 답변)",
                "context": "(사용된 Context 본문)",
                "additional_info": "부가 설명 (선택사항)"
            }}
        """),
        HumanMessage(user_input)
    ]

    return _chat(prompt)