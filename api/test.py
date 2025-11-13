from fastapi import APIRouter
from services.llm_service import llm_service
from services.vectorstore_service import vector_store_service

router = APIRouter()


# 메모 (TODO 삭제)
# @ : '데코레이터'라고 불림
# [HttpGet("/")]와 대응됨 (경로 작동 데코레이터)

@router.get("/")
def root():
    return {"Hello": "World"}

@router.get("/items/{item_id}")
def test_items(item_id: int):
    return {"item_id": item_id }



@router.get("/llm-connection")
def test_llm_connection():
    try:
        llm = llm_service.get_llm()
        response = llm.invoke("안녕")
        return {
            "status": "success",
            "message": "LLM 연결 성공",
            "response": response.content if hasattr(response, 'content') else str(response)
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"LLM 연결 실패: {str(e)}"
        }

@router.get("/vectorstore-connection")
def test_vectorstore_connection():
    try:
        database = vector_store_service.get_vectorstore()
        
        if database is None:
            return {
                "status": "error",
                "message": "Vector DB가 초기화되지 않았습니다."
            }
        
        return {
            "status": "success",
            "message": "Vector DB 로드 성공",
            "db_type": type(database).__name__
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Vector DB 연결 실패: {str(e)}"
        }

