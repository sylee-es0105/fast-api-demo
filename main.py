from fastapi import FastAPI
from api import chat, test
from dotenv import load_dotenv

load_dotenv()

# 메모 (TODO 삭제)
# FastAPI 인스턴스 생성 => uvicorn 서버 실행 : [ uvicorn main:app --reload {--port <포트번호>} ]
# 만약 app 대신 다른 이름으로 인스턴스 생성 시 : [ uvicorn main:{인스턴스명} --reload ]
# --reload : 코드 변경 시 서버가 자동으로 재시작되도록 설정하는 옵션

# swagger 접속 : http://127.0.0.1:{포트번호}/docs

app = FastAPI()

app.include_router(test.router, prefix="/api/test", tags=["test"])
app.include_router(chat.router, prefix="/api/chat", tags=["chat"])