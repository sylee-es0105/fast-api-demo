from pydantic import BaseModel

# 하고 싶은 것
# SystemMessage에 들어가는 내용을 구조화된 데이터로 관리하기
# 해당 프로퍼티에 값이 있다면
# [프로퍼티명]
# 값 (없으면 기본값)
# 이렇게 프로퍼티가 나중에 또 추가되더라도 자동으로 만들어지도록

class SystemMessageConfig(BaseModel):
    identity: str
    context: str # TODO 수정 필요
    rules: str

