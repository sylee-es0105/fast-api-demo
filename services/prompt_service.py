from langchain_core.messages import SystemMessage

class PromptService:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptService, cls).__new__(cls)
        return cls._instance

    # TODO class로 만들어서 파라미터 받도록 바꾸기
    # TODO 기본 Rules도 용도에 따라 변경될 가능성이 있으므로 커스텀할 수 있게
    def build_system_message(identity: str, context: str, etc: str = "") -> SystemMessage:
        # TODO : Response Format을 커스텀할 수 있도록 변경하자
        return SystemMessage(f"""
            [Identity]
            {identity}

            [Rules]
            - 제공된 Context를 활용하여 질문에 답하세요.
            - 내부 지식만으로 정확한 답변을 할 수 없는 경우, 웹 검색을 시도하세요.
            - 웹 검색 결과를 활용할 경우, 반드시 신뢰할 수 있는 출처를 참고하고, 출처 URL을 Response Format의 sources 항목에 포함하세요.
            - 웹 검색을 시도했음에도 정확한 답변이 불가능한 부분은 추측하지 말고 모른다고 응답하세요.
            - 언어가 따로 명시되지 않은 경우, 반드시 한국어로 답변해주세요.
            - 반드시 존댓말로 답변해주세요.
            {etc}

            [Context]
            {context}

            [Response Format (JSON)]
            {{
                "answer": "(질문에 대한 답변)",
                "context": "(사용된 Context 본문)",
                "additional_info": "부가 설명 (선택사항)",
                "sources": ["(출처 URL 1)", "(출처 URL 2)", ...]
            }}
        """)
prompt_service = PromptService()