import io
import os
import base64
import requests
import streamlit as st

from PIL import Image
from pinecone import Pinecone
from dotenv import load_dotenv
from utils import print_messages, StreamHandler
from langchain_core.messages import ChatMessage
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.callbacks.base import BaseCallbackHandler

# .env 파일 로드
load_dotenv()

OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_ENVIRONMENT = os.environ.get('PINECONE_ENVIRONMENT')
HOST=os.environ.get('HOST')
API_KEY=os.environ.get('API_KEY')
API_KEY_PRIMARY_VAL=os.environ.get('API_KEY_PRIMARY_VAL')
REQUEST_ID=os.environ.get('REQUEST_ID')

# pinecone에 연결
api_key = os.environ.get('PINECONE_API_KEY')
index_name = os.environ.get('INDEX_NAME')


# logo 이미지 base64 형식으로 변환
def image_to_base64(image):
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


logo_img = Image.open('basic_woomi_lynn.png')
st.set_page_config(page_title=" Woomi Lynn", page_icon=logo_img)

st.markdown(
    f"""
    <div style="display: flex; align-items: center;">
        <img src="data:image/png;base64,{image_to_base64(logo_img)}" alt="logo" width="50">
        <h1 style="margin-bottom: 0px;"> Woomi Lynn chatbot</h1>
    </div>
    """,
    unsafe_allow_html=True
)


if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Pinecone 클라이언트 구성
pc = Pinecone(api_key=api_key)

# Index 확인
pc.list_indexes()

print(pc.list_indexes())

index = pc.Index("canopy--document-uploader")

model_name = 'text-embedding-ada-002'

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
embedding = OpenAIEmbeddings()


# 이전 대화기록 출력
print_messages()


class CompletionExecutor(BaseCallbackHandler):
    def __init__(self, host, api_key, api_key_primary_val, request_id, stream_handler):
        super().__init__()
        self._host = host
        self._api_key = api_key
        self._api_key_primary_val = api_key_primary_val
        self._request_id = request_id
        self._stream_handler = stream_handler

    def execute(self, completion_request):
        headers = {
            'X-NCP-CLOVASTUDIO-API-KEY': self._api_key,
            'X-NCP-APIGW-API-KEY': self._api_key_primary_val,
            'X-NCP-CLOVASTUDIO-REQUEST-ID': self._request_id,
            'Content-Type': 'application/json; charset=utf-8',
            'Accept': 'text/event-stream'
        }

        with requests.post(self._host + '/testapp/v1/chat-completions/HCX-003',
                           headers=headers, json=completion_request, stream=True) as r:
            for line in r.iter_lines():
                if line:
                    response = line.decode("utf-8")
                    if response.startswith('data:{"message":{"role":"assistant","content":'):
                        response_text = response.split('"content":"')[-1]
                        response_text = response_text.split('"}')[0]
                        response_text = response_text.replace('\\n', '\n')
                        if not self._done_received:
                            self._stream_handler.on_llm_new_token(response_text)
                    elif response.strip() == 'data:{"message":{"role":"assistant","content":"DONE"}}':
                        self._done_received = True
                        break

    def handle_response(self, completion_request):
        self.execute(completion_request)


if __name__ == '__main__':
    user_input = st.chat_input("궁금하신 내용을 질문해 주세요.")

    if user_input:
        st.chat_message("user").write(f"{user_input}")
        st.session_state["messages"].append(ChatMessage(role="user", content=user_input))

        question = [user_input]

        embedded_question = embedding.embed_documents(question)

        query_result = index.query(
            vector=embedded_question,
            top_k=20,
            include_values=False,
            include_metadata=True
        )
        print(query_result.matches[0])

        # 결과를 출력한다.
        for result in query_result.matches:
            id = result.id
            text = result.metadata['text']  # 문서의 원본 텍스트
            score = result.score  # 문서의 유사도
            print(id, score)
            print("\n")
            print(text)
            print('=' * 10)

        preset_text = [{"role": "system",
                        "content": " - 다음 문서에만 기반하여 질문에 대답합니다. - 문서에서 알수 없는 내용인 경우 '자세한 사항은 반드시 견본주택 및 고객센터로 확인해 주시기 바랍니다. (안내사항의 오류가 있을 시는 관계법령이 우선합니다.) ' 라고 합니다."},
                       {"role": "user",
                        "content": user_input}]

        # Pinecone 검색 결과를 preset_text에 추가
        for result in query_result.matches:
            text = result.metadata['text']  # 문서의 원본 텍스트
            preset_text.append({"role": "system", "content": text})

        request_data = {
            'messages': preset_text,
            'topP': 0.6,
            'topK': 0,
            'maxTokens': 500,
            'temperature': 0.1,
            'repeatPenalty': 2.0,
            'stopBefore': [],
            'includeAiFilters': True,
            'seed': 0
        }

        stream_handler = StreamHandler(st.empty())

        completion_executor = CompletionExecutor(
            host='https://clovastudio.stream.ntruss.com',
            api_key=API_KEY,
            api_key_primary_val=API_KEY_PRIMARY_VAL,
            request_id=REQUEST_ID,
            stream_handler=stream_handler
        )

        completion_executor.handle_response(request_data)