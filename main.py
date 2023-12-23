from fastapi import FastAPI, HTTPException, Query, UploadFile, File
from pydantic import BaseModel
from openai import OpenAI, OpenAIError
from fastapi.middleware.cors import CORSMiddleware

# APIクライアントの設定
client = OpenAI(api_key="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")

app = FastAPI()

# CORSミドルウェアの設定
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # すべてのオリジンを許可する場合
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# モデル定義
class CreateAssistantRequest(BaseModel):
    name: str
    instructions: str
    model: str = "gpt-4-1106-preview"
    tools: list = [{"type": "code_interpreter"}]

class CreateThreadRequest(BaseModel):
    pass

class AddMessageRequest(BaseModel):
    thread_id: str
    role: str
    content: str

class RunAssistantRequest(BaseModel):
    thread_id: str
    assistant_id: str

# エンドポイント定義
@app.post("/create_assistant")
def create_assistant(request: CreateAssistantRequest):
    try:
        assistant = client.beta.assistants.create(
            name=request.name,
            instructions=request.instructions,
            model=request.model,
            tools=request.tools
        )
        return {"assistant_id": assistant.id}
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

@app.post("/create_thread")
def create_thread(request: CreateThreadRequest):
    try:
        thread = client.beta.threads.create()
        return {"thread_id": thread.id}
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

@app.post("/add_message")
def add_message(request: AddMessageRequest):
    try:
        message = client.beta.threads.messages.create(
            thread_id=request.thread_id,
            role=request.role,
            content=request.content
        )
        return {"message_id": message.id}
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

@app.post("/run_assistant")
def run_assistant(request: RunAssistantRequest):
    try:
        run = client.beta.threads.runs.create(
            thread_id=request.thread_id,
            assistant_id=request.assistant_id
        )
        return {"run_id": run.id}
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

@app.get("/check_run_status")
def check_run_status(thread_id: str = Query(...), run_id: str = Query(...)):
    try:
        run_status = client.beta.threads.runs.retrieve(
            thread_id=thread_id,
            run_id=run_id
        )
        return {"status": run_status.status}
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

@app.get("/get_responses")
def get_responses(thread_id: str = Query(...)):
    try:
        messages = client.beta.threads.messages.list(
            thread_id=thread_id
        )
        responses = [message.content for message in messages.data if message.role == "assistant"]
        return {"responses": responses}
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")

@app.post("/upload_file")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_content = await file.read()
        uploaded_file = client.files.create(
            file=file_content,
            purpose='assistants'
        )
        return {"file_id": uploaded_file.id}
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"OpenAI API error: {e}")