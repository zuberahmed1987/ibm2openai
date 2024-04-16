from typing import Union
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from ibm_watson_machine_learning.foundation_models import Model
from dotenv import load_dotenv
from datetime import datetime
import logging
import json
import os

load_dotenv()

app = FastAPI()

ibm_url = os.getenv("IBM_WATSONX_URL")
ibm_apikey = os.getenv("IBM_WATSONX_API_KEY")
model_id = os.getenv("IBM_WATSONX_MODEL_ID")
project_id = os.getenv("IBM_WATSONX_PROJECT_ID")
space_id = os.getenv("IBM_WATSONX_SPACE_ID")
debug = os.getenv("DEBUG", False)
log_level = os.getenv("LOG_LEVEL", "INFO")

logger = logging.getLogger(__name__)

logging.basicConfig(level=log_level)

class Comp(BaseModel):
    model: str
    prompt: str
    max_tokens: int = 2000
    temperature: float = 0.2
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "gpt-3.5-turbo",
                    "prompt": "Generate sample code for a dockerfile for python application",
                    "max_tokens": "1000",
                    "temperature": 0.2
                }
            ]
        }
    }

class Messages(BaseModel):
    role: str
    content: str

class Chat(BaseModel):
    model: str 
    messages: list[Messages]
    max_tokens: int = 2000
    temperature: float = 0.2
    stream: bool = False
    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "model": "gpt-3.5-turbo",
                    "messages": [
                    {
                        "role": "user",
                        "content": "Generate sample code for a dockerfile for python application"
                    }
                    ],
                    "max_tokens": "1000",
                    "stream": False,
                    "temperature": 0.2
                }
            ]
        }
    }

def get_credentials():
	return {
		"url" : ibm_url,
		"apikey" : ibm_apikey
	}

# Call IBM Watsonx
model = Model(
    model_id = model_id,
    params = None,
    credentials = get_credentials(),
    project_id = project_id,
    space_id = space_id
)

def json_to_text(json_data):
    data = json_data
    lines = []
    for message in data["messages"]:
        sender = message.role
        text = message.content
        line = f"{sender}: {text}"
        lines.append(line)
    conversation = "\n".join(lines)
    return conversation

def chunk_reply(chunk, model_id, current_time):
    finish_reason = None
    if chunk == '<eos_token>':
        finish_reason = "stop"
        chunk = ''

    chunk_data =  {
        "id": f"chatcmpl-{current_time}",
        "object": "chat.completion.chunk",
        "created": current_time,
        "model": model_id,
        "system_fingerprint": f"fp_{current_time}",
        "choices": [
            {
            "index": 0,
            "delta": {
                "role": "assistant",
                "content": chunk
            },
            "message": {
                "role": "assistant",
                "content": chunk
            },
            "finish_reason": finish_reason
            }
        ]
    }
    return chunk_data
    
# for streaming
def data_generator(response, model_id, current_time):
    logger.debug("inside stream generator")
    for chunk in response:
        logger.debug(f"returned chunk: {chunk}")
        try:
            yield f"data: {json.dumps(chunk_reply(chunk, model_id, current_time).dict())}\n\n"
        except:
            yield f"data: {json.dumps(chunk_reply(chunk, model_id, current_time))}\n\n"
    yield f"data: {json.dumps(chunk_reply('<eos_token>', model_id, current_time))}\n\n"

@app.post('/v1/chat/completions')
async def chat(chat_data: Chat):
    data = chat_data
    parameters = {}
    logger.debug(f'Received request with data: {data}')
    messages = data.messages
    model_id = data.model
    stream = data.stream
    prompt_input = json_to_text({"messages": data.messages})
    prompt_input = prompt_input + "\n<end_of_code>\nassistant: "
    
    logger.debug(f'Received request with prompt: {prompt_input}')

    parameters = {
    "decoding_method": "sample",
    "max_new_tokens": data.max_tokens,
    "min_new_tokens": 1,
    "temperature": data.temperature,
    "repetition_penalty": 1,
    "stop_sequences": ["<end_of_code>"]
    }

    current_time = (datetime.now() - datetime(1970, 1, 1)).total_seconds()
    
    logger.info("Submitting generation request...")
    if stream == False:
        generated_response = model.generate(prompt=prompt_input, params=parameters, guardrails=True)

        content = generated_response["results"][0]["generated_text"]
        prompt_tokens = generated_response["results"][0]["input_token_count"]
        completion_tokens = generated_response["results"][0]["generated_token_count"]
        finish_reason = generated_response["results"][0]["stop_reason"]
        if finish_reason == 'eos_token':
            finish_reason = "stop"

        # Return response as OpenAI format
        return {
            "id": f"chatcmpl-{current_time}",
            "object": "chat.completions",
            "created": current_time,
            "model": model_id,
            "choices": [
                {
                "index": 0,
                "finish_reason": finish_reason,
                "message": {
                    "role": "assistant",
                    "content": content
                    }
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens
            }
        }
    else:
        generated_response = model.generate_text_stream(prompt=prompt_input, params=parameters, guardrails=True)
        return StreamingResponse(
            data_generator(generated_response,model_id, current_time),
            media_type="text/event-stream",
        )


@app.post('/v1/completions')
async def completions(comp_data: Comp):
    data = comp_data
    parameters = {}
    logger.debug(f'Received request with data: {data}')
    messages = data.prompt
    system_messages = "You are ChatGPT, a large language model trained by OpenAI. Follow the user's instructions carefully. Respond using markdown."

    prompt_input = f"""system: {system_messages}
user: create dockerfile for python app
assisassistant: 
```Dockerfile
FROM python:3.9-slim-buster
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . /app
EXPOSE 80
CMD ["python", "app.py"]
```

user: {messages}
assistant: """
    
    #logger.debug(f'Received request with prompt: {prompt_input}')

    parameters = {
    "decoding_method": "sample",
    "max_new_tokens": data.max_tokens,
    "min_new_tokens": 1,
    "temperature": data.temperature,
    "repetition_penalty": 1
    }

    logger.info("Submitting generation request...")
    generated_response = model.generate(prompt=prompt_input, params=parameters, guardrails=True)

    #logger.debug(f'Received responce: {generated_response}')

    current_time = (datetime.now() - datetime(1970, 1, 1)).total_seconds()
    content = generated_response["results"][0]["generated_text"]
    prompt_tokens = generated_response["results"][0]["input_token_count"]
    completion_tokens = generated_response["results"][0]["generated_token_count"]

    # Return response as OpenAI format
    return {
        'id': f"comp-{current_time}",
        'object': 'text_document',
        'created': current_time,
        'content': content,
        'metadata': {
            'status': {
                'code': 200,
                'message': 'OK'
            }
        },
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens
        }
    }

@app.get('/v1/models')
def models():
    current_time = (datetime.now() - datetime(1970, 1, 1)).total_seconds()
    return {
    "data": [
        {
        "id": "gpt-3.5-turbo",
        "object": "model",
        "created": current_time,
        "owned_by": "openai"
        },
        {
        "id": model_id,
        "object": "model",
        "created": current_time,
        "owned_by": "openai"
        }
    ],
    "object": "list"
    }


@app.get("/v1/models/{model_id}")
def models(model_id: str):
    #current_time = (datetime.now() - datetime(1970, 1, 1)).total_seconds()
    return {
    "id": model_id,
    "object": "model",
    "created": 1686935002,
    "owned_by": "openai"
    }

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5050)
