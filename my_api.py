from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import logging
import os
import json
import sys
from my_model import get_tokenizer_and_model, torch_gc

# 对话历史数上限
MAX_HISTORY = 64

# 默认文本长度上限
DEFAULT_MAX_LENGTH = 512

# 中断控制
allow_generate = [True]

# 默认端口号
default_port = 5178


# 接入log
def getLogger(name, file_name, use_formatter=True):
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    console_handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s    %(message)s")
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    if file_name:
        handler = logging.FileHandler(file_name, encoding="utf8")
        handler.setLevel(logging.INFO)
        if use_formatter:
            formatter = logging.Formatter("%(asctime)s - %(name)s - %(message)s")
            handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


logger = getLogger("ChatGLM", "chatlog.log")
sessionIndexHandle = [0]


# 接入FastAPI
def start_server(quantize_level, http_address: str, port: int):
    tokenizer, model = get_tokenizer_and_model(quantize_level)

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def alertError(e, sessionIndex):
        logger.error(f"Error {sessionIndex}: {e}")
        yield f"data: {e}\n\n"

    def decorate(generator, sessionIndex):
        lastStr = ""
        for item in generator:
            lastStr = item[0]
            yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"
        logger.info("Output {} - {}".format(sessionIndex, {"response": lastStr}))

    @app.get("/")
    def index():
        return {"message": "Server started", "success": True}

    @app.post("/stream")
    def continue_question_stream(arg_dict: dict):
        sessionIndexHandle[0] += 1
        sessionIndex = sessionIndexHandle[0]
        try:
            history = arg_dict.get("history", [])
            history = history[-MAX_HISTORY:]
            history = [tuple(h) for h in history]
            inputs = {
                "tokenizer": tokenizer,
                "query": arg_dict["query"],
                "answer_prefix": arg_dict.get("answer_prefix", ""),
                "max_length": arg_dict.get("max_length", DEFAULT_MAX_LENGTH),
                "top_p": float(arg_dict.get("top_p", 0.7)),
                "temperature": float(arg_dict.get("temperature", 1.0)),
                "allow_generate": allow_generate,
                "history": history,
            }
            logData = inputs.copy()
            del logData["tokenizer"]
            del logData["allow_generate"]
            logData["sessionIndex"] = sessionIndex

            logger.info(
                "Inputs {} - {}".format(
                    sessionIndex, json.dumps(logData, ensure_ascii=False)
                )
            )
            return StreamingResponse(
                decorate(model.my_stream_chat(**inputs), sessionIndex)
            )
        except Exception as e:
            return StreamingResponse(alertError(e, sessionIndex))

    @app.post("/interrupt")
    def interrupt():
        allow_generate[0] = False
        logger.info("Interrupted.")
        return {"message": "OK", "success": True}

    logger.info("System - Server started.")
    serverParams = {
        "host": http_address,
        "port": port,
        "quantize_level": quantize_level,
        "max_history": MAX_HISTORY,
        "default_max_length": DEFAULT_MAX_LENGTH,
    }
    logger.info(f"System - Confgis = { json.dumps(serverParams, ensure_ascii=False)}")
    uvicorn.run(app=app, host=http_address, port=port)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream API Service for ChatGLM2-6B")
    parser.add_argument(
        "--quantize", "-q", help="level of quantize, option：16, 8 or 4", default=4
    )
    parser.add_argument("--host", "-H", help="host to listen", default="0.0.0.0")
    parser.add_argument(
        "--port", "-P", help="port of this service", default=default_port
    )
    args = parser.parse_args()
    start_server(int(args.quantize), args.host, int(args.port))
