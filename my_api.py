from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import argparse
import logging
import os
import json
import sys
from model import getTokenizerAndModel


# 超参数 用于控制模型回复时 上文的长度
MAX_HISTORY = 32

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


# 接入FastAPI
def start_server(quantize_level, http_address: str, port: int):
    tokenizer, model = getTokenizerAndModel(quantize_level)

    app = FastAPI()
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def index():
        return {"message": "Server started", "success": True}

    @app.post("/stream")
    def continue_question_stream(arg_dict: dict):
        def decorate(generator):
            for item in generator:
                yield f"data: {json.dumps(item, ensure_ascii=False)}\n\n"

        # inputs = [query, answer_prefix, max_length, top_p, temperature, allow_generate, history]
        try:
            query = arg_dict["query"]
            answer_prefix = arg_dict.get("answer_prefix", "")
            max_length = arg_dict.get("max_length", 256)
            top_p = float(arg_dict.get("top_p", 0.7))
            temperature = float(arg_dict.get("temperature", 1.0))
            history = arg_dict.get("history", [])
            logger.info("Query - {}".format(query))
            if answer_prefix:
                logger.info(f"answer_prefix - {answer_prefix}")
            history = history[-MAX_HISTORY:]
            if len(history) > 0:
                logger.info("History - {}".format(history))

            history = [tuple(h) for h in history]
            inputs = {
                "tokenizer": tokenizer,
                "query": query,
                "answer_prefix": answer_prefix,
                "max_length": max_length,
                "top_p": top_p,
                "temperature": temperature,
                "allow_generate": allow_generate,
                "history": history,
            }
            return StreamingResponse(decorate(model.predict_continue(**inputs)))
        except Exception as e:
            logger.error(f"error: {e}")
            return "ERROR"

    @app.post("/interrupt")
    def interrupt():
        allow_generate[0] = False
        logger.error("Interrupted.")
        return {"message": "OK", "success": True}

    @app.post("/check")
    def check_queue():
        return {
            "message": "generating" if allow_generate[0] else "idle",
            "generating": allow_generate[0],
            "success": True,
        }

    logger.info("Starting server...")
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
    start_server(args.quantize, args.host, int(args.port))
