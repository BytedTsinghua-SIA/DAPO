from typing import Dict, Optional, List
import logging

from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import StreamingResponse, JSONResponse

import ray
from ray import serve
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.engine.async_llm_engine import AsyncLLMEngine
from vllm.entrypoints.openai.cli_args import make_arg_parser
from vllm.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ErrorResponse,
)
from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
from vllm.entrypoints.openai.serving_models import OpenAIServingModels, BaseModelPath, LoRAModulePath, PromptAdapterPath
from vllm.executor.ray_distributed_executor import RayDistributedExecutor
from vllm.utils import FlexibleArgumentParser
from vllm.entrypoints.logger import RequestLogger

from pathlib import Path

logger = logging.getLogger("ray.serve")

app = FastAPI()

@serve.deployment(
    num_replicas=8,
    max_ongoing_requests=256,
)
@serve.ingress(app)
class VLLMDeployment:
    def __init__(
        self,
        engine_args: AsyncEngineArgs,
        response_role: str,
        lora_modules: Optional[List[LoRAModulePath]] = None,
        prompt_adapters: Optional[List[PromptAdapterPath]] = None,
        request_logger: Optional[RequestLogger] = None,
        chat_template: Optional[str] = None,
    ):
        engine_args.tokenizer = engine_args.model
        engine_args.distributed_executor_backend = RayDistributedExecutor

        logger.info(f"Starting with engine args: {engine_args}")
        self.openai_serving_chat = None
        self.engine_args = engine_args
        self.response_role = response_role
        self.lora_modules = lora_modules
        self.prompt_adapters = prompt_adapters
        self.request_logger = request_logger
        self.chat_template = chat_template
        self.engine = AsyncLLMEngine.from_engine_args(self.engine_args)

        self.serving_models = None
        if Path(engine_args.model).is_dir():
            self.model_name = Path(self.engine_args.model).name
        else:
            self.model_name = self.engine_args.model

    async def get_serving_models(self):
        if not self.serving_models:
            self.serving_models = OpenAIServingModels(
                engine_client=self.engine,
                model_config=await self.engine.get_model_config(),
                base_model_paths=[
                    BaseModelPath(
                        name=self.model_name, model_path=self.engine_args.model
                    )
                ],
                lora_modules=self.lora_modules,
                prompt_adapters=self.prompt_adapters,
            )
        return self.serving_models

    @app.post("/v1/chat/completions")
    async def create_chat_completion(
        self, request: ChatCompletionRequest, raw_request: Request
    ):
        """OpenAI-compatible HTTP endpoint.

        API reference:
            - https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html
        """
        if not self.openai_serving_chat:
            model_config = await self.engine.get_model_config()
            self.openai_serving_chat = OpenAIServingChat(
                engine_client=self.engine,
                model_config=model_config,
                models=await self.get_serving_models(),
                response_role=self.response_role,
                request_logger=self.request_logger,
                chat_template=self.chat_template,
                chat_template_content_format=None,
            )
        logger.info(f"Request: {request}")
        generator = await self.openai_serving_chat.create_chat_completion(
            request, raw_request
        )
        if isinstance(generator, ErrorResponse):
            return JSONResponse(
                content=generator.model_dump(), status_code=generator.code
            )
        if request.stream:
            return StreamingResponse(content=generator, media_type="text/event-stream")
        else:
            assert isinstance(generator, ChatCompletionResponse)
            return JSONResponse(content=generator.model_dump())


def parse_vllm_args(cli_args: Dict[str, str]):
    """Parses vLLM args based on CLI inputs.

    Currently uses argparse because vLLM doesn't expose Python models for all of the
    config options we want to support.
    """
    arg_parser = FlexibleArgumentParser(
        description="vLLM OpenAI-Compatible RESTful API server."
    )

    parser = make_arg_parser(arg_parser)
    arg_strings = []
    for key, value in cli_args.items():
        arg_strings.extend([f"--{key}", str(value)])
    logger.info(arg_strings)
    parsed_args = parser.parse_args(args=arg_strings)
    return parsed_args


def build_app(cli_args: Dict[str, str]) -> serve.Application:
    """Builds the Serve app based on CLI arguments.

    See https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html#command-line-arguments-for-the-server
    for the complete set of arguments.

    Supported engine arguments: https://docs.vllm.ai/en/latest/models/engine_args.html.
    """  # noqa: E501
    parsed_args = parse_vllm_args(cli_args)
    engine_args = AsyncEngineArgs.from_cli_args(parsed_args)
    engine_args.worker_use_ray = True

    tp = engine_args.tensor_parallel_size
    logger.info(f"Tensor parallelism = {tp}")
    pg_resources = []
    pg_resources.append({"CPU": 1})  # for the deployment replica
    for i in range(tp):
        pg_resources.append({"CPU": 1, "GPU": 1})  # for the vLLM actors
    print(f"{tp=}, {parsed_args=}, {engine_args=}")
    print("-"*100)

    # We use the "STRICT_PACK" strategy below to ensure all vLLM actors are placed on
    # the same Ray node.
    if not ray.is_initialized():
        ray.init()
    available_gpus = ray.available_resources()["GPU"]
    return VLLMDeployment.options(
        num_replicas=available_gpus // tp,
        placement_group_bundles=pg_resources,
        placement_group_strategy="STRICT_PACK",
    ).bind(
        engine_args,
        parsed_args.response_role,
        parsed_args.lora_modules,
        parsed_args.prompt_adapters,
        cli_args.get("request_logger"),
        parsed_args.chat_template,
    )
