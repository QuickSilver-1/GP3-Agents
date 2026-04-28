from dotenv import load_dotenv
from typing import Dict
import grpc
from sklearn import base
import yaml
from os import getenv

class Logger:
    level: str
    output_file: str
    
class Excel:
    raw_path: str
    processed_path: str
    raw_sheet_name: str
    processed_sheet_name: str

class Prompt:
    template: str
    params: list[str]
    
    def __init__(self, template: str, params: list[str]):
        self.template = template
        self.params = params

Prompts = Dict[str, Prompt]

class Weaviate:
    host: str
    port: int
    grpc_port: int
    
class Agent:
    name: str
    model: str
    prompt_optimizer_model: str
    use_prompt_optimizer: bool
    api_key: str
    base_url: str
    timeout: int
    max_retries: int
    artifacts_dir: str
    
class Config:
    logger: Logger = Logger()
    excel: Excel = Excel()
    prompts: Prompts = dict()
    weaviate: Weaviate = Weaviate()
    agent: Agent = Agent()
    
    def __init__(self, env_path: str = "./.env", yaml_path: str = "./config.yaml"):
        load_dotenv(env_path)
        self.agent.api_key = getenv("API_KEY")

        with open(yaml_path, "r", encoding="utf-8") as config_file:
            yamlData = yaml.safe_load(config_file)
        self.logger.level = yamlData["logger"]["level"]
        self.logger.output_file = yamlData["logger"]["output_file"]

        self.excel.raw_path = yamlData["excel"]["raw_path"]
        self.excel.processed_path = yamlData["excel"]["processed_path"]
        self.excel.raw_sheet_name = yamlData["excel"]["raw_sheet_name"]
        self.excel.processed_sheet_name = yamlData["excel"]["processed_sheet_name"]
        
        for prompt in yamlData["prompts"]:
            p = Prompt(prompt["template"], prompt["params"])
            self.prompts[prompt["name"]] = p
        
        self.weaviate.host = yamlData["weaviate"]["host"]
        self.weaviate.port = yamlData["weaviate"]["port"]
        self.weaviate.grpc_port = yamlData["weaviate"]["grpc_port"]
        
        self.agent.name = yamlData["agent"]["name"]
        self.agent.model = yamlData["agent"]["model"]
        self.agent.prompt_optimizer_model = yamlData["agent"].get("prompt_optimizer_model", self.agent.model)
        self.agent.use_prompt_optimizer = yamlData["agent"].get("use_prompt_optimizer", True)
        self.agent.base_url = yamlData["agent"]["base_url"]
        self.agent.timeout = yamlData["agent"]["timeout"]
        self.agent.max_retries = yamlData["agent"]["max_retries"]
        self.agent.artifacts_dir = yamlData["agent"].get("artifacts_dir", "artifacts")
        