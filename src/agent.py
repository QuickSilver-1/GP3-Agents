from typing import Annotated, List, TypedDict
import pandas as pd
from pandas import DataFrame
from enum import Enum
import os

from langgraph.graph.message import add_messages

from src.weaviate import WeaviateClient, MemoryType, ImportanceLevel
from src.config import Agent, Prompt, Prompts
from src.excel_handler import ExcelHandler, DataSetType
from structlog import BoundLogger


class Node(Enum):
    ANALYZE_DATA = "analyze_data"
    CLEAR_DATA = "clear_data"
    BUSINESS_CONCLUSION = "business_conclusion"
    CREATE_VISUALIZATION = "create_visualization"
    CREATE_ML_MODEL = "create_ml_model"
    GENERATE_REPORT = "generate_report"
    TOOLS = "tools"

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    data: DataFrame

class AgentGraph:
    
    def __init__(self, cfg: Agent, logger: BoundLogger, prompts: Prompts, excel_handler: ExcelHandler, weaviate_client: WeaviateClient):
        self.logger = logger
        self.excel_handler = excel_handler
        self.long_term_memory = weaviate_client
        self.artifacts_dir = cfg.artifacts_dir
        self.visualizations_dir = os.path.join(self.artifacts_dir, "visualizations")
        self.reports_dir = os.path.join(self.artifacts_dir, "reports")
        os.makedirs(self.visualizations_dir, exist_ok=True)
        os.makedirs(self.reports_dir, exist_ok=True)

        self.raw_datasets = excel_handler.get_df(DataSetType.RAW)
        self.data = pd.DataFrame()
        
    def analyze_data(self, state: AgentState) -> AgentState:
        self.logger.info("ANALYZE_DATA node started")
                
        info = ""
        for dataset in self.raw_datasets:
            info += dataset.head().to_string() + "\n\n"

        prompt = self.prompts["analyze_data"].template.format(Datasets=info, tool_calls=self.__tool_call_count)
        updated_messages, response = self._invoke_with_tool_loop(prompt, state["messages"], Node.ANALYZE_DATA.value)
        self.logger.info(f"RAW RESPONSE: {response}")
        
        self.long_term_memory.create_memory(
            type=MemoryType.FACT,
            importance=ImportanceLevel.HIGH,
            content=f"Data analysis completed: {response.content[:max(500, len(response.content)-1)]}"
        )
        
        self.logger.info("ANALYZE_DATA node completed")
        return {
            "messages": updated_messages,
            "data": state["data"],
            "last_node": Node.ANALYZE_DATA.value,
            "analyze": response.content
        }
   
    def concat_datasets(self, state: AgentState) -> AgentState:
        self.logger.info("CONCAT_DATASETS node started")
        
        prompt = self.prompts["concat_datasets"].template.format(INFO=state["analyze"], tool_calls=self.__tool_call_count)
        updated_messages, response = self._invoke_with_tool_loop(prompt, state["messages"], Node.CONCAT_DATASETS.value)
        self.long_term_memory.create_memory(
            type=MemoryType.FACT,
            importance=ImportanceLevel.HIGH,
            content=f"Datasets concatenated: {len(self.raw_datasets)} files → {self.data.shape}"
        )
    
        self.logger.info(f"CONCAT_DATASETS completed. Shape: {self.data.shape}")
        return {
            "messages": updated_messages,
            "data": self.data,
            "last_node": Node.CONCAT_DATASETS.value,
            "analyze": state["analyze"]
        }                
    
    def clear_data(self, state: AgentState) -> AgentState:
        self.logger.info("CLEAR_DATA node started")
        
        prompt = self.prompts["clear_data"].template.format(INFO=state["analyze"], tool_calls=self.__tool_call_count)
        updated_messages, response = self._invoke_with_tool_loop(prompt, state["messages"], Node.CLEAR_DATA.value)
        self.long_term_memory.create_memory(
            type=MemoryType.FACT,
            importance=ImportanceLevel.HIGH,
            content=f"Dataset cleared: {len(self.raw_datasets)} files → {self.data.shape}"
        )
    
        self.logger.info(f"CLEAR_DATA completed. Shape: {self.data.shape}")
        return {
            "messages": updated_messages,
            "data": self.data,
            "last_node": Node.CLEAR_DATA.value,
            "analyze": state["analyze"]
        }            
    
    def business_conclusion(self, state: AgentState) -> AgentState:
        self.logger.info("BUSINESS_CONCLUSION node started")
        
        print("PROMPT SIZE:", len(self.prompts["business_conclusion"].template.format(INFO=state["analyze"], tool_calls=self.__tool_call_count)))
        prompt = self.prompts["business_conclusion"].template.format(INFO=state["analyze"], tool_calls=self.__tool_call_count)
        updated_messages, response = self._invoke_with_tool_loop(prompt, state["messages"], Node.BUSINESS_CONCLUSION.value)
        self.long_term_memory.create_memory(
            type=MemoryType.FACT,
            importance=ImportanceLevel.HIGH,
            content=f"Business conclusion: {response.content[:max(500, len(response.content)-1)]}"
        )
    
        self.logger.info("BUSINESS_CONCLUSION completed")
        return {
            "messages": updated_messages,
            "data": state["data"],
            "last_node": Node.BUSINESS_CONCLUSION.value,
            "analyze": state["analyze"],
            "conclusion": response.content
        }
    
    def create_visualization(self, state: AgentState) -> AgentState:
        self.logger.info("CREATE_VISUALIZATION node started")
        
        prompt = self.prompts["create_visualization"].template.format(INFO=state["analyze"], CONCLUSION=state["conclusion"], tool_calls=self.__tool_call_count)
        updated_messages, response = self._invoke_with_tool_loop(prompt, state["messages"], Node.CREATE_VISUALIZATION.value)
        self.long_term_memory.create_memory(
            type=MemoryType.FACT,
            importance=ImportanceLevel.HIGH,
            content=f"Visualization created: {response.content[:max(500, len(response.content)-1)]}"
        )
    
        self.logger.info("CREATE_VISUALIZATION completed")
        return {
            "messages": updated_messages,
            "data": state["data"],
            "last_node": Node.CREATE_VISUALIZATION.value,
            "analyze": state["analyze"],
            "conclusion": state["conclusion"]
        }
    
    def create_ml_model(self, state: AgentState) -> AgentState:
        self.logger.info("CREATE_ML_MODEL node started")
        
        prompt = self.prompts["create_ml_model"].template.format(INFO=state["analyze"], CONCLUSION=state["conclusion"], tool_calls=self.__tool_call_count)
        updated_messages, response = self._invoke_with_tool_loop(prompt, state["messages"], Node.CREATE_ML_MODEL.value)
        self.long_term_memory.create_memory(
            type=MemoryType.FACT,
            importance=ImportanceLevel.HIGH,
            content=f"ML model created: {response.content[:max(500, len(response.content)-1)]}"
        )
    
        self.logger.info("CREATE_ML_MODEL completed")
        return {
            "messages": updated_messages,
            "data": state["data"],
            "last_node": Node.CREATE_ML_MODEL.value,
            "analyze": state["analyze"],
            "conclusion": state["conclusion"]
        }
    
    def generate_report(self, state: AgentState) -> AgentState:
        self.logger.info("GENERATE_REPORT node started")
        
        prompt = self.prompts["generate_report"].template.format(INFO=state["analyze"], CONCLUSION=state["conclusion"], tool_calls=self.__tool_call_count)
        updated_messages, response = self._invoke_with_tool_loop(prompt, state["messages"], Node.GENERATE_REPORT.value)
        self.long_term_memory.create_memory(
            type=MemoryType.FACT,
            importance=ImportanceLevel.HIGH,
            content=f"Report generated: {response.content[:max(500, len(response.content)-1)]}"
        )
    
        self.logger.info("GENERATE_REPORT completed")
        self.logger.info(f"Generated report: {response.content}")
        report_path = self._save_detailed_report(state, response.content)
        self.logger.info(f"Detailed report saved: {report_path}")
        
        return {
            "messages": updated_messages,
            "data": state["data"],
            "last_node": Node.GENERATE_REPORT.value,
            "analyze": state["analyze"],
            "conclusion": state["conclusion"]
        }