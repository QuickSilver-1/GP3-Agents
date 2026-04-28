
import json
import os
import re
from datetime import datetime
from typing import Annotated, List, TypedDict
from langgraph.prebuilt import ToolNode
import pandas as pd
from pandas import DataFrame
from structlog import BoundLogger
from enum import Enum
import plotly.io as pio

from langchain_core.tools import BaseTool, tool
from langchain_core.messages import AIMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage

from src.weaviate import WeaviateClient, MemoryType, ImportanceLevel
from src.config import Agent, Prompt, Prompts
from src.excel_handler import ExcelHandler, DataSetType

class Node(Enum):
    ANALYZE_DATA = "analyze_data"
    CONCAT_DATASETS = "concat_datasets"
    CLEAR_DATA = "clear_data"
    BUSINESS_CONCLUSION = "business_conclusion"
    CREATE_VISUALIZATION = "create_visualization"
    CREATE_ML_MODEL = "create_ml_model"
    GENERATE_REPORT = "generate_report"
    TOOLS = "tools"

class AgentState(TypedDict):
    messages: Annotated[List, add_messages]
    data: DataFrame
    last_node: Node
    analyze: str
    conclusion: str

class AgentGraph:
    graph: StateGraph
    llm: ChatOpenAI
    tools: List[BaseTool]
    long_term_memory: WeaviateClient
    excel_handler: ExcelHandler
    forbidden_words: List[str]
    data: DataFrame
    raw_datasets: List[DataFrame]
    prompts: Prompts
    __tool_node: ToolNode
    __tool_call_count: int = 0
    __max_tool_rounds: int = 8
    __max_state_messages: int = 10
    artifacts_dir: str
    visualizations_dir: str
    reports_dir: str
    
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
        
        add_memory_desc = """
        Save the memory in the long-term memory vector database with MemoryType and ImportanceLevel.
        Use this tool when need to save information for later use.
        You must save the infromation abount used tecniques, models,
        all used tools with result (successfully/unsuccessfully/test/...) and all made steps.
        class MemoryType(Enum):
            FACT = "fact"
            PREFERENCE = "preference"
            CONTEXT = "context"
            CONVERSATION = "conversation"
            COMMAND = "command"
            RULE = "rule"
            TEMPLATE = "template"

        class ImportanceLevel(Enum):
            TRIVIAL = 1
            LOW = 2
            NORMAL = 3
            MEDIUM = 4
            HIGH = 5
            CRITICAL = 6
        """
        get_memory_desc = """
        Get memories from the long-term memory vectr database by MemoryType, ImportanceLevel, query and limit
        Use this tool when you need to recall information from the using
        class MemoryType(Enum):
            FACT = "fact"
            PREFERENCE = "preference"
            CONTEXT = "context"
            CONVERSATION = "conversation"
            COMMAND = "command"
            RULE = "rule"
            TEMPLATE = "template"

        class ImportanceLevel(Enum):
            TRIVIAL = 1
            LOW = 2
            NORMAL = 3
            MEDIUM = 4
            HIGH = 5
            CRITICAL = 6
        """
        create_new_tool_desc = """
        A tool for creating other tools,
        it receives a one valid Python function as
        a string with a detailed description and,
        using the exec() function, adds a new tool.
        
        AVAILABLE NAMESPACE:
        - np (numpy)
        - pd (pandas)  
        - plt (matplotlib.pyplot)
        - plotly (plotly)
        - plotly_express (plotly.express)
        
        DATASET:
        self.data: pd.DataFrame
        
        LOGGER:
        self.logger: BoundLogger
        """
        save_dataset = """
        Save the current dataset to an Excel file.
        """
        list_raw_datasets_desc = """
        Return metadata and first rows for all loaded raw datasets.
        Use this tool to inspect available source datasets before analysis/merge.
        """
        use_raw_dataset_desc = """
        Select one raw dataset by index and copy it into the working dataset self.data.
        """
        concat_raw_datasets_desc = """
        Concatenate all raw datasets row-wise into self.data.
        keep_common_only=True keeps only intersection of columns.
        """
        get_working_dataset_desc = """
        Return metadata and first rows for current working dataset self.data.
        """
        save_plotly_figure_desc = """
        Save a Plotly figure (as JSON string) into an HTML file in artifacts/visualizations.
        Returns saved file path.
        """
        save_text_file_desc = """
        Save a text content into a file in artifacts directory.
        Use it for report sections or intermediate notes.
        """
        self.model_name = cfg.model
        
        @tool(description=add_memory_desc)
        def _add_memory_wrapper(type: str, importance: int, content: str) -> dict:
            type = MemoryType(type.upper())
            importance = ImportanceLevel(importance)
            return self._add_memory(type, importance, content)
        
        @tool(description=get_memory_desc)
        def _get_memories_wrapper(types: List[str], importances: List[int], query: str, limit: int = 10) -> dict:
            types = [MemoryType(t.upper()) for t in types]
            importances = [ImportanceLevel(i) for i in importances]
            return self._get_memories(types, importances, query, limit)
        
        @tool(description=create_new_tool_desc)
        def _create_new_tool_wrapper(function: str) -> dict:
            return self._create_new_tool(function)
        
        @tool(description=save_dataset)
        def _save_dataset_wrapper() -> dict:
            return self._save_dataset()

        @tool(description=list_raw_datasets_desc)
        def _list_raw_datasets_wrapper(rows: int = 3) -> dict:
            return self._list_raw_datasets(rows)

        @tool(description=use_raw_dataset_desc)
        def _use_raw_dataset_wrapper(index: int = 0) -> dict:
            return self._use_raw_dataset(index)

        @tool(description=concat_raw_datasets_desc)
        def _concat_raw_datasets_wrapper(keep_common_only: bool = True) -> dict:
            return self._concat_raw_datasets(keep_common_only)

        @tool(description=get_working_dataset_desc)
        def _get_working_dataset_info_wrapper(rows: int = 5) -> dict:
            return self._get_working_dataset_info(rows)

        @tool(description=save_plotly_figure_desc)
        def _save_plotly_figure_wrapper(figure_json: str, filename: str) -> dict:
            return self._save_plotly_figure(figure_json, filename)

        @tool(description=save_text_file_desc)
        def _save_text_file_wrapper(content: str, filename: str) -> dict:
            return self._save_text_file(content, filename)
        
        self.tools = [
            _add_memory_wrapper,
            _get_memories_wrapper,
            _create_new_tool_wrapper,
            _save_dataset_wrapper,
            _list_raw_datasets_wrapper,
            _use_raw_dataset_wrapper,
            _concat_raw_datasets_wrapper,
            _get_working_dataset_info_wrapper,
            _save_plotly_figure_wrapper,
            _save_text_file_wrapper,
        ]
        self.__tool_node = ToolNode(self.tools)
        
        self.llm = ChatOpenAI(
            model=cfg.model,
            openai_api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
        )
        if cfg.use_prompt_optimizer:
            prompt_optimizer_llm = ChatOpenAI(
                model=cfg.prompt_optimizer_model,
                openai_api_key=cfg.api_key,
                base_url=cfg.base_url,
                timeout=cfg.timeout,
                max_retries=cfg.max_retries,
            )
            self.prompts = self.create_ai_prompts(prompts, self.tools, prompt_optimizer_llm)
        else:
            self.prompts = prompts
        
        self.llm = self.llm.bind_tools(self.tools)
                
        self.forbidden_words = ["import", "os", "sys", "eval", "exec", "__import__", "open",
                                "file", "compile", "globals", "locals", "__builtins__", "input"]
              
        self.agent = self.__build_agent()
    
    def run(self):
        initial_state: AgentState = {
            "messages": [],
            "data": self.data,
            "last_node": None,
            "analyze": "",
            "conclusion": ""
        }
        final_state = self.agent.invoke(initial_state)
        return final_state

    def _trim_messages(self, messages: List) -> List:
        if len(messages) > self.__max_state_messages:
            messages = messages[-self.__max_state_messages:]

        
        while messages and isinstance(messages[0], ToolMessage):
            messages = messages[1:]

        return messages

    def _get_tool_calls(self, message: AIMessage):
        tool_calls = getattr(message, "tool_calls", None)
        if not tool_calls and hasattr(message, "additional_kwargs"):
            tool_calls = message.additional_kwargs.get("tool_calls")
        return tool_calls or []

    def _invoke_with_tool_loop(self, prompt: str, state_messages: List, node_name: str):
        messages = self._trim_messages(state_messages) + [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        messages.append(response)

        for round_idx in range(self.__max_tool_rounds):
            tool_calls = self._get_tool_calls(response)
            if not tool_calls:
                break

            self.logger.info(f"{node_name}: tool round {round_idx + 1} with {len(tool_calls)} call(s)")
            tools_by_name = {tool.name: tool for tool in self.tools}
            for call in tool_calls:
                tool_name = call["name"]
                tool_args = call.get("args", {})
                tool_id = call["id"]

                tool_obj = tools_by_name.get(tool_name)
                if tool_obj is None:
                    result = {"error": f"Tool '{tool_name}' not found"}
                else:
                    try:
                        result = tool_obj.invoke(tool_args)
                    except Exception as e:
                        result = {"error": f"Tool '{tool_name}' failed: {str(e)}"}

                messages.append(
                    ToolMessage(
                        content=json.dumps(result, ensure_ascii=False),
                        tool_call_id=tool_id,
                        name=tool_name,
                    )
                )

            response = self.llm.invoke(messages)
            messages.append(response)
        else:
            self.logger.warning(f"{node_name}: max tool rounds reached, forcing next node")
            pending_calls = self._get_tool_calls(response)
            for call in pending_calls:
                messages.append(
                    ToolMessage(
                        content=json.dumps(
                            {"warning": "Tool loop stopped: max tool rounds reached"},
                            ensure_ascii=False,
                        ),
                        tool_call_id=call["id"],
                        name=call["name"],
                    )
                )

        return self._trim_messages(messages), response
    
    def _add_memory(self, type: MemoryType, importance: ImportanceLevel, content: str) -> dict:
        """
        Save the memory in the long-term memory vector database with MemoryType and ImportanceLevel.
        Use this tool when need to save information for later use.
        You must save the infromation abount used tecniques, models,
        all used tools with result (successfully/unsuccessfully/test/...) and all made steps.
        class MemoryType(Enum):
            FACT = "fact"
            PREFERENCE = "preference"
            CONTEXT = "context"
            CONVERSATION = "conversation"
            COMMAND = "command"
            RULE = "rule"
            TEMPLATE = "template"

        class ImportanceLevel(Enum):
            TRIVIAL = 1
            LOW = 2
            NORMAL = 3
            MEDIUM = 4
            HIGH = 5
            CRITICAL = 6
        """
        
        self.long_term_memory.create_memory(type, importance, content)
        return {"memory": f"Memory saved (type={type}, importance={importance}, content={content[:50]})"} 
    
    def _get_memories(self, types: List[MemoryType], importances: List[ImportanceLevel], query: str, limit: int = 10) -> dict:
        """
        Get memories from the long-term memory vectr database by MemoryType, ImportanceLevel, query and limit
        Use this tool when you need to recall information from the using
        class MemoryType(Enum):
            FACT = "fact"
            PREFERENCE = "preference"
            CONTEXT = "context"
            CONVERSATION = "conversation"
            COMMAND = "command"
            RULE = "rule"
            TEMPLATE = "template"

        class ImportanceLevel(Enum):
            TRIVIAL = 1
            LOW = 2
            NORMAL = 3
            MEDIUM = 4
            HIGH = 5
            CRITICAL = 6
        """
    
        memories = self.long_term_memory.get_memories(types, importances, query, limit)
        content = "\n".join([f"{m['content']} (type: {m['type']}, importance: {m['importance']})" for m in memories])
        
        return {"memories": content}
    
    def _create_new_tool(self, function: str) -> dict:
        """
        A tool for creating other tools,
        it receives a one valid Python function as
        a string with a detailed description and,
        using the exec() function, adds a new tool.
        
        AVAILABLE NAMESPACE:
        - np (numpy)
        - pd (pandas)  
        - plt (matplotlib.pyplot)
        - plotly (plotly)
        - plotly_express (plotly.express)
        
        DATASET:
        self.data: pd.DataFrame
        
        LOGGER:
        self.logger: BoundLogger
        """
        
        if not self.__func_security_check(function):
            return {"tool": "Security check failed"}
        
        try:
            namespace = {
                "np": __import__("numpy"),
                "pd": __import__("pandas"),
                "plt": __import__("matplotlib.pyplot"),
                "plotly": __import__("plotly"),
                "plotly_express": __import__("plotly.express"),
            }
            exec(function, namespace)
            
            new_func_name:str = None
            for name, data in namespace.items():
                if callable(data) and not name.startswith("_"):
                    new_func_name = name
                    
            if new_func_name:
                new_tool = tool(namespace[new_func_name])
                self.tools.append(new_tool)
                self.llm = self.llm.bind_tools(self.tools)
                
                self.__tool_node.tools = self.tools
                self.__tool_node.tools_by_name = {tool.name: tool for tool in self.tools}
                
                return {"tool": f"Tool '{new_func_name}' created successfully"}
                    
            else:
                return {"tool": "No function found in code"}
            
        except Exception as e:
            return {"tool": f"Error creating tool: {str(e)}"}        
    
    def _save_dataset(self) -> dict:
        """
        Save the current dataset to an Excel file.
        """
        self.excel_handler.save(type=DataSetType.PROCESSED, df=self.data)
        return {"dataset": "Dataset saved"}

    def _list_raw_datasets(self, rows: int = 3) -> dict:
        datasets = []
        for idx, df in enumerate(self.raw_datasets):
            datasets.append(
                {
                    "index": idx,
                    "shape": df.shape,
                    "columns": list(df.columns),
                    "head": df.head(rows).to_dict(orient="records"),
                }
            )
        return {"raw_datasets": datasets, "count": len(datasets)}

    def _use_raw_dataset(self, index: int = 0) -> dict:
        if index < 0 or index >= len(self.raw_datasets):
            return {"error": f"Dataset index {index} is out of range"}

        self.data = self.raw_datasets[index].copy()
        return {"working_dataset_shape": self.data.shape, "index": index}

    def _concat_raw_datasets(self, keep_common_only: bool = True) -> dict:
        if not self.raw_datasets:
            return {"error": "No raw datasets loaded"}

        if keep_common_only:
            common_columns = set(self.raw_datasets[0].columns)
            for df in self.raw_datasets[1:]:
                common_columns &= set(df.columns)
            common_columns = list(common_columns)

            if not common_columns:
                return {"error": "No common columns between raw datasets"}

            self.data = pd.concat([df[common_columns] for df in self.raw_datasets], ignore_index=True)
        else:
            self.data = pd.concat(self.raw_datasets, ignore_index=True, sort=False)

        return {
            "working_dataset_shape": self.data.shape,
            "columns": list(self.data.columns),
        }

    def _get_working_dataset_info(self, rows: int = 5) -> dict:
        if self.data.empty:
            return {"warning": "Working dataset self.data is empty"}

        return {
            "shape": self.data.shape,
            "columns": list(self.data.columns),
            "head": self.data.head(rows).to_dict(orient="records"),
            "dtypes": {k: str(v) for k, v in self.data.dtypes.to_dict().items()},
        }

    def _save_plotly_figure(self, figure_json: str, filename: str) -> dict:
        try:
            fig = pio.from_json(figure_json)
            safe_filename = filename if filename.endswith(".html") else f"{filename}.html"
            path = os.path.join(self.visualizations_dir, safe_filename)
            pio.write_html(fig, file=path, auto_open=False)
            return {"visualization_path": path}
        except Exception as e:
            return {"error": f"Failed to save visualization: {str(e)}"}

    def _save_text_file(self, content: str, filename: str) -> dict:
        safe_filename = filename if filename.endswith(".md") or filename.endswith(".txt") else f"{filename}.md"
        path = os.path.join(self.artifacts_dir, safe_filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        return {"file_path": path}

    def _save_detailed_report(self, state: AgentState, llm_report: str) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.reports_dir, f"final_report_{timestamp}.md")

        lines = [
            "# Final Agent Report",
            "",
            f"Generated at: {datetime.now().isoformat()}",
            "",
            "## Pipeline Summary",
            "- Node 1: analyze_data",
            "- Node 2: concat_datasets",
            "- Node 3: clear_data",
            "- Node 4: business_conclusion",
            "- Node 5: create_visualization",
            "- Node 6: create_ml_model",
            "- Node 7: generate_report",
            "",
            "## Dataset Status",
            f"- Raw datasets loaded: {len(self.raw_datasets)}",
            f"- Final dataset shape: {self.data.shape}",
            f"- Final dataset columns: {list(self.data.columns)}",
            "",
            "## Analysis Output",
            state.get("analyze", ""),
            "",
            "## Business Conclusion",
            state.get("conclusion", ""),
            "",
            "## Model/Report Output",
            llm_report,
            "",
            "## Execution Trace (Recent Messages)",
        ]

        for idx, message in enumerate(state.get("messages", []), start=1):
            msg_type = type(message).__name__
            content = getattr(message, "content", "")
            if isinstance(content, list):
                content = str(content)
            content = str(content).strip().replace("\n", " ")
            lines.append(f"{idx}. [{msg_type}] {content[:600]}")

        lines.extend(
            [
                "",
                "## Produced Artifacts",
                f"- Visualizations directory: `{self.visualizations_dir}`",
                f"- Reports directory: `{self.reports_dir}`",
                "",
                "## Notes",
                "- This report is auto-generated by the agent.",
                "- It includes both LLM outputs and execution metadata.",
            ]
        )

        with open(report_path, "w", encoding="utf-8") as report_file:
            report_file.write("\n".join(lines))

        return report_path
    
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
        
    def __build_agent(self):
        graph = StateGraph(AgentState)

        graph.add_node(Node.ANALYZE_DATA.value, self.analyze_data)
        graph.add_node(Node.CONCAT_DATASETS.value, self.concat_datasets)
        graph.add_node(Node.CLEAR_DATA.value, self.clear_data)
        graph.add_node(Node.BUSINESS_CONCLUSION.value, self.business_conclusion)
        graph.add_node(Node.CREATE_VISUALIZATION.value, self.create_visualization)
        graph.add_node(Node.CREATE_ML_MODEL.value, self.create_ml_model)
        graph.add_node(Node.GENERATE_REPORT.value, self.generate_report)
        
        graph.add_edge(START, Node.ANALYZE_DATA.value)
        graph.add_edge(Node.ANALYZE_DATA.value, Node.CONCAT_DATASETS.value)
        graph.add_edge(Node.CONCAT_DATASETS.value, Node.CLEAR_DATA.value)
        graph.add_edge(Node.CLEAR_DATA.value, Node.BUSINESS_CONCLUSION.value)
        graph.add_edge(Node.BUSINESS_CONCLUSION.value, Node.CREATE_VISUALIZATION.value)
        graph.add_edge(Node.CREATE_VISUALIZATION.value, Node.CREATE_ML_MODEL.value)
        graph.add_edge(Node.CREATE_ML_MODEL.value, Node.GENERATE_REPORT.value)
        graph.add_edge(Node.GENERATE_REPORT.value, END)
        
        return graph.compile()

    def __func_security_check(self, function: str) ->  bool:
        for word in self.forbidden_words:
            if word in function:
                return False
            
        return True
    
    def create_ai_prompts(self, prompts: Prompts, tools: List[BaseTool], llm: ChatOpenAI) -> Prompts:
        self.logger.info("Creating AI prompts")
        
        tools = {tool.name: tool.description for tool in tools}
        
        optimizer_prompt_template = """
        You are an AI prompt engineer for main AI agent with ten years of experience.
        You receive a human prompt and should return a great professional prompt
        with detailed instructions for AI agent to get the best result.
        You should use the best modern techniques and practices of prompt engineering
        
        TOOLS:
        {tools}
        
        Requirements for great prompt:
        1. Role definition for the AI agent
        2. Step-by-step instructions
        3. Output format requirements
        4. Use chain of thought
        5. Examples of good/bad responses
        6. Add critic agent to evaluate the response and give feedback for improvement
        
        VERY IMPORTANT - follow the requirements strictly. Do not miss any of them.:
        - You MUST NOT introduce any new template variables/parameters.
        You must use the variables explicitly provided in the prompt.

        - This prompt is processed using Python str.format().
        Therefore, any literal curly braces that are NOT template variables MUST be escaped.

        - Dont use curly braces for any purpose other than template variables. If you need to use curly braces in the prompt, escape them like this: {{ or }}.
                
        PROMPT - {prompt_name}:
        {prompt_template}
        
        PARAMS:
        {prompt_params}
        
        OUTPUT FORMAT:
        Prompt: ... <- your great prompt here
        """
        
        great_prompts = dict()
        for name, prompt_data in prompts.items():
            try:
                optimizer_prompt = optimizer_prompt_template
                optimizer_prompt = optimizer_prompt.replace("{prompt_name}", name)
                optimizer_prompt = optimizer_prompt.replace("{prompt_template}", prompt_data.template)
                optimizer_prompt = optimizer_prompt.replace("{prompt_params}", ", ".join(prompt_data.params))
                optimizer_prompt = optimizer_prompt.replace("{tools}", "\n".join([f"{tool_name}: {desc}" for tool_name, desc in tools.items()]))

                response = llm.invoke(optimizer_prompt)
                candidate_prompt = response.content or ""
                if "Prompt:" in candidate_prompt:
                    candidate_prompt = candidate_prompt.split("Prompt:", 1)[1].strip()
                candidate_prompt = self.__remove_bad_braces(candidate_prompt, prompt_data.params).strip()

                required_placeholders = [f"{{{param}}}" for param in prompt_data.params]
                has_all_params = all(ph in candidate_prompt for ph in required_placeholders)
                max_prompt_size = max(6000, int(len(prompt_data.template) * 1.5))
                is_too_large = len(candidate_prompt) > max_prompt_size
                if not candidate_prompt or not has_all_params or is_too_large:
                    self.logger.warning(f"Prompt optimizer fallback for '{name}': invalid optimized prompt")
                    great_prompts[name] = Prompt(template=prompt_data.template, params=prompt_data.params)
                    continue

                great_prompts[name] = Prompt(template=candidate_prompt, params=prompt_data.params)
                self.logger.info(f"Great prompt for '{name}' created: {candidate_prompt[:max(500, len(candidate_prompt)-1)]}")
            except Exception as e:
                self.logger.warning(f"Prompt optimizer fallback for '{name}': {str(e)}")
                great_prompts[name] = Prompt(template=prompt_data.template, params=prompt_data.params)

        return great_prompts
     
    def __remove_bad_braces(self, text: str, allowed_params):
        allowed = set(allowed_params)

        def replacer(match):
            content = match.group(1).strip()
            if content in allowed:
                return match.group(0)

            return content

        return re.sub(r"\{([^{}]+)\}", replacer, text)