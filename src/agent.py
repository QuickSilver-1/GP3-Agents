
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
import matplotlib.pyplot as plt

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

NODE_PIPELINE: List[tuple[int, str]] = [
    (1, Node.ANALYZE_DATA.value),
    (2, Node.CONCAT_DATASETS.value),
    (3, Node.CLEAR_DATA.value),
    (4, Node.BUSINESS_CONCLUSION.value),
    (5, Node.CREATE_VISUALIZATION.value),
    (6, Node.CREATE_ML_MODEL.value),
    (7, Node.GENERATE_REPORT.value),
]

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
    stages_dir: str
    visualizations_dir: str
    reports_dir: str
    
    def __init__(self, cfg: Agent, logger: BoundLogger, prompts: Prompts, excel_handler: ExcelHandler, weaviate_client: WeaviateClient):
        self.logger = logger
        self.excel_handler = excel_handler
        self.long_term_memory = weaviate_client
        self.artifacts_dir = cfg.artifacts_dir
        self.stages_dir = os.path.join(self.artifacts_dir, "stages")
        self.visualizations_dir = os.path.join(self.artifacts_dir, "visualizations")
        self.reports_dir = os.path.join(self.artifacts_dir, "reports")
        os.makedirs(self.stages_dir, exist_ok=True)
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
        Save a Plotly figure (as JSON string) as a PNG image under artifacts/visualizations/.
        Requires figure JSON from fig.to_json() (Plotly). Filename should end with .png
        (if you pass another extension, it will be saved as .png).
        You MUST produce multiple distinct charts for this pipeline step (at least 5 different
        filenames), each tied to a concrete insight from the business conclusion and the dataset.
        Example names: orders_by_status.png, revenue_top_categories.png.
        Returns saved file path per call.
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

        self._plain_llm = ChatOpenAI(
            model=cfg.model,
            openai_api_key=cfg.api_key,
            base_url=cfg.base_url,
            timeout=cfg.timeout,
            max_retries=cfg.max_retries,
            temperature=0.35,
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

    def _invoke_with_tool_loop(self, prompt: str, state_messages: List, node_name: str, max_tool_rounds: int | None = None):
        cap = self.__max_tool_rounds if max_tool_rounds is None else max_tool_rounds
        messages = self._trim_messages(state_messages) + [HumanMessage(content=prompt)]
        response = self.llm.invoke(messages)
        messages.append(response)

        for round_idx in range(cap):
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
            self.logger.warning(f"{node_name}: max tool rounds reached ({cap}), forcing next node")
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
            
            base, _, ext = filename.rpartition(".")
            if ext.lower() in ("png", "jpg", "jpeg", "webp", "svg", "pdf"):
                safe_filename = filename
                stem = base if base else "chart"
            
            else:
                stem = base if base else filename.replace(".", "_")
                safe_filename = f"{stem}.png"
            
            path = os.path.join(self.visualizations_dir, safe_filename)
            fmt = safe_filename.rsplit(".", 1)[-1].lower()
            if fmt not in ("png", "jpg", "jpeg", "webp", "svg", "pdf"):
                fmt = "png"
                safe_filename = f"{stem}.png"
                path = os.path.join(self.visualizations_dir, safe_filename)
            
            pio.write_image(fig, path, format=fmt, width=1100, height=650, scale=1)
            return {"visualization_path": path}
        
        
        except Exception as e:
            return {
                "error": (
                    f"Failed to sve Plotly chart as PNG: {str(e)}. "
                    "Install Kaleido: pip install 'kaleido>=1.0' (or plotly[kaleido])."
                )
            }

    def _save_text_file(self, content: str, filename: str) -> dict:
        safe_filename = filename if filename.endswith(".md") or filename.endswith(".txt") else f"{filename}.md"
        path = os.path.join(self.artifacts_dir, safe_filename)
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as file:
            file.write(content)
        
        return {"file_path": path}

    def _text_preview(self, text: str, limit: int = 8000) -> str:
        if not text:
            return ""
        
        text = str(text).strip()
        
        if len(text) <= limit:
            return text
        return text[:limit] + "\n\n…(truncated)…"

    def _save_node_stage_result(self, step_index: int, node_key: str, llm_text: str, extra_sections: str | None = None) -> str:
        path = os.path.join(self.stages_dir, f"node_{step_index}_result.md")
        raw_shapes = ", ".join([f"{i}:{tuple(df.shape)}" for i, df in enumerate(self.raw_datasets)])
        head_md = ""
        
        if self.data is not None and not self.data.empty:
            head_md = "```\n" + self.data.head(8).to_string() + "\n```"
        
        lines = [
            f"# Node {step_index}: `{node_key}`",
            "",
            f"**Completed at:** {datetime.now().isoformat()}",
            "",
            "## Model output (this step)",
            self._text_preview(llm_text or "(empty)", 12000),
            "",
            "## Data snapshot",
            f"- Raw datasets (index:shape): {raw_shapes or 'none'}",
            f"- Working `self.data` shape: `{getattr(self.data, 'shape', None)}`",
            "",
        ]
        
        if head_md:
            lines.extend(["### Sample rows (working dataset)", "", head_md, ""])
        if extra_sections:
            lines.extend(["## Additional artifacts", "", extra_sections.strip(), ""])
        
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        self.logger.info(f"Stage artifact written: {path}")
                
        return path

    def _save_mpl_figure(self, basename: str) -> str:
        os.makedirs(self.visualizations_dir, exist_ok=True)
        stem = basename[:-4] if basename.lower().endswith(".png") else basename
        
        safe = f"{stem}.png"
        path = os.path.join(self.visualizations_dir, safe)
        
        plt.savefig(path, dpi=120, bbox_inches="tight")
        plt.close()
        return path

    def _append_auto_business_visualizations(self, conclusion: str) -> str:
        df = self.data
        if df is None or df.empty:
            return "(no auto charts: working dataset empty)"

        lines: List[str] = []
        prefix = "node5_auto"
        subtitle = self._text_preview(conclusion, 200).replace("\n", " ")

        def _numeric_cols(limit: int = 4):
            out = []
            
            for c in df.columns:
                if pd.api.types.is_numeric_dtype(df[c]) and df[c].notna().any():
                    out.append(c)
            return out[:limit]

        def _cat_cols(limit: int = 2):
            out = []
            for c in df.columns:
                s = df[c]
                is_cat = (
                    pd.api.types.is_object_dtype(s)
                    or pd.api.types.is_categorical_dtype(s)
                    or pd.api.types.is_string_dtype(s)
                )
                if not is_cat:
                    continue
                nu = s.nunique(dropna=True)
                if 2 <= nu <= 24:
                    out.append(c)
            return out[:limit]

        try:
            miss = df.isnull().sum().sort_values(ascending=False).head(25)
            miss = miss[miss > 0]
            
            if not miss.empty:
                plt.figure(figsize=(10, 5))
                plt.bar(miss.index.astype(str), miss.values, color="steelblue")
                plt.xticks(rotation=45, ha="right")
                plt.ylabel("missing_count")
                plt.title(f"Missing values by column\n{subtitle}" if subtitle else "Missing values by column")
                plt.tight_layout()
                lines.append(self._save_mpl_figure(f"{prefix}_01_missing_values.png"))
        except Exception as e:
            lines.append(f"(skip missingness chart: {e})")

        nums = _numeric_cols(4)
        cats = _cat_cols(2)

        if nums:
            try:
                plt.figure(figsize=(10, 5))
                plt.hist(df[nums[0]].dropna(), bins=40, color="teal", edgecolor="white")
                plt.xlabel(nums[0])
                plt.ylabel("count")
                plt.title(f"Distribution: {nums[0]}\n{subtitle}" if subtitle else f"Distribution: {nums[0]}")
                plt.tight_layout()
                lines.append(self._save_mpl_figure(f"{prefix}_02_hist_{nums[0]}.png"))
            except Exception as e:
                lines.append(f"(skip histogram: {e})")

        if cats:
            try:
                vc = df[cats[0]].astype(str).value_counts().head(15)
                plt.figure(figsize=(10, 5))
                plt.bar(vc.index.astype(str), vc.values, color="darkorange")
                plt.xticks(rotation=45, ha="right")
                plt.ylabel("count")
                plt.title(f"Top categories: {cats[0]}\n{subtitle}" if subtitle else f"Top categories: {cats[0]}")
                plt.tight_layout()
                lines.append(self._save_mpl_figure(f"{prefix}_03_bar_{cats[0]}.png"))
            except Exception as e:
                lines.append(f"(skip bar chart: {e})")

        if len(nums) >= 2:
            try:
                sub = df[[nums[0], nums[1]]].dropna()
                if len(sub) > 1:
                    plt.figure(figsize=(8, 6))
                    plt.scatter(sub[nums[0]], sub[nums[1]], alpha=0.35, s=8)
                    plt.xlabel(nums[0])
                    plt.ylabel(nums[1])
                    plt.title(
                        f"Scatter: {nums[0]} vs {nums[1]}\n{subtitle}" if subtitle else f"Scatter: {nums[0]} vs {nums[1]}"
                    )
                    plt.tight_layout()
                    lines.append(self._save_mpl_figure(f"{prefix}_04_scatter_{nums[0]}_{nums[1]}.png"))
            except Exception as e:
                lines.append(f"(skip scatter: {e})")

        if nums and cats:
            try:
                fig, ax = plt.subplots(figsize=(10, 5))
                df.boxplot(column=nums[0], by=cats[0], ax=ax, rot=45)
                fig.suptitle("")
                
                ax.set_title(f"{nums[0]} by {cats[0]}\n{subtitle}" if subtitle else f"{nums[0]} by {cats[0]}")
                ax.set_xlabel(cats[0])
                ax.set_ylabel(nums[0])
                fig.tight_layout()
                
                os.makedirs(self.visualizations_dir, exist_ok=True)
                path = os.path.join(self.visualizations_dir, f"{prefix}_05_box_{nums[0]}_by_{cats[0]}.png")
                fig.savefig(path, dpi=120, bbox_inches="tight")
                plt.close(fig)
                lines.append(path)
           
            except Exception as e:
                lines.append(f"(skip box: {e})")

        if len(nums) >= 3:
            try:
                sub = df[nums[: min(8, len(nums))]].dropna()
                if sub.shape[1] >= 2 and len(sub) > 20:
                    corr = sub.corr(numeric_only=True)
                    plt.figure(figsize=(8, 6))
                    plt.imshow(corr.values, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.xticks(range(len(corr.columns)), corr.columns, rotation=45, ha="right")
                    plt.yticks(range(len(corr.index)), corr.index)
                    plt.title(
                        f"Correlation heatmap\n{subtitle}" if subtitle else "Correlation heatmap"
                    )
                    plt.tight_layout()
                    lines.append(self._save_mpl_figure(f"{prefix}_06_correlation.png"))
            except Exception as e:
                lines.append(f"(skip heatmap: {e})")

        return "\n".join(p if p.startswith("(") else f"- `{p}`" for p in lines)

    def _build_global_synthesis(self, analyze: str, conclusion: str, node_report: str) -> str:
        prompt = f"""You are a senior strategy advisor for an e-commerce / digital operations leadership team.
        Using ONLY the material below, produce a GLOBAL synthesis in Markdown.

        Required sections (use these headings exactly):
        ## Strategic picture
        ## Market & customer dynamics (inferred)
        ## Operational levers
        ## Risks & unknowns
        ## Next 90 days — priorities
        ## KPIs to monitor
        ## Honest limitations of this analysis

        Rules:
        - Integrate across data understanding, business hypotheses, and ML results; avoid copy-pasting earlier bullets verbatim.
        - Be substantive but readable (about 900–1500 words).
        - If the inputs are mostly Russian, write in Russian; otherwise English.

        ### Data / analysis notes
        {self._text_preview(analyze or "", 6000)}

        ### Business layer
        {self._text_preview(conclusion or "", 4000)}

        ### Model / technical layer
        {self._text_preview(node_report or "", 4000)}
        """
        
        try:
            out = self._plain_llm.invoke([HumanMessage(content=prompt)])
            return (out.content or "").strip()
        except Exception as e:
            return f"(global synthesis unavailable: {e})"

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
                f"- Per-node stage files: `{self.stages_dir}` / `node_1_result.md` … `node_7_result.md`",
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
        self._save_node_stage_result(1, Node.ANALYZE_DATA.value, response.content or "", None)
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
        self._save_node_stage_result(2, Node.CONCAT_DATASETS.value, response.content or "", None)
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
        self._save_node_stage_result(3, Node.CLEAR_DATA.value, response.content or "", None)
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
        self._save_node_stage_result(4, Node.BUSINESS_CONCLUSION.value, response.content or "", None)
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
        updated_messages, response = self._invoke_with_tool_loop(
            prompt,
            state["messages"],
            Node.CREATE_VISUALIZATION.value,
            max_tool_rounds=14,
        )
        auto_viz_list = self._append_auto_business_visualizations(state.get("conclusion") or "")
        self.long_term_memory.create_memory(
            type=MemoryType.FACT,
            importance=ImportanceLevel.HIGH,
            content=f"Visualization created: {response.content[:max(500, len(response.content)-1)]}"
        )
    
        self.logger.info("CREATE_VISUALIZATION completed")
        viz_notes = (
            "### Auto-generated charts (baseline pack; LLM may add more PNG files under visualizations/)\n\n"
            + auto_viz_list
        )
        self._save_node_stage_result(5, Node.CREATE_VISUALIZATION.value, response.content or "", viz_notes)
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
        self._save_node_stage_result(6, Node.CREATE_ML_MODEL.value, response.content or "", None)
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
        global_md = self._build_global_synthesis(
            state.get("analyze") or "",
            state.get("conclusion") or "",
            response.content or "",
        )
        full_report_body = (
            (response.content or "").strip()
            + "\n\n---\n\n## Синтез для руководства (глобальный вывод)\n\n"
            + global_md
        )
        self.long_term_memory.create_memory(
            type=MemoryType.FACT,
            importance=ImportanceLevel.HIGH,
            content=f"Report generated: {response.content[:max(500, len(response.content)-1)]}"
        )
    
        self.logger.info("GENERATE_REPORT completed")
        self.logger.info(f"Generated report: {response.content}")
        report_path = self._save_detailed_report(state, full_report_body)
        self.logger.info(f"Detailed report saved: {report_path}")
        self._save_node_stage_result(
            7,
            Node.GENERATE_REPORT.value,
            full_report_body,
            f"Aggregated markdown report: `{report_path}`",
        )
        
        agent_dict = self.agent.get_graph().to_json()
        with open("artifacts/graph.json", "w", encoding="utf-8") as f:
            json.dump(agent_dict, f, indent=2, default=str)
        
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