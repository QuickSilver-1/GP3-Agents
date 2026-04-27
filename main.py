from src import config, logger
from src.excel_handler import ExcelHandler
from src.agent import AgentGraph
from src.weaviate import WeaviateClient

def main() -> None:
    cfg = config.Config()
    
    excel = ExcelHandler(cfg.excel)
    
    log = logger.get_logger(cfg.logger.level, cfg.logger.output_file)
    log.info("Starting the application...")
    
    weaviate = WeaviateClient(cfg.weaviate, log)

    try:
        agent = AgentGraph(cfg.agent, log, cfg.prompts, excel, weaviate)
        agent.run()
    finally:
        weaviate.close()

if __name__ == "__main__":
    main()