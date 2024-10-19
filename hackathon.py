from glob import glob
import json

from dotenv import load_dotenv
import sqlalchemy as sa
import geoalchemy2   # NEED THIS OTHERWISE GEOMETRY COLUMNS ARE NOT "SEEN" BY SQLALCHEMY WHEN DESCRIBING TABLES

from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.prompts import PromptTemplate
from langchain.agents.format_scratchpad.openai_tools import (
    format_to_openai_tool_messages,
)
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain.chains import APIChain
from langchain_openai import OpenAI

from osmai import create_agent

load_dotenv()

CHATGPT_VERSION="gpt-4o"
#CHATGPT_VERSION="gpt-3.5-turbo"
STOP_PATTERN = "$$STOP$$"
COUNTRY = "andorra"
COUNTRY_DB_PORT = {"andorra": 5433,
                   "france": 5432}
engine = sa.create_engine(f'postgresql+psycopg2://osmai:osmai@localhost:{COUNTRY_DB_PORT[COUNTRY]}/osmai')
llm = ChatOpenAI(model=CHATGPT_VERSION, temperature=0)

table_descriptions = {"planet_osm_point": open("osmai/prompts/table_descriptions/planet_osm_point.txt").read(),
                      "planet_osm_line": open("osmai/prompts/table_descriptions/planet_osm_line.txt").read(),
                      "planet_osm_polygon": open("osmai/prompts/table_descriptions/planet_osm_polygon.txt").read()}

db = SQLDatabase(engine,
                 include_tables=["planet_osm_point", "planet_osm_line", "planet_osm_polygon"],
                 custom_table_info=table_descriptions)
prompt = PromptTemplate.from_template(open("osmai/prompts/featureset.txt").read())

# # taginfo API
# chain = APIChain.from_llm_and_api_docs(
#     llm,
#     open("osmai/prompts/taginfo_docs_plaintext.txt").read(),
#     verbose=True,
#     limit_to_domains=["https://taginfo.openstreetmap.org/taginfo/"],
# )
# chain.run(
#     "Get the top 10 values for the 'natural' key"
# )




featureset_prompt_content = open("osmai/prompts/featureset.txt").read()
final_query_builder_prompt_template = open("osmai/prompts/final_query_builder.txt").read()


sample_inputs = open("sample_inputs/task_decomposer.txt").readlines()

for sample_input in sample_inputs:

    # decomposer_agent = create_agent(llm, "decomposer")
    # decomposer_agent_executor = AgentExecutor(agent=decomposer_agent, tools=[], verbose=True)
    # response = decomposer_agent_executor.invoke({"input": [HumanMessage(content=sample_input)]})
    # pass
    #
    # feature_set_names, feature_set_relationships = response["output"].split("\n")
    #
    # # resolve subqueries
    # subqueries = {}
    # for feature_set_name in [fsn.strip() for fsn in feature_set_names.split(";")]:
    #     featureset_prompt_template = PromptTemplate.from_template(featureset_prompt_content)
    #     sql_chain = SQLDatabaseChain.from_llm(prompt=featureset_prompt_template, llm=llm, db=db, verbose=True)
    #     sql_agent_executor = create_sql_agent(
    #         llm=llm,
    #         prefix="",
    #         prompt=featureset_prompt_template,
    #         suffix="",
    #         toolkit=SQLDatabaseToolkit(db=db, llm=llm, system_message=featureset_prompt_template),
    #         verbose=True,
    #         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #         max_iterations=5,
    #         agent_executor_kwargs={"handle_parsing_errors": True})
    #     generated_sql = sql_agent_executor.invoke({"input": feature_set_name})
    #     subqueries[feature_set_name] = generated_sql["output"]
    #
    # # write query with relationships between subqueries
    #
    # final_query_inputs = {
    #     "subqueries": subqueries,
    #     "relationships": feature_set_relationships
    # }

    final_query_inputs = {
        "subqueries": {
            "farmland parcels": """SELECT "osm_id", "landuse", "way" FROM planet_osm_polygon WHERE "landuse" = 'farmland';""",
            "major power lines": """SELECT "osm_id", "power", "way" FROM planet_osm_line WHERE "power" = 'line';"""
        },
        "relationships": ["farmland parcels is within 5000 meters of major power lines"]
    }

    final_query_agent = create_agent(llm, "final_query_builder")
    final_query_prompt_template = PromptTemplate.from_template(final_query_builder_prompt_template)
    final_query_agent_executor = create_sql_agent(
        llm=llm,
        prefix="",
        prompt=final_query_prompt_template,
        suffix="",
        toolkit=SQLDatabaseToolkit(db=db, llm=llm, system_message=final_query_prompt_template),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=5,
        agent_executor_kwargs={"handle_parsing_errors": True})
    generated_sql = final_query_agent_executor.invoke({"input": json.dumps(final_query_inputs)})

    break