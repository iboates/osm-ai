import json
from datetime import datetime

from dotenv import load_dotenv
import sqlalchemy as sa
import geoalchemy2   # NEED THIS OTHERWISE GEOMETRY COLUMNS ARE NOT "SEEN" BY SQLALCHEMY WHEN DESCRIBING TABLES

from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType

import osmai

load_dotenv()


# other things to try
# 1. pre-made list of top X tag values for important tags to help query definition
# 2. post-processing of final query to analyze for performance concerns (potentially with EXPLAIN)
# 3. agent for interacting with nominatim for named places



# NOTE: It seems that 3.5-turbo is not smart enough to fully understand all the prompts.
# Switching to 4o makes it a lot more aware of important rules like respecting desired CRS transformations

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

sample_inputs = open("sample_inputs/task_decomposer.txt").readlines()

for sample_input in sample_inputs:

    decomposer_agent = osmai.agents.create_agent(llm, "decomposer")
    decomposer_agent_executor = AgentExecutor(agent=decomposer_agent, tools=[], verbose=True)
    response = decomposer_agent_executor.invoke({"input": [HumanMessage(content=sample_input)]})
    pass

    feature_set_names, feature_set_relationships = response["output"].split("\n")

    # resolve subqueries
    subqueries = {}
    for feature_set_name in [fsn.strip() for fsn in feature_set_names.split(";")]:

        featureset_prompt_content = open("osmai/prompts/featureset.txt").read()
        featureset_prompt_template = PromptTemplate.from_template(featureset_prompt_content)
        sql_chain = SQLDatabaseChain.from_llm(prompt=featureset_prompt_template, llm=llm, db=db, verbose=True)

        sql_agent_executor = create_sql_agent(
            llm=llm,
            prefix="",
            prompt=featureset_prompt_template,
            suffix="",
            toolkit=SQLDatabaseToolkit(db=db, llm=llm, system_message=featureset_prompt_template),
            verbose=True,
            agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            max_iterations=10,
            agent_executor_kwargs={"handle_parsing_errors": True})
        generated_sql = sql_agent_executor.invoke({"input": feature_set_name})
        subqueries[feature_set_name] = generated_sql["output"]
        pass

    # write query with relationships between subqueries

    final_query_inputs = {
        "subqueries": subqueries,
        "relationships": feature_set_relationships
    }


    final_query_builder_prompt_template = open("osmai/prompts/final_query_builder.txt").read()
    final_query_prompt_template = PromptTemplate.from_template(final_query_builder_prompt_template)
    final_query_sql_chain = SQLDatabaseChain.from_llm(prompt=final_query_prompt_template, llm=llm, db=db, verbose=True)

    final_query_agent_executor = create_sql_agent(
        llm=llm,
        prefix="",
        prompt=final_query_prompt_template,
        suffix="",
        toolkit=SQLDatabaseToolkit(db=db, llm=llm, system_message=final_query_prompt_template),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=10,
        agent_executor_kwargs={"handle_parsing_errors": True})
    generated_sql = final_query_agent_executor.invoke({"input": json.dumps(final_query_inputs)})
    final_query = generated_sql["output"].strip()

    formatted_date = datetime.now().strftime("%Y%m%d%H%M%S")
    with engine.connect() as conn:
        transaction = conn.begin()

        if final_query[:6] == "```sql":
            final_query = final_query[6:]

        if final_query[-3:] == "```":
            final_query = final_query[:-3]

        final_query = final_query.strip()

        if final_query[-1] == ";":
            final_query = final_query[:-1]

        create_query = f"""
        
        create schema if not exists osmai_results;
        drop table if exists osmai_results.osmai_result_{formatted_date};
        create table osmai_results.osmai_result_{formatted_date} as (
            {final_query}
        );
        
        """

        conn.execute(sa.text(create_query))
        transaction.commit()


    break