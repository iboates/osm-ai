from glob import glob
import re

import sqlalchemy as sa
import geoalchemy2  # NEED THIS OTHERWISE GEOMETRY COLUMNS ARE NOT "SEEN" BY SQLALCHEMY WHEN DESCRIBING TABLES
from dotenv import load_dotenv
from langchain_community.agent_toolkits.sql.base import create_sql_agent
from langchain_community.agent_toolkits.sql.toolkit import SQLDatabaseToolkit
from langchain.agents.agent_types import AgentType
from langchain_community.utilities import SQLDatabase
from langchain_experimental.sql import SQLDatabaseChain
from langchain.chat_models import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough


load_dotenv()

# MONKEYPATCHES:
#
# 1. Added markdown backtick fix to site-packages/langchain_community/utilities/sql_database.py

llm = ChatOpenAI(model_name='gpt-4o')

STOP_PATTERN = "$$STOP$$"
COUNTRY = "andorra"
COUNTRY_DB_PORT = {"andorra": 5433,
                   "france": 5432}
load_dotenv()

# decompose task
step_prompt = PromptTemplate.from_template(open("old/prompts/task_decomposer.txt").read())
task_decomposer_chain = (
    RunnablePassthrough.assign()
    | step_prompt
    | llm.bind(stop=[STOP_PATTERN])
    | StrOutputParser()
)
query = """

    What are some hotels near the town of Aixovall?

"""
# result = task_decomposer_chain.invoke({"input": query})
# print(result)
# steps = re.split(r'\d+\.\s+', result)
# steps = [step.strip() for step in steps if step.strip() and not step.startswith("Steps")]

steps = [
    "Identify the location of the town of Aixovall in the OpenStreetMap database, potentially using place identifiers or administrative boundaries.",
    "Retrieve the set of features classified as \"hotels\" within the OpenStreetMap database.",
    "Determine the spatial relationship by finding hotels that are within a certain distance (e.g., 5 kilometers) from the boundary of Aixovall."
]

step_prompt_template = """You are a PostgreSQL and PostGIS expert. Given an input question, first create a syntactically correct PostgreSQL/POSTGIS query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".

When looking for places based on a name You should:

* Look in the "planet_osm_point" and "planet_osm_polygon" table
* It is extremely important that you use "LIKE" or "ILIKE" with wildcards instead of the equality operator
* If multiple places are found, narrow down the search to return what seems to be the closest match

The SQL query should return:

* The osm_id
* Any relevant identifying information that was used (e.g. name)
* The geometry

Only use the following tables:
{table_info}

Only use the below tools. Only use the information returned by the below tools to construct your final answer.
You MUST double check your query before executing it. If you get an error while executing a query, rewrite the query and try again.

{tools}

Use the following format. Always include the but before the colon character, even when there is no corresponding value:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: The last working SQL query that was used to successfully return the result. Do not return anything except the SQL code. Do not explain the SQL code. Do not wrap the code in markdown-style formatting. Remove all LIMIT clauses from the SQL code which were only there to constrain the result set for the sake of brevity.

Begin!

Question: {input}
Thought: I should look at the tables in the database to see what I can query.  Then I should query the schema of the most relevant tables.
{agent_scratchpad}
"""

engine = sa.create_engine(f'postgresql+psycopg2://osmai:osmai@localhost:{COUNTRY_DB_PORT[COUNTRY]}/osmai')
table_descriptions = {table.split(".")[0]: open(table).read() for table in glob("old/prompts/table_descriptions/*.txt")}
db = SQLDatabase(engine,
                 include_tables=[table.split("/")[-1].split(".")[0] for table in glob(
                     "old/prompts/table_descriptions/*.txt")],
                 custom_table_info=table_descriptions)

# analyze each step individually
for step in steps:

    step_prompt = PromptTemplate.from_template(step_prompt_template)
    sql_chain = SQLDatabaseChain.from_llm(prompt=step_prompt, llm=llm, db=db, verbose=True)
    agent_executor = create_sql_agent(
        llm=llm,
        prefix="",
        prompt=step_prompt,
        suffix="",
        toolkit=SQLDatabaseToolkit(db=db, llm=llm, system_message=step_prompt),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=5,
        agent_executor_kwargs={"handle_parsing_errors": True},
        # format_instructions=format_instructions
    )
    generated_sql = agent_executor.invoke({"input": step})
    print(generated_sql["output"])
    print()