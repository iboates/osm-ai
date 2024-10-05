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


# MONKEYPATCHES:
#
# 1. Added markdown backtick fix to site-packages/langchain_community/utilities/sql_database.py

llm = ChatOpenAI(model_name='gpt-4o')

STOP_PATTERN = "$$STOP$$"
COUNTRY = "andorra"
COUNTRY_DB_PORT = {"andorra": 5433,
                   "france": 5432}
load_dotenv()
engine = sa.create_engine(f'postgresql+psycopg2://osmai:osmai@localhost:{COUNTRY_DB_PORT[COUNTRY]}/osmai')
table_descriptions = {table.split(".")[0]: open(table).read() for table in glob("prompts/table_descriptions/*.txt")}
db = SQLDatabase(engine,
                 include_tables=[table.split("/")[-1].split(".")[0] for table in glob("prompts/table_descriptions/*.txt")],
                 custom_table_info=table_descriptions)

# decompose task
prompt = PromptTemplate.from_template(open("prompts/task_decomposer.txt").read())
task_decomposer_chain = (
    RunnablePassthrough.assign()
    | prompt
    | llm.bind(stop=[STOP_PATTERN])
    | StrOutputParser()
)
query = """

    What are some hotels near the town of Axiovall?

"""
result = task_decomposer_chain.invoke({"input": query})
steps = re.split(r'\d+\.\s+', result)
steps = [step.strip() for step in steps if step.strip() and not step.startswith("Steps")]

print(result)


BASE_PROMPT = """You are a PostgreSQL expert. Given an input question, first create a syntactically correct PostgreSQL query to run, then look at the results of the query and return the answer to the input question.
Unless the user specifies in the question a specific number of examples to obtain, query for at most {top_k} results using the LIMIT clause as per PostgreSQL. You can order the results to return the most informative data in the database.
Never query for all columns from a table. You must query only the columns that are needed to answer the question. Wrap each column name in double quotes (") to denote them as delimited identifiers.
Pay attention to use only the column names you can see in the tables below. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.
Pay attention to use CURRENT_DATE function to get the current date, if the question involves "today".

You are an expert on geographic information systems and locating features for geographic information system workflows related to an OpenStreetMap database.

You will receive a request from a human which can be answered by retrieving a a feature or group of features based on a location.

You should return an SQL query which properly returns a rowset of features that you are confident matches the description provided in the request.

This SQL will be given to another expert who will use it to build a solution to a larger, more complex problem, so make sure that it can be easily encapsulated in a subquery

Only use the following tables:
{table_info}

Question: {input}

Use the following format:

Question: Question here
SQLQuery: SQL Query to run
SQLResult: Result of the SQLQuery
Answer: Final answer here

"""

# analyze each step individually
for step in steps:

    prompt = PromptTemplate.from_template(BASE_PROMPT)
    sql_chain = SQLDatabaseChain.from_llm(prompt=prompt, llm=llm, db=db, verbose=True)
    agent_executor = create_sql_agent(
        llm=llm,
        # prompt=prompt,
        toolkit=SQLDatabaseToolkit(db=db, llm=llm, system_message=prompt),
        verbose=True,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        max_iterations=5
    )
    agent_executor.invoke({"input": step})
    pass