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
sql_chain = SQLDatabaseChain.from_llm(llm=llm, db=db, verbose=True)
agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=5
)

query = "What are some hotels near the town of Aixovall?"

agent_executor.invoke({"input": query})