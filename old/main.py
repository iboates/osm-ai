from glob import glob

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

# MONKEYPATCHES:
#
# 1. Added markdown backtick fix to site-packages/langchain_community/utilities/sql_database.py

COUNTRY = "andorra"

COUNTRY_DB_PORT = {"andorra": 5433,
                   "france": 5432}

load_dotenv()

engine = sa.create_engine(f'postgresql+psycopg2://osmai:osmai@localhost:{COUNTRY_DB_PORT[COUNTRY]}/osmai')

llm = ChatOpenAI(model_name='gpt-4o')

table_descriptions = {table.split(".")[0]: open(table).read() for table in glob("old/prompts/table_descriptions/*.txt")}
db = SQLDatabase(engine,
                 include_tables=[table.split("/")[-1].split(".")[0] for table in glob(
                     "prompts/table_descriptions/*.txt")],
                 custom_table_info=table_descriptions)

prompt = PromptTemplate.from_template(open("old/prompts/locale_descriptor.txt").read())

sql_chain = SQLDatabaseChain.from_llm(prompt=prompt, llm=llm, db=db, verbose=True)

agent_executor = create_sql_agent(
    llm=llm,
    toolkit=SQLDatabaseToolkit(db=db, llm=llm, system_message=prompt),
    verbose=True,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    max_iterations=30
)



# query = """

# How many schools are there? Tell me their names.

# """

#query = """
#
#What is the largest school by area, and what is the nearest hospital to it?
#
#"""

# query = """

# Tell me about other businesses which are located within 500 meters of the Danone Research Centre Daniel Carasso.

# """

query = """

    Tell me about the amenities in the town of Aixovall

"""

agent_executor.invoke(query)