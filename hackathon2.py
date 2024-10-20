from dotenv import load_dotenv
import sqlalchemy as sa

from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate

import osmai

load_dotenv()

#CHATGPT_VERSION="gpt-4o"
CHATGPT_VERSION="gpt-3.5-turbo"
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

chain = create_sql_query_chain(llm=llm, db=db, prompt=prompt, k=5)
response = chain.invoke({"input": "grocery stores", "question": ""})

pass