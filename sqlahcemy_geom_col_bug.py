import sqlalchemy as sa
from sqlalchemy.schema import CreateTable, Table, MetaData
from geoalchemy2 import Geometry

engine = sa.create_engine(f'postgresql+psycopg2://osmai:osmai@localhost:5433/osmai')

metadata = MetaData()
metadata.reflect(bind=engine, schema="public")

for table in metadata.sorted_tables:
    create_table = CreateTable(table).compile(engine)
    pass