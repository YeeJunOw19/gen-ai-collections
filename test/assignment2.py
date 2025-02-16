
import yaml
import polars as pl
from pathlib import Path
from src.data_ingestion.mdutils import motherduck_setup
from src.env_vars import MOTHERDUCK_TOKEN

CONFIG = yaml.safe_load(open(Path(__file__).joinpath("..", "config.yaml").resolve(), mode="r"))["Query_Prompt_Engineering_Results"]


def main() -> None:
    query_string = f'SELECT * FROM "{CONFIG["MotherDuck_Schema"]}".{CONFIG["MotherDuck_Table"]} ORDER BY RunDate DESC, ModelAccuracy DESC'

    # Set up MotherDuck Engine and query the data
    md = motherduck_setup.MotherDucking(CONFIG["MotherDuck_Database"], True)
    df = motherduck_setup.md_read_table(
        duck_engine=md.duckdb_engine, md_schema=CONFIG["MotherDuck_Schema"], md_table=CONFIG["MotherDuck_Table"],
        keep_columns=None, custom_query=query_string
    ).collect()

    with pl.Config(fmt_str_lengths=1000, tbl_width_chars=1000, tbl_rows=100):
        print(df)


if __name__ == "__main__":
    main()
