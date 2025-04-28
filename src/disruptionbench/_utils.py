from pathlib import Path


def get_query_folder() -> Path:
    cf = Path(__file__).parent.absolute()
    qdir = cf / 'queries'
    return qdir


def get_query_from_sql_file(
        file_name: str,
        kwargs: dict = None
) -> str:
    with open(get_query_folder() / file_name, 'r') as f:
        query = f.read()
    # injecting kwargs if not None
    if kwargs is not None:
        query = query.format(**kwargs)
    return query
