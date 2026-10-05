import sqlite3

import pytest

from sql_safety import QueryTimeoutError, UnsafeSQLError, readonly_connection, run_safe_query, validate_sql


@pytest.fixture
def conn():
    c = sqlite3.connect(":memory:")
    c.execute("CREATE TABLE data (`Name` TEXT, `Age` INTEGER, `Update Date` TEXT)")
    c.executemany("INSERT INTO data VALUES (?, ?, ?)", [("Ann", 30, "2024-01-01"), ("Bob", 45, "2024-02-01")])
    c.commit()
    yield c
    c.close()


@pytest.mark.parametrize("sql", [
    "DROP TABLE data",
    "SELECT * FROM data; DELETE FROM data",
    "SELECT * FROM data;DROP TABLE data;",
    "ATTACH DATABASE 'evil.db' AS evil",
    "PRAGMA table_info(data)",
    "SELECT * FROM data; PRAGMA writable_schema = 1",
    "DELETE FROM data",
    "UPDATE data SET Age = 0",
    "INSERT INTO data VALUES ('x', 1, '2024-01-01')",
    "CREATE TABLE t (a)",
    "ALTER TABLE data ADD COLUMN x",
    "VACUUM",
    "WITH x AS (SELECT 1) DELETE FROM data",
    "SELECT load_extension('evil.dll')",
    "/* comment */ DROP TABLE data",
    "",
    "   ",
])
def test_unsafe_sql_is_rejected(sql):
    with pytest.raises(UnsafeSQLError):
        validate_sql(sql)


@pytest.mark.parametrize("sql", [
    "SELECT * FROM data",
    "SELECT * FROM data;",
    "select count(*) from data where `Name` = 'Ann'",
    "WITH t AS (SELECT `Age` FROM data) SELECT AVG(`Age`) FROM t",
    # Keywords inside literals / quoted identifiers / comments are not statements.
    "SELECT * FROM data WHERE `Name` = 'drop table; delete'",
    "SELECT `Update Date` FROM data",
    "SELECT * FROM data -- please don't DROP anything",
])
def test_safe_sql_is_accepted(sql):
    assert validate_sql(sql)


def test_trailing_semicolon_is_stripped():
    assert validate_sql("SELECT 1;  ") == "SELECT 1"


def test_run_safe_query_returns_rows(conn):
    df = run_safe_query("SELECT `Name`, `Age` FROM data ORDER BY `Age`", conn)
    assert df.values.tolist() == [["Ann", 30], ["Bob", 45]]


def test_row_cap(conn):
    df = run_safe_query("WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n WHERE i < 50) SELECT i FROM n",
                        conn, max_rows=10)
    assert len(df) == 10 and df.attrs["truncated"] is True


def test_timeout(conn):
    endless = "WITH RECURSIVE n(i) AS (SELECT 1 UNION ALL SELECT i + 1 FROM n) SELECT MAX(i) FROM n"
    with pytest.raises(QueryTimeoutError):
        run_safe_query(endless, conn, timeout_s=0.2)


def test_readonly_connection_blocks_writes_even_without_validation(conn):
    ro = readonly_connection(conn)
    with pytest.raises(sqlite3.Error):
        ro.execute("DELETE FROM data")
    with pytest.raises(sqlite3.Error):
        ro.execute("ATTACH DATABASE ':memory:' AS other")
    with pytest.raises(sqlite3.Error):
        ro.execute("PRAGMA query_only = OFF")
    ro.close()
    assert conn.execute("SELECT COUNT(*) FROM data").fetchone()[0] == 2  # original untouched
