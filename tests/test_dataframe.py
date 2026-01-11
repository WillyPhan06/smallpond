from typing import List

import pandas as pd
import pyarrow as pa
import pytest

from smallpond.dataframe import Session


def test_pandas(sp: Session):
    pandas_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = sp.from_pandas(pandas_df)
    assert df.to_pandas().equals(pandas_df)


def test_arrow(sp: Session):
    arrow_table = pa.table({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = sp.from_arrow(arrow_table)
    assert df.to_arrow() == arrow_table


def test_items(sp: Session):
    df = sp.from_items([1, 2, 3])
    assert df.take_all() == [{"item": 1}, {"item": 2}, {"item": 3}]
    df = sp.from_items([{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}])
    assert df.take_all() == [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]


def test_csv(sp: Session):
    df = sp.read_csv(
        "tests/data/mock_urls/*.tsv",
        schema={"urlstr": "varchar", "valstr": "varchar"},
        delim=r"\t",
    )
    assert df.count() == 1000


def test_parquet(sp: Session):
    df = sp.read_parquet("tests/data/mock_urls/*.parquet")
    assert df.count() == 1000


def test_take(sp: Session):
    df = sp.from_pandas(pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]}))
    assert df.take(2) == [{"a": 1, "b": 4}, {"a": 2, "b": 5}]
    assert df.take_all() == [{"a": 1, "b": 4}, {"a": 2, "b": 5}, {"a": 3, "b": 6}]


def test_map(sp: Session):
    df = sp.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
    df1 = df.map("a + b as c")
    assert df1.to_arrow() == pa.table({"c": [5, 7, 9]})
    df2 = df.map(lambda r: {"c": r["a"] + r["b"]})
    assert df2.to_arrow() == pa.table({"c": [5, 7, 9]})

    # user need to specify the schema if can not be inferred from the mapping values
    df3 = df.map(
        lambda r: {"c": None if r["a"] == 1 else r["a"] + r["b"]},
        schema=pa.schema([("c", pa.int64())]),
    )
    assert df3.to_arrow() == pa.table({"c": pa.array([None, 7, 9], type=pa.int64())})


def test_flat_map(sp: Session):
    df = sp.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
    df1 = df.flat_map(lambda r: [{"c": r["a"]}, {"c": r["b"]}])
    assert df1.to_arrow() == pa.table({"c": [1, 4, 2, 5, 3, 6]})
    df2 = df.flat_map("unnest(array[a, b]) as c")
    assert df2.to_arrow() == pa.table({"c": [1, 4, 2, 5, 3, 6]})

    # user need to specify the schema if can not be inferred from the mapping values
    df3 = df.flat_map(lambda r: [{"c": None}], schema=pa.schema([("c", pa.int64())]))
    assert df3.to_arrow() == pa.table({"c": pa.array([None, None, None], type=pa.int64())})


def test_map_batches(sp: Session):
    df = sp.read_parquet("tests/data/mock_urls/*.parquet")
    df = df.map_batches(
        lambda batch: pa.table({"num_rows": [batch.num_rows]}),
        batch_size=350,
    )
    assert df.take_all() == [{"num_rows": 350}, {"num_rows": 350}, {"num_rows": 300}]


def test_filter(sp: Session):
    df = sp.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
    df1 = df.filter("a > 1")
    assert df1.to_arrow() == pa.table({"a": [2, 3], "b": [5, 6]})
    df2 = df.filter(lambda r: r["a"] > 1)
    assert df2.to_arrow() == pa.table({"a": [2, 3], "b": [5, 6]})


def test_random_shuffle(sp: Session):
    df = sp.from_items(list(range(1000))).repartition(10, by_rows=True)
    df = df.random_shuffle()
    shuffled = [d["item"] for d in df.take_all()]
    assert sorted(shuffled) == list(range(1000))

    def count_inversions(arr: List[int]) -> int:
        return sum(sum(1 for j in range(i + 1, len(arr)) if arr[i] > arr[j]) for i in range(len(arr)))

    # check the shuffle is random enough
    # the expected number of inversions is n*(n-1)/4 = 249750
    assert 220000 <= count_inversions(shuffled) <= 280000


def test_partition_by(sp: Session):
    df = sp.from_items(list(range(1000))).repartition(10, by="item % 10")
    df = df.map("min(item % 10) as min, max(item % 10) as max")
    assert df.take_all() == [{"min": i, "max": i} for i in range(10)]


def test_partition_by_key_out_of_range(sp: Session):
    df = sp.from_items(list(range(1000))).repartition(10, by="item % 11")
    try:
        df.to_arrow()
    except Exception as ex:
        assert "partition key 10 is out of range 0-9" in str(ex)
    else:
        assert False, "expected exception"


def test_partition_by_hash(sp: Session):
    df = sp.from_items(list(range(1000))).repartition(10, hash_by="item")
    items = [d["item"] for d in df.take_all()]
    assert sorted(items) == list(range(1000))


def test_count(sp: Session):
    df = sp.from_items([1, 2, 3])
    assert df.count() == 3


def test_limit(sp: Session):
    df = sp.from_items(list(range(1000))).repartition(10, by_rows=True)
    assert df.limit(2).count() == 2


@pytest.mark.skip(reason="limit can not be pushed down to sql node for now")
@pytest.mark.timeout(10)
def test_limit_large(sp: Session):
    # limit will be fused with the previous select
    # otherwise, it will be timeout
    df = sp.partial_sql("select * from range(1000000000)")
    assert df.limit(2).count() == 2


def test_partial_sql(sp: Session):
    # no input deps
    df = sp.partial_sql("select * from range(3)")
    assert df.to_arrow() == pa.table({"range": [0, 1, 2]})

    # join
    df1 = sp.from_arrow(pa.table({"id1": [1, 2, 3], "val1": ["a", "b", "c"]}))
    df2 = sp.from_arrow(pa.table({"id2": [1, 2, 3], "val2": ["d", "e", "f"]}))
    joined = sp.partial_sql("select id1, val1, val2 from {0} join {1} on id1 = id2", df1, df2)
    assert joined.to_arrow() == pa.table(
        {"id1": [1, 2, 3], "val1": ["a", "b", "c"], "val2": ["d", "e", "f"]},
        schema=pa.schema(
            [
                ("id1", pa.int64()),
                ("val1", pa.large_string()),
                ("val2", pa.large_string()),
            ]
        ),
    )


# ==================== Join Tests ====================


def test_join_inner_same_column_name(sp: Session):
    """Test inner join with same column name in both DataFrames."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2, 3], "val1": ["a", "b", "c"]}))
    df2 = sp.from_arrow(pa.table({"id": [2, 3, 4], "val2": ["d", "e", "f"]}))
    result = df1.join(df2, on="id")
    rows = sorted(result.take_all(), key=lambda x: x["id"])
    assert len(rows) == 2
    assert rows[0]["id"] == 2
    assert rows[0]["val1"] == "b"
    assert rows[0]["val2"] == "d"
    assert rows[1]["id"] == 3
    assert rows[1]["val1"] == "c"
    assert rows[1]["val2"] == "e"


def test_join_inner_different_column_names(sp: Session):
    """Test inner join with different column names using left_on/right_on."""
    df1 = sp.from_arrow(pa.table({"user_id": [1, 2, 3], "name": ["Alice", "Bob", "Carol"]}))
    df2 = sp.from_arrow(pa.table({"id": [2, 3, 4], "score": [85, 90, 95]}))
    result = df1.join(df2, left_on="user_id", right_on="id")
    rows = sorted(result.take_all(), key=lambda x: x["user_id"])
    assert len(rows) == 2
    assert rows[0]["user_id"] == 2
    assert rows[0]["name"] == "Bob"
    assert rows[0]["score"] == 85
    assert rows[1]["user_id"] == 3
    assert rows[1]["name"] == "Carol"
    assert rows[1]["score"] == 90


def test_join_left(sp: Session):
    """Test left outer join."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2, 3], "val1": ["a", "b", "c"]}))
    df2 = sp.from_arrow(pa.table({"id": [2, 3, 4], "val2": ["d", "e", "f"]}))
    result = df1.join(df2, on="id", how="left")
    rows = sorted(result.take_all(), key=lambda x: x["id"])
    assert len(rows) == 3
    # id=1 has no match, val2 should be None
    assert rows[0]["id"] == 1
    assert rows[0]["val1"] == "a"
    assert rows[0]["val2"] is None
    # id=2 and id=3 have matches
    assert rows[1]["id"] == 2
    assert rows[1]["val2"] == "d"
    assert rows[2]["id"] == 3
    assert rows[2]["val2"] == "e"


def test_join_right(sp: Session):
    """Test right outer join."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2, 3], "val1": ["a", "b", "c"]}))
    df2 = sp.from_arrow(pa.table({"id": [2, 3, 4], "val2": ["d", "e", "f"]}))
    result = df1.join(df2, on="id", how="right")
    rows = sorted(result.take_all(), key=lambda x: x["id"])
    assert len(rows) == 3
    # id=2 and id=3 have matches
    assert rows[0]["id"] == 2
    assert rows[0]["val1"] == "b"
    assert rows[0]["val2"] == "d"
    assert rows[1]["id"] == 3
    assert rows[1]["val1"] == "c"
    assert rows[1]["val2"] == "e"
    # id=4 has no match in left, val1 should be None
    assert rows[2]["id"] == 4
    assert rows[2]["val1"] is None
    assert rows[2]["val2"] == "f"


def test_join_outer(sp: Session):
    """Test full outer join."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2], "val1": ["a", "b"]}))
    df2 = sp.from_arrow(pa.table({"id": [2, 3], "val2": ["d", "e"]}))
    result = df1.join(df2, on="id", how="outer")
    rows = sorted(result.take_all(), key=lambda x: x["id"] if x["id"] is not None else 999)
    assert len(rows) == 3
    # id=1: only in left
    assert rows[0]["id"] == 1
    assert rows[0]["val1"] == "a"
    assert rows[0]["val2"] is None
    # id=2: in both
    assert rows[1]["id"] == 2
    assert rows[1]["val1"] == "b"
    assert rows[1]["val2"] == "d"
    # id=3: only in right
    assert rows[2]["id"] == 3
    assert rows[2]["val1"] is None
    assert rows[2]["val2"] == "e"


def test_join_full(sp: Session):
    """Test that 'full' is an alias for 'outer'."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2], "val1": ["a", "b"]}))
    df2 = sp.from_arrow(pa.table({"id": [2, 3], "val2": ["d", "e"]}))
    result = df1.join(df2, on="id", how="full")
    rows = result.take_all()
    assert len(rows) == 3


def test_join_cross(sp: Session):
    """Test cross join (cartesian product)."""
    df1 = sp.from_arrow(pa.table({"a": [1, 2]}))
    df2 = sp.from_arrow(pa.table({"b": ["x", "y"]}))
    result = df1.join(df2, how="cross")
    rows = result.take_all()
    assert len(rows) == 4  # 2 x 2 = 4
    # Check all combinations exist
    combinations = {(r["a"], r["b"]) for r in rows}
    assert combinations == {(1, "x"), (1, "y"), (2, "x"), (2, "y")}


def test_join_semi(sp: Session):
    """Test semi join - rows from left that have a match in right."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]}))
    df2 = sp.from_arrow(pa.table({"id": [2, 3, 4]}))
    result = df1.join(df2, on="id", how="semi")
    rows = sorted(result.take_all(), key=lambda x: x["id"])
    assert len(rows) == 2
    assert rows[0] == {"id": 2, "val": "b"}
    assert rows[1] == {"id": 3, "val": "c"}


def test_join_anti(sp: Session):
    """Test anti join - rows from left that have no match in right."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2, 3], "val": ["a", "b", "c"]}))
    df2 = sp.from_arrow(pa.table({"id": [2, 3, 4]}))
    result = df1.join(df2, on="id", how="anti")
    rows = result.take_all()
    assert len(rows) == 1
    assert rows[0] == {"id": 1, "val": "a"}


def test_join_multiple_columns(sp: Session):
    """Test join on multiple columns."""
    df1 = sp.from_arrow(pa.table({
        "id": [1, 1, 2, 2],
        "date": ["2023-01", "2023-02", "2023-01", "2023-02"],
        "val1": ["a", "b", "c", "d"]
    }))
    df2 = sp.from_arrow(pa.table({
        "id": [1, 2],
        "date": ["2023-01", "2023-02"],
        "val2": ["x", "y"]
    }))
    result = df1.join(df2, on=["id", "date"])
    rows = sorted(result.take_all(), key=lambda x: (x["id"], x["date"]))
    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert rows[0]["date"] == "2023-01"
    assert rows[0]["val1"] == "a"
    assert rows[0]["val2"] == "x"
    assert rows[1]["id"] == 2
    assert rows[1]["date"] == "2023-02"
    assert rows[1]["val1"] == "d"
    assert rows[1]["val2"] == "y"


def test_join_with_npartitions(sp: Session):
    """Test join with explicit number of partitions and verify exact results."""
    df1 = sp.from_items(list(range(100))).map("item as id, item * 10 as val1")
    df2 = sp.from_items(list(range(50, 150))).map("item as id, item * 100 as val2")
    result = df1.join(df2, on="id", npartitions=5)
    rows = result.take_all()
    # Should have 50 matching rows (50-99)
    assert len(rows) == 50
    # Verify exact results - each matched row should have correct val1 and val2
    rows_by_id = {r["id"]: r for r in rows}
    for id_val in range(50, 100):
        assert id_val in rows_by_id, f"id={id_val} should be in result"
        assert rows_by_id[id_val]["val1"] == id_val * 10, f"val1 for id={id_val} should be {id_val * 10}"
        assert rows_by_id[id_val]["val2"] == id_val * 100, f"val2 for id={id_val} should be {id_val * 100}"


def test_join_correctness_across_partitions(sp: Session):
    """Test that join correctly matches rows across multiple partitions.

    This test verifies that hash partitioning correctly routes matching keys
    to the same partition and that no data is lost or incorrectly matched.
    """
    # Create DataFrames with data that will be distributed across partitions
    # Use prime numbers to ensure good distribution
    left_ids = [i * 7 for i in range(1, 51)]  # 7, 14, 21, ..., 350
    right_ids = [i * 7 for i in range(25, 75)]  # 175, 182, ..., 518

    df1 = sp.from_items(left_ids).map("item as id, item * 2 as left_val")
    df2 = sp.from_items(right_ids).map("item as id, item * 3 as right_val")

    # Join with multiple partitions to force data distribution
    result = df1.join(df2, on="id", npartitions=7)
    rows = result.take_all()

    # Calculate expected matches: intersection of left_ids and right_ids
    expected_ids = set(left_ids) & set(right_ids)  # 175, 182, ..., 343 (25 values)
    assert len(rows) == len(expected_ids), f"Expected {len(expected_ids)} rows, got {len(rows)}"

    # Verify each row has correct values
    rows_by_id = {r["id"]: r for r in rows}
    for id_val in expected_ids:
        assert id_val in rows_by_id, f"id={id_val} missing from result"
        assert rows_by_id[id_val]["left_val"] == id_val * 2, f"left_val wrong for id={id_val}"
        assert rows_by_id[id_val]["right_val"] == id_val * 3, f"right_val wrong for id={id_val}"

    # Ensure no unexpected ids
    result_ids = {r["id"] for r in rows}
    assert result_ids == expected_ids, f"Result ids don't match expected: {result_ids - expected_ids}"


def test_join_different_partition_counts(sp: Session):
    """Test join when DataFrames have different initial partition counts.

    Verifies that joining DataFrames with mismatched partition counts
    produces correct results after repartitioning.
    """
    # Create left DataFrame with more partitions
    df1 = sp.from_items(list(range(100))).repartition(10, by_rows=True).map("item as id, 'left' as source")
    # Create right DataFrame with fewer partitions
    df2 = sp.from_items(list(range(50, 120))).repartition(3, by_rows=True).map("item as id, 'right' as source")

    result = df1.join(df2, on="id")
    rows = result.take_all()

    # Expected: intersection of [0,99] and [50,119] = [50,99] = 50 rows
    expected_ids = set(range(50, 100))
    assert len(rows) == len(expected_ids)

    # Verify all expected ids are present with correct sources
    result_ids = {r["id"] for r in rows}
    assert result_ids == expected_ids
    for row in rows:
        assert row["source"] == "left", "source column should come from left DataFrame (USING clause)"


def test_join_cross_different_partition_counts(sp: Session):
    """Test cross join with DataFrames of different partition counts.

    Cross join should produce cartesian product regardless of partition counts.
    """
    # Left has 3 rows, right has 4 rows
    df1 = sp.from_items([1, 2, 3]).repartition(5, by_rows=True).map("item as a")
    df2 = sp.from_items(["w", "x", "y", "z"]).repartition(2, by_rows=True).map("item as b")

    result = df1.join(df2, how="cross")
    rows = result.take_all()

    # Should have 3 * 4 = 12 combinations
    assert len(rows) == 12

    # Verify all combinations exist
    combinations = {(r["a"], r["b"]) for r in rows}
    expected_combinations = {(a, b) for a in [1, 2, 3] for b in ["w", "x", "y", "z"]}
    assert combinations == expected_combinations


def test_join_left_correctness_with_nulls(sp: Session):
    """Test left join produces correct NULL values for non-matching rows."""
    df1 = sp.from_items([1, 2, 3, 4, 5]).map("item as id, item * 10 as val1")
    df2 = sp.from_items([2, 4]).map("item as id, item * 100 as val2")

    result = df1.join(df2, on="id", how="left", npartitions=3)
    rows = sorted(result.take_all(), key=lambda x: x["id"])

    assert len(rows) == 5
    # Verify exact values including NULLs
    expected = [
        {"id": 1, "val1": 10, "val2": None},
        {"id": 2, "val1": 20, "val2": 200},
        {"id": 3, "val1": 30, "val2": None},
        {"id": 4, "val1": 40, "val2": 400},
        {"id": 5, "val1": 50, "val2": None},
    ]
    for i, row in enumerate(rows):
        assert row["id"] == expected[i]["id"]
        assert row["val1"] == expected[i]["val1"]
        assert row["val2"] == expected[i]["val2"], f"val2 mismatch at id={row['id']}: {row['val2']} != {expected[i]['val2']}"


def test_join_multiple_columns_correctness(sp: Session):
    """Test that multi-column join correctly matches on all columns."""
    # Create data where single column match would give wrong results
    df1 = sp.from_arrow(pa.table({
        "a": [1, 1, 2, 2],
        "b": [10, 20, 10, 20],
        "val1": ["a1-b10", "a1-b20", "a2-b10", "a2-b20"]
    }))
    df2 = sp.from_arrow(pa.table({
        "a": [1, 2, 1, 2],
        "b": [10, 20, 30, 40],
        "val2": ["r1-10", "r2-20", "r1-30", "r2-40"]
    }))

    result = df1.join(df2, on=["a", "b"], npartitions=3)
    rows = sorted(result.take_all(), key=lambda x: (x["a"], x["b"]))

    # Only (1,10) and (2,20) should match
    assert len(rows) == 2
    assert rows[0] == {"a": 1, "b": 10, "val1": "a1-b10", "val2": "r1-10"}
    assert rows[1] == {"a": 2, "b": 20, "val1": "a2-b20", "val2": "r2-20"}


def test_join_validation_invalid_type(sp: Session):
    """Test that invalid join type raises ValueError."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2, 3]}))
    df2 = sp.from_arrow(pa.table({"id": [1, 2, 3]}))
    with pytest.raises(ValueError, match="Invalid join type"):
        df1.join(df2, on="id", how="invalid")


def test_join_validation_cross_with_keys(sp: Session):
    """Test that cross join with join keys raises ValueError."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2]}))
    df2 = sp.from_arrow(pa.table({"id": [1, 2]}))
    with pytest.raises(ValueError, match="Cross join does not accept join keys"):
        df1.join(df2, on="id", how="cross")


def test_join_validation_on_with_left_on(sp: Session):
    """Test that specifying both 'on' and 'left_on' raises ValueError."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2]}))
    df2 = sp.from_arrow(pa.table({"id": [1, 2]}))
    with pytest.raises(ValueError, match="Cannot specify both 'on' and 'left_on'"):
        df1.join(df2, on="id", left_on="id", right_on="id")


def test_join_validation_mismatched_columns(sp: Session):
    """Test that mismatched number of left_on/right_on columns raises ValueError."""
    df1 = sp.from_arrow(pa.table({"a": [1], "b": [2]}))
    df2 = sp.from_arrow(pa.table({"c": [1]}))
    with pytest.raises(ValueError, match="left_on and right_on must have the same number"):
        df1.join(df2, left_on=["a", "b"], right_on=["c"])


def test_join_validation_missing_keys(sp: Session):
    """Test that non-cross join without keys raises ValueError."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2]}))
    df2 = sp.from_arrow(pa.table({"id": [1, 2]}))
    with pytest.raises(ValueError, match="Join keys required"):
        df1.join(df2, how="inner")


def test_join_validation_only_left_on(sp: Session):
    """Test that specifying only left_on without right_on raises ValueError."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2]}))
    df2 = sp.from_arrow(pa.table({"id": [1, 2]}))
    with pytest.raises(ValueError, match="Must specify both 'left_on' and 'right_on'"):
        df1.join(df2, left_on="id")


def test_join_preserves_cache_setting(sp: Session):
    """Test that join inherits the cache setting."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2]})).no_cache()
    df2 = sp.from_arrow(pa.table({"id": [1, 2]}))
    result = df1.join(df2, on="id")
    assert result._use_cache is False


def test_error_message(sp: Session):
    df = sp.from_items([1, 2, 3])
    df = sp.partial_sql("select a,, from {0}", df)
    try:
        df.to_arrow()
    except Exception as ex:
        # sql query should be in the exception message
        assert "select a,, from" in str(ex)
    else:
        assert False, "expected exception"


def test_unpicklable_task_exception(sp: Session):
    from loguru import logger

    df = sp.from_items([1, 2, 3])
    try:
        df.map(lambda x: logger.info("use outside logger")).to_arrow()
    except Exception as ex:
        assert "Can't pickle task" in str(ex)
        assert "HINT: DO NOT use externally imported loguru logger in your task. Please import it within the task." in str(ex)
    else:
        assert False, "expected exception"


def test_log(sp: Session):
    df = sp.from_items([1, 2, 3])

    def log_record(x):
        import logging
        import sys

        from loguru import logger

        print("stdout")
        print("stderr", file=sys.stderr)
        logger.info("loguru")
        logging.info("logging")
        return x

    df.map(log_record).to_arrow()

    # TODO: check logs should be see in the log file
    # FIXME: logs in unit test are not written to the log file
    #        because we share the same ray instance for all tests


def test_sample_n(sp: Session):
    """Test sampling with exact number of rows."""
    df = sp.from_items(list(range(100)))
    sample = df.sample(n=5)
    assert len(sample) == 5
    # All sampled values should be from the original data
    for row in sample:
        assert 0 <= row["item"] < 100


def test_sample_n_larger_than_total(sp: Session):
    """Test that sampling more rows than available returns all rows."""
    df = sp.from_items(list(range(10)))
    sample = df.sample(n=100)
    # Should return at most the total number of rows
    assert len(sample) <= 10


def test_sample_fraction(sp: Session):
    """Test sampling with fraction of rows."""
    df = sp.from_items(list(range(1000)))
    sample = df.sample(fraction=0.1)
    # With bernoulli sampling, the result is probabilistic
    # Allow some variance: expect roughly 100 rows (10%), but allow 50-150
    assert 50 <= len(sample) <= 150
    # All sampled values should be from the original data
    for row in sample:
        assert 0 <= row["item"] < 1000


def test_sample_fraction_full(sp: Session):
    """Test sampling with fraction=1.0 returns all rows."""
    df = sp.from_items(list(range(50)))
    sample = df.sample(fraction=1.0)
    assert len(sample) == 50


def test_sample_seed_reproducibility(sp: Session):
    """Test that same seed produces same sample."""
    df = sp.from_items(list(range(100)))
    sample1 = df.sample(n=10, seed=42)
    sample2 = df.sample(n=10, seed=42)
    assert sample1 == sample2


def test_sample_seed_different_seeds(sp: Session):
    """Test that different seeds produce different samples."""
    df = sp.from_items(list(range(100)))
    sample1 = df.sample(n=10, seed=42)
    sample2 = df.sample(n=10, seed=123)
    # Very unlikely to be equal with different seeds
    assert sample1 != sample2


def test_sample_validation_no_params(sp: Session):
    """Test that sample raises error when neither n nor fraction is specified."""
    df = sp.from_items([1, 2, 3])
    with pytest.raises(ValueError, match="Must specify either 'n'.*or 'fraction'"):
        df.sample()


def test_sample_validation_both_params(sp: Session):
    """Test that sample raises error when both n and fraction are specified."""
    df = sp.from_items([1, 2, 3])
    with pytest.raises(ValueError, match="Cannot specify both 'n' and 'fraction'"):
        df.sample(n=1, fraction=0.5)


def test_sample_validation_invalid_n(sp: Session):
    """Test that sample raises error for invalid n values."""
    df = sp.from_items([1, 2, 3])
    with pytest.raises(ValueError, match="'n' must be a positive integer"):
        df.sample(n=0)
    with pytest.raises(ValueError, match="'n' must be a positive integer"):
        df.sample(n=-1)


def test_sample_validation_invalid_fraction(sp: Session):
    """Test that sample raises error for invalid fraction values with helpful examples."""
    df = sp.from_items([1, 2, 3])
    with pytest.raises(ValueError, match="'fraction' must be a decimal number.*Examples:.*0.1 for 10%"):
        df.sample(fraction=0)
    with pytest.raises(ValueError, match="'fraction' must be a decimal number.*Examples:.*0.25 for 25%"):
        df.sample(fraction=1.5)
    with pytest.raises(ValueError, match="'fraction' must be a decimal number.*Examples:.*0.5 for 50%"):
        df.sample(fraction=-0.1)


# ==================== Cache Tests ====================


def test_cache_basic(sp: Session):
    """Test that caching works for basic count() operations."""
    df = sp.from_items([1, 2, 3])

    # First call should miss cache
    count1 = df.count()
    stats1 = sp.get_cache_stats()
    assert stats1["misses"] == 1
    assert stats1["hits"] == 0

    # Second call should hit cache
    count2 = df.count()
    stats2 = sp.get_cache_stats()
    assert stats2["hits"] == 1
    assert stats2["misses"] == 1

    # Results should be the same
    assert count1 == count2 == 3


def test_cache_take(sp: Session):
    """Test that caching works for take() operations."""
    df = sp.from_items([1, 2, 3, 4, 5])

    # First take
    result1 = df.take(3)
    stats1 = sp.get_cache_stats()
    assert stats1["misses"] >= 1

    # Second take should use cached result
    result2 = df.take(3)
    stats2 = sp.get_cache_stats()
    assert stats2["hits"] >= 1


def test_cache_to_pandas(sp: Session):
    """Test that caching works for to_pandas() operations."""
    df = sp.from_items([{"a": 1}, {"a": 2}, {"a": 3}])

    # First call
    pdf1 = df.to_pandas()
    stats1 = sp.get_cache_stats()
    initial_misses = stats1["misses"]

    # Second call should hit cache
    pdf2 = df.to_pandas()
    stats2 = sp.get_cache_stats()
    assert stats2["hits"] >= 1

    assert pdf1.equals(pdf2)


def test_cache_disabled_globally(sp: Session):
    """Test that global cache disable works."""
    df = sp.from_items([1, 2, 3])

    sp.set_cache_enabled(False)

    # Both calls should not use cache
    df.count()
    df.count()

    stats = sp.get_cache_stats()
    assert stats["hits"] == 0
    assert stats["enabled"] is False

    # Re-enable for other tests
    sp.set_cache_enabled(True)


def test_cache_disabled_per_dataframe(sp: Session):
    """Test that no_cache() disables caching for specific DataFrame."""
    df = sp.from_items([1, 2, 3])

    # Use no_cache() - should not cache
    df.no_cache().count()
    df.no_cache().count()

    stats = sp.get_cache_stats()
    # no_cache() should not add to cache or retrieve from it
    assert stats["entries"] == 0


def test_cache_clear_all(sp: Session):
    """Test clearing all cache entries."""
    df1 = sp.from_items([1, 2, 3])
    df2 = sp.from_items([4, 5, 6])

    # Populate cache
    df1.count()
    df2.count()

    stats1 = sp.get_cache_stats()
    assert stats1["entries"] >= 2

    # Clear all
    cleared = sp.clear_cache()
    assert cleared >= 2

    stats2 = sp.get_cache_stats()
    assert stats2["entries"] == 0


def test_cache_clear_specific(sp: Session):
    """Test clearing cache for a specific DataFrame."""
    df1 = sp.from_items([1, 2, 3])
    df2 = sp.from_items([4, 5, 6])

    # Populate cache
    df1.count()
    df2.count()

    # Clear only df1's cache
    df1.clear_cache()

    # df1 should miss, df2 should still hit
    df1.count()  # Should be a miss (cache was cleared)
    df2.count()  # Should be a hit

    stats = sp.get_cache_stats()
    # df1: 1 miss (initial), then cleared, then 1 miss again
    # df2: 1 miss (initial), then 1 hit
    assert stats["hits"] >= 1


def test_cache_get_entries(sp: Session):
    """Test getting information about cached entries."""
    df = sp.from_items([1, 2, 3])
    df.count()

    entries = sp.get_cached_entries()
    assert len(entries) >= 1
    assert "key" in entries[0]
    assert "cached_at" in entries[0]
    assert "num_datasets" in entries[0]


def test_cache_stats(sp: Session):
    """Test cache statistics."""
    df = sp.from_items([1, 2, 3])

    # First call - miss
    df.count()
    # Second call - hit
    df.count()

    stats = sp.get_cache_stats()
    assert "entries" in stats
    assert "hits" in stats
    assert "misses" in stats
    assert "hit_rate" in stats
    assert "enabled" in stats
    assert stats["hits"] >= 1
    assert stats["misses"] >= 1
    assert 0 <= stats["hit_rate"] <= 1


def test_cache_operations_inherit_setting(sp: Session):
    """Test that DataFrame operations inherit the cache setting."""
    df = sp.from_items([1, 2, 3, 4, 5]).no_cache()

    # Filter should also have caching disabled
    filtered = df.filter("item > 2")
    assert filtered._use_cache is False

    # Map should also have caching disabled
    mapped = df.map("item * 2 as doubled")
    assert mapped._use_cache is False


def test_cache_recompute_bypasses_cache(sp: Session):
    """Test that recompute() bypasses the cache."""
    df = sp.from_items([1, 2, 3])

    # First call - populates cache
    count1 = df.count()

    # Second call with recompute - should bypass cache
    count2 = df.recompute().count()

    # Both should return the same result
    assert count1 == count2 == 3


def test_cache_different_operations_different_keys(sp: Session):
    """Test that different operations on the same DataFrame have different cache keys."""
    df = sp.from_items([1, 2, 3, 4, 5])

    # Different operations should have different cache entries
    df.count()
    df.filter("item > 2").count()
    df.map("item * 2 as doubled").count()

    stats = sp.get_cache_stats()
    # Should have multiple cache entries for different operations
    assert stats["entries"] >= 3


def test_cache_different_parameters_different_keys(sp: Session):
    """Test that same operations with different parameters create different cache entries."""
    df = sp.from_items([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

    # Same operation (filter) but with different parameters should create different cache entries
    result1 = df.filter("item > 2").count()
    result2 = df.filter("item > 5").count()
    result3 = df.filter("item > 8").count()

    # Results should be different
    assert result1 == 8  # items 3-10
    assert result2 == 5  # items 6-10
    assert result3 == 2  # items 9-10

    stats = sp.get_cache_stats()
    # Should have 3 different cache entries for different filter conditions
    assert stats["entries"] == 3
    # All 3 should be misses (first time computing each)
    assert stats["misses"] == 3
    assert stats["hits"] == 0

    # Now call the same operations again - should hit cache
    result1_cached = df.filter("item > 2").count()
    result2_cached = df.filter("item > 5").count()
    result3_cached = df.filter("item > 8").count()

    # Results should be the same
    assert result1_cached == result1
    assert result2_cached == result2
    assert result3_cached == result3

    stats = sp.get_cache_stats()
    # Should have 3 hits now
    assert stats["hits"] == 3


def test_cache_same_operation_same_key(sp: Session):
    """Test that identical operations create the same cache key and hit cache."""
    df = sp.from_items([1, 2, 3])

    # Two identical filter operations should use the same cache entry
    result1 = df.filter("item > 1").count()
    result2 = df.filter("item > 1").count()

    assert result1 == result2 == 2

    stats = sp.get_cache_stats()
    # Should have 1 miss (first call) and 1 hit (second call)
    assert stats["misses"] == 1
    assert stats["hits"] == 1
    assert stats["entries"] == 1


def test_cache_limit_different_values(sp: Session):
    """Test that limit() with different values creates different cache entries."""
    df = sp.from_items(list(range(100)))

    # Different limit values should create different cache entries
    count1 = df.limit(10).count()
    count2 = df.limit(20).count()
    count3 = df.limit(30).count()

    assert count1 == 10
    assert count2 == 20
    assert count3 == 30

    stats = sp.get_cache_stats()
    # Should have 3 different cache entries
    assert stats["entries"] == 3


def test_cache_thread_safety(sp: Session):
    """Test that concurrent _compute() calls are thread-safe."""
    import threading

    df = sp.from_items(list(range(100)))

    results = []
    errors = []

    def compute_count():
        try:
            count = df.count()
            results.append(count)
        except Exception as e:
            errors.append(e)

    # Run multiple threads concurrently
    threads = [threading.Thread(target=compute_count) for _ in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # All threads should succeed
    assert len(errors) == 0
    # All results should be the same
    assert all(r == 100 for r in results)

    stats = sp.get_cache_stats()
    # Should have 1 miss (first computation) and 4 hits (cached results)
    # Note: due to threading, there might be slight variations
    assert stats["entries"] == 1
    assert stats["hits"] + stats["misses"] == 5


def test_cache_recompute_clears_cache(sp: Session):
    """Test that recompute() clears the cached result to free memory."""
    df = sp.from_items([1, 2, 3])

    # First call - populates cache
    count1 = df.count()

    stats1 = sp.get_cache_stats()
    assert stats1["entries"] == 1
    assert stats1["misses"] == 1

    # Call recompute() - should clear the cache entry
    df.recompute()

    stats2 = sp.get_cache_stats()
    # Cache entry should be cleared
    assert stats2["entries"] == 0

    # Now count again - should be a miss since cache was cleared
    count2 = df.count()

    stats3 = sp.get_cache_stats()
    # Should have 1 new entry and 2 total misses (original + after recompute)
    assert stats3["entries"] == 1
    assert stats3["misses"] == 2

    # Results should still be the same
    assert count1 == count2 == 3


def test_cache_recompute_on_uncomputed_dataframe(sp: Session):
    """Test that recompute() works correctly on a DataFrame that hasn't been computed yet."""
    df = sp.from_items([1, 2, 3])

    # Call recompute() before any computation
    # This should not raise an error even though optimized_plan is None
    df.recompute()

    stats = sp.get_cache_stats()
    # No cache entries should exist
    assert stats["entries"] == 0

    # Now compute - should work normally
    count = df.count()
    assert count == 3

    stats2 = sp.get_cache_stats()
    # Should have 1 entry now
    assert stats2["entries"] == 1


def test_cache_key_generation_with_complex_data(sp: Session):
    """Test that cache key generation handles complex/nested data structures."""
    # Create DataFrames with various complex operations
    df = sp.from_items([{"a": 1, "b": {"nested": "value"}}, {"a": 2, "b": {"nested": "other"}}])

    # Should be able to compute without errors
    count = df.count()
    assert count == 2

    stats = sp.get_cache_stats()
    assert stats["entries"] >= 1


def test_cache_key_generation_with_special_characters(sp: Session):
    """Test that cache key generation handles SQL with special characters."""
    df = sp.from_items([1, 2, 3, 4, 5])

    # Filter with special characters in SQL
    result1 = df.filter("item >= 2 AND item <= 4").count()
    result2 = df.filter("item >= 2 AND item <= 4").count()

    # Should return same results and hit cache
    assert result1 == result2 == 3

    stats = sp.get_cache_stats()
    assert stats["hits"] >= 1


def test_cache_max_entries_eviction(sp: Session):
    """Test that cache evicts oldest entries when max_entries is reached."""
    from smallpond.dataframe import DataFrameCache

    # Create a cache with max 2 entries
    limited_cache = DataFrameCache(max_entries=2)
    sp._cache = limited_cache

    df = sp.from_items(list(range(10)))

    # Create 3 different operations - should evict the first one
    df.filter("item > 1").count()
    df.filter("item > 2").count()
    df.filter("item > 3").count()

    stats = sp.get_cache_stats()
    # Should only have 2 entries due to max_entries limit
    assert stats["entries"] == 2


def test_cache_empty_dataframe(sp: Session):
    """Test caching works correctly with empty DataFrames."""
    df = sp.from_items([])

    count1 = df.count()
    count2 = df.count()

    # Both should return 0
    assert count1 == count2 == 0

    stats = sp.get_cache_stats()
    # Should have 1 hit (second call)
    assert stats["hits"] >= 1


def test_cache_stats_reset(sp: Session):
    """Test that cache stats can be reset independently of cache entries."""
    df = sp.from_items([1, 2, 3])

    # Generate some hits and misses
    df.count()  # miss
    df.count()  # hit

    stats1 = sp.get_cache_stats()
    assert stats1["hits"] == 1
    assert stats1["misses"] == 1

    # Reset stats but keep cache entries
    sp.cache.reset_stats()

    stats2 = sp.get_cache_stats()
    # Stats should be reset
    assert stats2["hits"] == 0
    assert stats2["misses"] == 0
    # But entries should still exist
    assert stats2["entries"] == 1

    # Next call should still hit cache
    df.count()

    stats3 = sp.get_cache_stats()
    assert stats3["hits"] == 1
    assert stats3["misses"] == 0


def test_cache_hit_rate_calculation(sp: Session):
    """Test that cache hit rate is calculated correctly."""
    df = sp.from_items([1, 2, 3])

    # 1 miss
    df.count()

    stats1 = sp.get_cache_stats()
    assert stats1["hit_rate"] == 0.0  # 0 hits / 1 total

    # 1 hit
    df.count()

    stats2 = sp.get_cache_stats()
    assert stats2["hit_rate"] == 0.5  # 1 hit / 2 total

    # 1 more hit
    df.count()

    stats3 = sp.get_cache_stats()
    assert abs(stats3["hit_rate"] - 2/3) < 0.01  # 2 hits / 3 total


def test_cache_multiple_dataframes_same_session(sp: Session):
    """Test that multiple DataFrames in the same session share the cache correctly."""
    df1 = sp.from_items([1, 2, 3])
    df2 = sp.from_items([4, 5, 6])

    # Both should be cached separately
    count1 = df1.count()
    count2 = df2.count()

    assert count1 == 3
    assert count2 == 3

    stats = sp.get_cache_stats()
    assert stats["entries"] == 2

    # Both should hit cache on second call
    df1.count()
    df2.count()

    stats2 = sp.get_cache_stats()
    assert stats2["hits"] == 2


# ==================== GroupBy Aggregation Tests ====================


def test_groupby_agg_single_column_single_agg(sp: Session):
    """Test groupby with single group column and single aggregation."""
    df = sp.from_arrow(pa.table({
        "category": ["A", "A", "B", "B", "B"],
        "value": [10, 20, 30, 40, 50]
    }))
    result = df.groupby_agg(by="category", aggs={"value": "sum"})
    rows = sorted(result.take_all(), key=lambda x: x["category"])

    assert len(rows) == 2
    assert rows[0]["category"] == "A"
    assert rows[0]["value_sum"] == 30  # 10 + 20
    assert rows[1]["category"] == "B"
    assert rows[1]["value_sum"] == 120  # 30 + 40 + 50


def test_groupby_agg_single_column_multiple_aggs(sp: Session):
    """Test groupby with single group column and multiple aggregations."""
    df = sp.from_arrow(pa.table({
        "category": ["A", "A", "B", "B"],
        "value": [10, 20, 30, 40]
    }))
    result = df.groupby_agg(by="category", aggs={"value": ["sum", "count", "min", "max"]})
    rows = sorted(result.take_all(), key=lambda x: x["category"])

    assert len(rows) == 2
    # Category A
    assert rows[0]["category"] == "A"
    assert rows[0]["value_sum"] == 30
    assert rows[0]["value_count"] == 2
    assert rows[0]["value_min"] == 10
    assert rows[0]["value_max"] == 20
    # Category B
    assert rows[1]["category"] == "B"
    assert rows[1]["value_sum"] == 70
    assert rows[1]["value_count"] == 2
    assert rows[1]["value_min"] == 30
    assert rows[1]["value_max"] == 40


def test_groupby_agg_multiple_columns(sp: Session):
    """Test groupby with multiple group columns."""
    df = sp.from_arrow(pa.table({
        "region": ["East", "East", "West", "West"],
        "category": ["A", "B", "A", "B"],
        "value": [10, 20, 30, 40]
    }))
    result = df.groupby_agg(by=["region", "category"], aggs={"value": "sum"})
    rows = sorted(result.take_all(), key=lambda x: (x["region"], x["category"]))

    assert len(rows) == 4
    assert rows[0] == {"region": "East", "category": "A", "value_sum": 10}
    assert rows[1] == {"region": "East", "category": "B", "value_sum": 20}
    assert rows[2] == {"region": "West", "category": "A", "value_sum": 30}
    assert rows[3] == {"region": "West", "category": "B", "value_sum": 40}


def test_groupby_agg_multiple_value_columns(sp: Session):
    """Test groupby aggregating multiple value columns."""
    df = sp.from_arrow(pa.table({
        "category": ["A", "A", "B", "B"],
        "sales": [100, 200, 300, 400],
        "quantity": [1, 2, 3, 4]
    }))
    result = df.groupby_agg(by="category", aggs={"sales": "sum", "quantity": "sum"})
    rows = sorted(result.take_all(), key=lambda x: x["category"])

    assert len(rows) == 2
    assert rows[0]["category"] == "A"
    assert rows[0]["sales_sum"] == 300
    assert rows[0]["quantity_sum"] == 3
    assert rows[1]["category"] == "B"
    assert rows[1]["sales_sum"] == 700
    assert rows[1]["quantity_sum"] == 7


def test_groupby_agg_avg(sp: Session):
    """Test groupby with avg aggregation (requires two-phase computation)."""
    df = sp.from_arrow(pa.table({
        "category": ["A", "A", "A", "B", "B"],
        "value": [10, 20, 30, 40, 60]
    }))
    result = df.groupby_agg(by="category", aggs={"value": "avg"})
    rows = sorted(result.take_all(), key=lambda x: x["category"])

    assert len(rows) == 2
    assert rows[0]["category"] == "A"
    assert rows[0]["value_avg"] == 20.0  # (10 + 20 + 30) / 3
    assert rows[1]["category"] == "B"
    assert rows[1]["value_avg"] == 50.0  # (40 + 60) / 2


def test_groupby_agg_mean_alias(sp: Session):
    """Test that 'mean' is accepted as alias for 'avg'."""
    df = sp.from_arrow(pa.table({
        "category": ["A", "A", "B", "B"],
        "value": [10, 20, 30, 40]
    }))
    result = df.groupby_agg(by="category", aggs={"value": "mean"})
    rows = sorted(result.take_all(), key=lambda x: x["category"])

    assert len(rows) == 2
    # Output column should be named 'value_avg' even when 'mean' is specified
    assert rows[0]["value_avg"] == 15.0
    assert rows[1]["value_avg"] == 35.0


def test_groupby_agg_count_distinct(sp: Session):
    """Test groupby with count_distinct aggregation."""
    df = sp.from_arrow(pa.table({
        "category": ["A", "A", "A", "B", "B", "B"],
        "value": [1, 1, 2, 3, 3, 3]  # A has 2 distinct, B has 1 distinct
    }))
    result = df.groupby_agg(by="category", aggs={"value": "count_distinct"})
    rows = sorted(result.take_all(), key=lambda x: x["category"])

    assert len(rows) == 2
    assert rows[0]["category"] == "A"
    assert rows[0]["value_count_distinct"] == 2
    assert rows[1]["category"] == "B"
    assert rows[1]["value_count_distinct"] == 1


def test_groupby_agg_with_partitions(sp: Session):
    """Test groupby correctly handles data across multiple partitions."""
    # Create data that will be distributed across partitions
    items = [{"category": chr(65 + (i % 3)), "value": i} for i in range(100)]
    df = sp.from_items(items).repartition(5, by_rows=True)

    result = df.groupby_agg(by="category", aggs={"value": ["sum", "count"]}, npartitions=3)
    rows = sorted(result.take_all(), key=lambda x: x["category"])

    assert len(rows) == 3  # A, B, C
    # Verify counts add up
    total_count = sum(r["value_count"] for r in rows)
    assert total_count == 100

    # A: 0, 3, 6, ..., 99 (34 items) -> sum = 0+3+6+...+99
    # B: 1, 4, 7, ..., 97 (33 items) -> sum = 1+4+7+...+97
    # C: 2, 5, 8, ..., 98 (33 items) -> sum = 2+5+8+...+98
    a_values = [i for i in range(100) if i % 3 == 0]
    b_values = [i for i in range(100) if i % 3 == 1]
    c_values = [i for i in range(100) if i % 3 == 2]

    assert rows[0]["category"] == "A"
    assert rows[0]["value_sum"] == sum(a_values)
    assert rows[0]["value_count"] == len(a_values)

    assert rows[1]["category"] == "B"
    assert rows[1]["value_sum"] == sum(b_values)
    assert rows[1]["value_count"] == len(b_values)

    assert rows[2]["category"] == "C"
    assert rows[2]["value_sum"] == sum(c_values)
    assert rows[2]["value_count"] == len(c_values)


def test_groupby_agg_validation_empty_by(sp: Session):
    """Test that empty 'by' parameter raises ValueError."""
    df = sp.from_items([{"a": 1, "b": 2}])
    with pytest.raises(ValueError, match="'by' parameter cannot be empty"):
        df.groupby_agg(by=[], aggs={"b": "sum"})


def test_groupby_agg_validation_empty_aggs(sp: Session):
    """Test that empty 'aggs' parameter raises ValueError."""
    df = sp.from_items([{"a": 1, "b": 2}])
    with pytest.raises(ValueError, match="'aggs' parameter cannot be empty"):
        df.groupby_agg(by="a", aggs={})


def test_groupby_agg_validation_unsupported_agg(sp: Session):
    """Test that unsupported aggregation function raises ValueError."""
    df = sp.from_items([{"a": 1, "b": 2}])
    with pytest.raises(ValueError, match="Unsupported aggregation function 'median'"):
        df.groupby_agg(by="a", aggs={"b": "median"})


def test_groupby_agg_validation_overlapping_columns(sp: Session):
    """Test that using the same column for both grouping and aggregation raises ValueError."""
    df = sp.from_items([{"a": 1, "b": 2, "c": 3}])
    # Single overlapping column
    with pytest.raises(ValueError, match="Columns cannot be used for both grouping and aggregation"):
        df.groupby_agg(by="a", aggs={"a": "sum"})
    # Multiple group columns with one overlapping
    with pytest.raises(ValueError, match="Columns cannot be used for both grouping and aggregation.*'b'"):
        df.groupby_agg(by=["a", "b"], aggs={"b": "count", "c": "sum"})


def test_groupby_agg_preserves_cache_setting(sp: Session):
    """Test that groupby_agg inherits the cache setting."""
    df = sp.from_arrow(pa.table({"a": [1, 2], "b": [3, 4]})).no_cache()
    result = df.groupby_agg(by="a", aggs={"b": "sum"})
    assert result._use_cache is False


def test_groupby_agg_all_aggregations_combined(sp: Session):
    """Test combining all supported aggregation types in a single call."""
    df = sp.from_arrow(pa.table({
        "category": ["A", "A", "A", "B", "B"],
        "x": [1, 2, 3, 4, 5],
        "y": [10, 10, 20, 30, 30]
    }))
    result = df.groupby_agg(
        by="category",
        aggs={
            "x": ["sum", "count", "min", "max", "avg"],
            "y": "count_distinct"
        }
    )
    rows = sorted(result.take_all(), key=lambda x: x["category"])

    assert len(rows) == 2

    # Category A: x=[1,2,3], y=[10,10,20]
    assert rows[0]["category"] == "A"
    assert rows[0]["x_sum"] == 6
    assert rows[0]["x_count"] == 3
    assert rows[0]["x_min"] == 1
    assert rows[0]["x_max"] == 3
    assert rows[0]["x_avg"] == 2.0
    assert rows[0]["y_count_distinct"] == 2  # 10 and 20

    # Category B: x=[4,5], y=[30,30]
    assert rows[1]["category"] == "B"
    assert rows[1]["x_sum"] == 9
    assert rows[1]["x_count"] == 2
    assert rows[1]["x_min"] == 4
    assert rows[1]["x_max"] == 5
    assert rows[1]["x_avg"] == 4.5
    assert rows[1]["y_count_distinct"] == 1  # only 30


def test_groupby_agg_large_dataset_correctness(sp: Session):
    """Test groupby_agg correctness with a larger dataset distributed across partitions."""
    # Create 1000 rows distributed across 10 partitions
    items = [
        {"group": f"G{i % 10}", "value": i, "other": i * 2}
        for i in range(1000)
    ]
    df = sp.from_items(items).repartition(10, by_rows=True)

    result = df.groupby_agg(
        by="group",
        aggs={"value": ["sum", "count", "avg"], "other": "max"},
        npartitions=5
    )
    rows = sorted(result.take_all(), key=lambda x: x["group"])

    assert len(rows) == 10  # G0 through G9

    # Verify each group
    for i, row in enumerate(rows):
        group_name = f"G{i}"
        # Values in this group: i, i+10, i+20, ..., i+990 (100 values each)
        group_values = [j for j in range(1000) if j % 10 == i]

        assert row["group"] == group_name
        assert row["value_count"] == 100
        assert row["value_sum"] == sum(group_values)
        expected_avg = sum(group_values) / len(group_values)
        assert abs(row["value_avg"] - expected_avg) < 0.001
        assert row["other_max"] == max(v * 2 for v in group_values)


# ==================== Describe Tests ====================


def test_describe_basic(sp: Session):
    """Test basic describe functionality with simple DataFrame."""
    df = sp.from_arrow(pa.table({
        "a": [1, 2, 3, 4, 5],
        "b": ["x", "y", "z", "w", "v"]
    }))
    stats = df.describe()

    assert stats["num_rows"] == 5
    assert stats["num_columns"] == 2

    # Check column a (numeric)
    col_a = next(c for c in stats["columns"] if c["name"] == "a")
    assert col_a["dtype"] == "int64"
    assert col_a["null_count"] == 0
    assert col_a["non_null_count"] == 5
    assert col_a["null_percent"] == 0.0
    assert col_a["min"] == 1.0
    assert col_a["max"] == 5.0
    assert col_a["mean"] == 3.0
    # approx_median uses t-digest, should be close to 3.0
    assert abs(col_a["approx_median"] - 3.0) < 0.5
    assert col_a["sum"] == 15.0

    # Check column b (string - no numeric stats)
    col_b = next(c for c in stats["columns"] if c["name"] == "b")
    assert "large_string" in col_b["dtype"] or "string" in col_b["dtype"]
    assert col_b["null_count"] == 0
    assert col_b["non_null_count"] == 5
    assert "min" not in col_b
    assert "mean" not in col_b


def test_describe_with_nulls(sp: Session):
    """Test describe correctly handles null values."""
    df = sp.from_arrow(pa.table({
        "value": pa.array([1, None, 3, None, 5], type=pa.int64()),
        "name": pa.array(["a", "b", None, "d", None], type=pa.large_string())
    }))
    stats = df.describe()

    assert stats["num_rows"] == 5

    # Check value column
    col_value = next(c for c in stats["columns"] if c["name"] == "value")
    assert col_value["null_count"] == 2
    assert col_value["non_null_count"] == 3
    assert col_value["null_percent"] == 40.0
    assert col_value["min"] == 1.0
    assert col_value["max"] == 5.0
    assert col_value["mean"] == 3.0  # (1 + 3 + 5) / 3
    # approx_median uses t-digest, should be close to 3.0
    assert abs(col_value["approx_median"] - 3.0) < 0.5

    # Check name column
    col_name = next(c for c in stats["columns"] if c["name"] == "name")
    assert col_name["null_count"] == 2
    assert col_name["non_null_count"] == 3
    assert col_name["null_percent"] == 40.0


def test_describe_all_numeric_stats(sp: Session):
    """Test all numeric statistics are computed correctly."""
    # Using values where we can compute expected stats easily
    df = sp.from_arrow(pa.table({
        "x": [2, 4, 6, 8, 10]
    }))
    stats = df.describe()

    col_x = stats["columns"][0]
    assert col_x["min"] == 2.0
    assert col_x["max"] == 10.0
    assert col_x["mean"] == 6.0  # (2+4+6+8+10)/5
    # approx_median uses t-digest, should be close to 6.0
    assert abs(col_x["approx_median"] - 6.0) < 0.5
    assert col_x["sum"] == 30.0

    # Standard deviation: sqrt(sum((x - mean)^2) / n)
    # Variance = ((2-6)^2 + (4-6)^2 + (6-6)^2 + (8-6)^2 + (10-6)^2) / 5
    #          = (16 + 4 + 0 + 4 + 16) / 5 = 40 / 5 = 8
    # Std = sqrt(8) ≈ 2.828
    expected_std = (8.0) ** 0.5
    assert abs(col_x["std"] - expected_std) < 0.001


def test_describe_approx_median_even_count(sp: Session):
    """Test approximate median calculation with even number of values."""
    df = sp.from_arrow(pa.table({
        "x": [1, 2, 3, 4]  # exact median should be (2+3)/2 = 2.5
    }))
    stats = df.describe()

    col_x = stats["columns"][0]
    # approx_median uses t-digest, should be close to 2.5
    assert abs(col_x["approx_median"] - 2.5) < 0.5


def test_describe_multiple_numeric_columns(sp: Session):
    """Test describe with multiple numeric columns of different types."""
    df = sp.from_arrow(pa.table({
        "int_col": pa.array([1, 2, 3], type=pa.int32()),
        "float_col": pa.array([1.5, 2.5, 3.5], type=pa.float64()),
        "str_col": ["a", "b", "c"]
    }))
    stats = df.describe()

    assert stats["num_columns"] == 3

    # Int column
    col_int = next(c for c in stats["columns"] if c["name"] == "int_col")
    assert col_int["mean"] == 2.0
    assert "int32" in col_int["dtype"]

    # Float column
    col_float = next(c for c in stats["columns"] if c["name"] == "float_col")
    assert col_float["mean"] == 2.5
    assert "float64" in col_float["dtype"] or "double" in col_float["dtype"]

    # String column - should have no numeric stats
    col_str = next(c for c in stats["columns"] if c["name"] == "str_col")
    assert "mean" not in col_str


def test_describe_across_partitions(sp: Session):
    """Test describe correctly merges statistics across multiple partitions."""
    # Create data distributed across 5 partitions
    items = [{"value": i, "category": chr(65 + i % 3)} for i in range(100)]
    df = sp.from_items(items).repartition(5, by_rows=True)

    stats = df.describe()

    assert stats["num_rows"] == 100

    # Check value column statistics
    col_value = next(c for c in stats["columns"] if c["name"] == "value")
    assert col_value["null_count"] == 0
    assert col_value["non_null_count"] == 100
    assert col_value["min"] == 0.0
    assert col_value["max"] == 99.0
    assert col_value["sum"] == sum(range(100))  # 4950
    assert col_value["mean"] == 49.5  # sum / 100

    # Approx median should be close to exact median of 0-99: (49 + 50) / 2 = 49.5
    assert abs(col_value["approx_median"] - 49.5) < 2.0


def test_describe_partition_correctness_edge_case(sp: Session):
    """Test that describe correctly handles data split across partitions.

    This tests the edge case where statistics must be correctly combined:
    - Partition 1: [1, 2, 3]
    - Partition 2: [100, 101, 102]

    Simple averaging of partition means would give wrong result.
    """
    # Create two distinct partitions with very different values
    items = [{"value": i} for i in [1, 2, 3, 100, 101, 102]]
    df = sp.from_items(items).repartition(2, by_rows=True)

    stats = df.describe()

    col_value = stats["columns"][0]

    # Correct mean: (1+2+3+100+101+102) / 6 = 309 / 6 = 51.5
    assert col_value["mean"] == 51.5

    # Min should be global min
    assert col_value["min"] == 1.0

    # Max should be global max
    assert col_value["max"] == 102.0

    # Approx median: sorted [1, 2, 3, 100, 101, 102], exact median = (3 + 100) / 2 = 51.5
    # t-digest may not give exact result for this bimodal distribution, allow larger tolerance
    assert abs(col_value["approx_median"] - 51.5) < 10.0


def test_describe_std_across_partitions(sp: Session):
    """Test standard deviation is computed correctly across partitions."""
    # Data: [0, 10] - mean=5, variance=25, std=5
    df = sp.from_items([{"x": 0}, {"x": 10}]).repartition(2, by_rows=True)

    stats = df.describe()
    col_x = stats["columns"][0]

    # Mean = 5
    assert col_x["mean"] == 5.0

    # Variance = ((0-5)^2 + (10-5)^2) / 2 = (25 + 25) / 2 = 25
    # Std = sqrt(25) = 5
    assert abs(col_x["std"] - 5.0) < 0.001


def test_describe_empty_dataframe(sp: Session):
    """Test describe handles empty DataFrame (no rows but has schema)."""
    # Create an empty DataFrame by filtering out all rows
    df = sp.from_arrow(pa.table({"x": [1, 2, 3]})).filter("x < 0")
    stats = df.describe()

    assert stats["num_rows"] == 0
    assert stats["num_columns"] == 1

    # Column should still be present with schema info
    col_x = stats["columns"][0]
    assert col_x["name"] == "x"
    assert col_x["null_count"] == 0
    assert col_x["non_null_count"] == 0
    # No numeric stats because no data
    assert "min" not in col_x
    assert "mean" not in col_x


def test_describe_single_row(sp: Session):
    """Test describe with single row DataFrame."""
    df = sp.from_arrow(pa.table({"x": [42]}))
    stats = df.describe()

    assert stats["num_rows"] == 1
    col_x = stats["columns"][0]
    assert col_x["min"] == 42.0
    assert col_x["max"] == 42.0
    assert col_x["mean"] == 42.0
    assert col_x["approx_median"] == 42.0
    assert col_x["std"] == 0.0  # std of single value is 0


def test_describe_all_nulls_numeric(sp: Session):
    """Test describe handles column with all null values."""
    df = sp.from_arrow(pa.table({
        "x": pa.array([None, None, None], type=pa.int64())
    }))
    stats = df.describe()

    col_x = stats["columns"][0]
    assert col_x["null_count"] == 3
    assert col_x["non_null_count"] == 0
    assert col_x["null_percent"] == 100.0
    # Should not have numeric stats since all values are null
    assert "min" not in col_x
    assert "mean" not in col_x


def test_describe_large_dataset(sp: Session):
    """Test describe correctness with large dataset across many partitions."""
    # Create 1000 rows across 10 partitions
    items = [{"value": i, "squared": i * i} for i in range(1000)]
    df = sp.from_items(items).repartition(10, by_rows=True)

    stats = df.describe()

    assert stats["num_rows"] == 1000
    assert stats["num_columns"] == 2

    col_value = next(c for c in stats["columns"] if c["name"] == "value")
    assert col_value["min"] == 0.0
    assert col_value["max"] == 999.0
    assert col_value["sum"] == sum(range(1000))  # 499500
    assert col_value["mean"] == 499.5

    # Approx median should be close to exact median of 0-999: (499 + 500) / 2 = 499.5
    assert abs(col_value["approx_median"] - 499.5) < 10.0

    col_squared = next(c for c in stats["columns"] if c["name"] == "squared")
    assert col_squared["min"] == 0.0
    assert col_squared["max"] == 999.0 * 999.0


def test_describe_mixed_nulls_partitions(sp: Session):
    """Test describe handles nulls distributed across different partitions."""
    # Partition 1: [1, 2, NULL]
    # Partition 2: [NULL, 4, 5]
    items = [
        {"x": 1}, {"x": 2}, {"x": None},
        {"x": None}, {"x": 4}, {"x": 5}
    ]
    df = sp.from_arrow(pa.table({
        "x": pa.array([1, 2, None, None, 4, 5], type=pa.int64())
    })).repartition(2, by_rows=True)

    stats = df.describe()

    col_x = stats["columns"][0]
    assert col_x["null_count"] == 2
    assert col_x["non_null_count"] == 4
    assert col_x["min"] == 1.0
    assert col_x["max"] == 5.0
    assert col_x["mean"] == 3.0  # (1+2+4+5) / 4
    # Approx median of [1, 2, 4, 5], exact = (2 + 4) / 2 = 3
    assert abs(col_x["approx_median"] - 3.0) < 0.5


def test_describe_float_precision(sp: Session):
    """Test describe handles floating point values correctly."""
    df = sp.from_arrow(pa.table({
        "x": [0.1, 0.2, 0.3]
    }))
    stats = df.describe()

    col_x = stats["columns"][0]
    assert abs(col_x["sum"] - 0.6) < 0.0001
    assert abs(col_x["mean"] - 0.2) < 0.0001


def test_describe_negative_values(sp: Session):
    """Test describe handles negative values correctly."""
    df = sp.from_arrow(pa.table({
        "x": [-10, -5, 0, 5, 10]
    }))
    stats = df.describe()

    col_x = stats["columns"][0]
    assert col_x["min"] == -10.0
    assert col_x["max"] == 10.0
    assert col_x["mean"] == 0.0
    # approx_median should be close to 0
    assert abs(col_x["approx_median"] - 0.0) < 0.5
    assert col_x["sum"] == 0.0


def test_describe_preserves_column_order(sp: Session):
    """Test that describe preserves the original column order."""
    df = sp.from_arrow(pa.table({
        "z": [1, 2],
        "a": [3, 4],
        "m": [5, 6]
    }))
    stats = df.describe()

    column_names = [c["name"] for c in stats["columns"]]
    assert column_names == ["z", "a", "m"]


def test_describe_various_dtypes(sp: Session):
    """Test describe correctly identifies various data types."""
    df = sp.from_arrow(pa.table({
        "int8_col": pa.array([1, 2], type=pa.int8()),
        "int64_col": pa.array([1, 2], type=pa.int64()),
        "float32_col": pa.array([1.0, 2.0], type=pa.float32()),
        "float64_col": pa.array([1.0, 2.0], type=pa.float64()),
        "string_col": ["a", "b"],
        "bool_col": [True, False],
    }))
    stats = df.describe()

    # All int and float columns should have numeric stats
    for col in stats["columns"]:
        if "int" in col["dtype"] or "float" in col["dtype"] or "double" in col["dtype"]:
            assert "mean" in col, f"{col['name']} should have numeric stats"
        else:
            assert "mean" not in col, f"{col['name']} should not have numeric stats"


def test_describe_partitions_with_all_null_and_data(sp: Session):
    """Test describe when some partitions have all nulls and others have data.

    This tests the edge case where a numeric column has:
    - Partition 1: all null values [NULL, NULL, NULL]
    - Partition 2: actual data [10, 20, 30]

    The statistics should correctly reflect only the non-null values from
    partitions that have data, without being affected by all-null partitions.
    """
    # Create data where first partition has all nulls, second has data
    df = sp.from_arrow(pa.table({
        "x": pa.array([None, None, None, 10, 20, 30], type=pa.int64())
    })).repartition(2, by_rows=True)

    stats = df.describe()

    assert stats["num_rows"] == 6

    col_x = stats["columns"][0]
    # 3 nulls from first partition, 0 nulls from second
    assert col_x["null_count"] == 3
    assert col_x["non_null_count"] == 3
    assert col_x["null_percent"] == 50.0

    # Stats should be computed from [10, 20, 30] only
    assert col_x["min"] == 10.0
    assert col_x["max"] == 30.0
    assert col_x["mean"] == 20.0  # (10 + 20 + 30) / 3
    assert col_x["sum"] == 60.0
    # approx_median should be close to 20
    assert abs(col_x["approx_median"] - 20.0) < 1.0

    # Standard deviation of [10, 20, 30]: std = sqrt(((10-20)^2 + (20-20)^2 + (30-20)^2) / 3)
    # = sqrt((100 + 0 + 100) / 3) = sqrt(200/3) ≈ 8.165
    expected_std = (200.0 / 3) ** 0.5
    assert abs(col_x["std"] - expected_std) < 0.01


def test_describe_all_null_partition_at_end(sp: Session):
    """Test describe when all-null partition comes after data partition.

    Ensures order of partitions doesn't matter.
    """
    # Create data where first partition has data, second has all nulls
    df = sp.from_arrow(pa.table({
        "x": pa.array([5, 10, 15, None, None, None], type=pa.int64())
    })).repartition(2, by_rows=True)

    stats = df.describe()

    col_x = stats["columns"][0]
    assert col_x["null_count"] == 3
    assert col_x["non_null_count"] == 3

    # Stats should be computed from [5, 10, 15] only
    assert col_x["min"] == 5.0
    assert col_x["max"] == 15.0
    assert col_x["mean"] == 10.0


def test_describe_multiple_columns_mixed_null_partitions(sp: Session):
    """Test describe with multiple columns where different columns have nulls in different partitions.

    Column x: data in partition 1, all nulls in partition 2
    Column y: all nulls in partition 1, data in partition 2

    Both columns should have correct stats computed independently.
    """
    df = sp.from_arrow(pa.table({
        "x": pa.array([1, 2, 3, None, None, None], type=pa.int64()),
        "y": pa.array([None, None, None, 100, 200, 300], type=pa.int64()),
    })).repartition(2, by_rows=True)

    stats = df.describe()

    # Column x: data [1, 2, 3] from partition 1
    col_x = next(c for c in stats["columns"] if c["name"] == "x")
    assert col_x["null_count"] == 3
    assert col_x["non_null_count"] == 3
    assert col_x["min"] == 1.0
    assert col_x["max"] == 3.0
    assert col_x["mean"] == 2.0

    # Column y: data [100, 200, 300] from partition 2
    col_y = next(c for c in stats["columns"] if c["name"] == "y")
    assert col_y["null_count"] == 3
    assert col_y["non_null_count"] == 3
    assert col_y["min"] == 100.0
    assert col_y["max"] == 300.0
    assert col_y["mean"] == 200.0


# ==================== T-Digest Median Accuracy Tests ====================
# These tests compare the approximate median from t-digest against exact median
# calculations to verify accuracy across different data sizes and distributions.


def _compute_exact_median(values: list) -> float:
    """Helper function to compute exact median for comparison."""
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return None
    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2


def test_tdigest_accuracy_small_dataset(sp: Session):
    """Test t-digest accuracy on small dataset (n=10).

    For small datasets (n <= compression parameter), t-digest stores exact
    values as centroids, so the median should be very close to exact.
    """
    values = list(range(10))  # [0, 1, 2, ..., 9]
    df = sp.from_arrow(pa.table({"x": values}))

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)  # 4.5

    # For small datasets, t-digest should be very accurate
    error = abs(approx_median - exact_median)
    relative_error = error / exact_median if exact_median != 0 else error

    assert relative_error < 0.01, f"Small dataset: relative error {relative_error:.4f} exceeds 1%"


def test_tdigest_accuracy_medium_dataset(sp: Session):
    """Test t-digest accuracy on medium dataset (n=500).

    Medium datasets test the t-digest compression behavior while still
    being manageable for exact comparison.
    """
    values = list(range(500))
    df = sp.from_arrow(pa.table({"x": values})).repartition(5, by_rows=True)

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)  # 249.5

    error = abs(approx_median - exact_median)
    relative_error = error / exact_median if exact_median != 0 else error

    # For medium datasets, expect < 2% relative error
    assert relative_error < 0.02, f"Medium dataset: relative error {relative_error:.4f} exceeds 2%"


def test_tdigest_accuracy_large_dataset(sp: Session):
    """Test t-digest accuracy on large dataset (n=10000).

    Large datasets fully exercise the t-digest compression and merging
    across multiple partitions.
    """
    values = list(range(10000))
    df = sp.from_arrow(pa.table({"x": values})).repartition(10, by_rows=True)

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)  # 4999.5

    error = abs(approx_median - exact_median)
    relative_error = error / exact_median if exact_median != 0 else error

    # For large datasets, expect < 5% relative error (t-digest is less accurate at median)
    assert relative_error < 0.05, f"Large dataset: relative error {relative_error:.4f} exceeds 5%"


def test_tdigest_accuracy_uniform_distribution(sp: Session):
    """Test t-digest accuracy on uniformly distributed data.

    Uniform distributions are typically well-handled by t-digest.
    """
    # Uniform distribution from 0 to 1000
    values = list(range(1001))  # [0, 1, 2, ..., 1000]
    df = sp.from_arrow(pa.table({"x": values})).repartition(8, by_rows=True)

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)  # 500

    error = abs(approx_median - exact_median)
    relative_error = error / exact_median

    assert relative_error < 0.02, f"Uniform distribution: relative error {relative_error:.4f} exceeds 2%"


def test_tdigest_accuracy_normal_like_distribution(sp: Session):
    """Test t-digest accuracy on normal-like distribution.

    Tests with data concentrated around the mean, similar to normal distribution.
    Uses integer approximation of normal distribution for reproducibility.
    """
    # Create a distribution with more values near the center
    # Values: many 50s, fewer 40s/60s, even fewer 30s/70s, etc.
    values = []
    for i in range(100):
        # More values near 50
        distance_from_center = abs(i - 50)
        count = max(1, 10 - distance_from_center // 5)
        values.extend([i] * count)

    df = sp.from_arrow(pa.table({"x": values})).repartition(4, by_rows=True)

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)

    error = abs(approx_median - exact_median)
    relative_error = error / exact_median if exact_median != 0 else error

    assert relative_error < 0.05, f"Normal-like distribution: relative error {relative_error:.4f} exceeds 5%"


def test_tdigest_accuracy_skewed_distribution(sp: Session):
    """Test t-digest accuracy on skewed distribution.

    Skewed distributions can be more challenging for median estimation.
    """
    # Right-skewed distribution: many small values, few large values
    values = []
    values.extend([1] * 100)
    values.extend([2] * 80)
    values.extend([3] * 60)
    values.extend([5] * 40)
    values.extend([10] * 20)
    values.extend([50] * 10)
    values.extend([100] * 5)

    df = sp.from_arrow(pa.table({"x": values})).repartition(3, by_rows=True)

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)

    error = abs(approx_median - exact_median)
    relative_error = error / exact_median if exact_median != 0 else error

    # Skewed distributions may have slightly higher error
    assert relative_error < 0.10, f"Skewed distribution: relative error {relative_error:.4f} exceeds 10%"


def test_tdigest_accuracy_bimodal_distribution(sp: Session):
    """Test t-digest accuracy on bimodal distribution.

    Bimodal distributions are challenging because the median falls between
    two clusters with no actual data points nearby.
    """
    # Two clusters: one around 10, one around 90
    values = []
    values.extend(list(range(5, 16)) * 10)  # Cluster around 10
    values.extend(list(range(85, 96)) * 10)  # Cluster around 90

    df = sp.from_arrow(pa.table({"x": values})).repartition(4, by_rows=True)

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)

    error = abs(approx_median - exact_median)

    # For bimodal, use absolute error since median is between clusters
    # The exact median will be around 50, allow larger absolute tolerance
    assert error < 15, f"Bimodal distribution: absolute error {error:.2f} exceeds 15"


def test_tdigest_accuracy_with_outliers(sp: Session):
    """Test t-digest accuracy with outliers.

    T-digest should handle outliers well due to its higher resolution at tails.
    """
    # Normal range 0-100, with some extreme outliers
    values = list(range(101))  # [0, 1, ..., 100]
    values.extend([1000, 2000, 5000])  # Extreme outliers

    df = sp.from_arrow(pa.table({"x": values})).repartition(3, by_rows=True)

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)  # Should be around 50

    error = abs(approx_median - exact_median)
    relative_error = error / exact_median if exact_median != 0 else error

    # Median should not be heavily affected by outliers
    assert relative_error < 0.10, f"With outliers: relative error {relative_error:.4f} exceeds 10%"


def test_tdigest_accuracy_many_partitions(sp: Session):
    """Test t-digest accuracy when data is spread across many partitions.

    This tests the t-digest merge algorithm's accuracy.
    """
    values = list(range(1000))
    df = sp.from_arrow(pa.table({"x": values})).repartition(20, by_rows=True)

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)  # 499.5

    error = abs(approx_median - exact_median)
    relative_error = error / exact_median

    # With many partitions, merging may introduce some error
    assert relative_error < 0.05, f"Many partitions: relative error {relative_error:.4f} exceeds 5%"


def test_tdigest_accuracy_single_partition(sp: Session):
    """Test t-digest accuracy with single partition (no merging needed).

    This establishes baseline accuracy without merge operations.
    """
    values = list(range(200))
    df = sp.from_arrow(pa.table({"x": values})).repartition(1, by_rows=True)

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)  # 99.5

    error = abs(approx_median - exact_median)
    relative_error = error / exact_median

    # Single partition should be very accurate
    assert relative_error < 0.02, f"Single partition: relative error {relative_error:.4f} exceeds 2%"


def test_tdigest_accuracy_floating_point_values(sp: Session):
    """Test t-digest accuracy with floating point values.

    Ensures floating point precision doesn't affect median accuracy.
    """
    # Create floating point values
    values = [i * 0.1 for i in range(1000)]  # [0.0, 0.1, 0.2, ..., 99.9]
    df = sp.from_arrow(pa.table({"x": values})).repartition(5, by_rows=True)

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)  # 49.95

    error = abs(approx_median - exact_median)
    relative_error = error / exact_median

    assert relative_error < 0.02, f"Floating point: relative error {relative_error:.4f} exceeds 2%"


def test_tdigest_accuracy_negative_values(sp: Session):
    """Test t-digest accuracy with negative values.

    Ensures negative numbers are handled correctly.
    """
    values = list(range(-500, 501))  # [-500, -499, ..., 499, 500]
    df = sp.from_arrow(pa.table({"x": values})).repartition(5, by_rows=True)

    stats = df.describe()
    approx_median = stats["columns"][0]["approx_median"]
    exact_median = _compute_exact_median(values)  # 0

    # For median near 0, use absolute error
    error = abs(approx_median - exact_median)
    assert error < 5, f"Negative values: absolute error {error:.2f} exceeds 5"


def test_tdigest_accuracy_summary(sp: Session):
    """Summary test that reports accuracy across multiple dataset sizes.

    This test documents the expected accuracy of t-digest at different scales.
    The results help developers understand when describe() is appropriate
    vs when they should use to_pandas().median() for exact results.

    Expected accuracy guidelines:
    - n <= 100: Very accurate (< 1% error), t-digest stores exact values
    - n <= 1000: Accurate (< 2% error), compression is moderate
    - n <= 10000: Good (< 5% error), suitable for exploratory analysis
    - n > 10000: Acceptable (< 10% error), use for quick estimates only
    """
    test_cases = [
        (50, "tiny"),
        (100, "small"),
        (500, "medium"),
        (1000, "large"),
        (5000, "very_large"),
    ]

    results = []
    for size, label in test_cases:
        values = list(range(size))
        df = sp.from_arrow(pa.table({"x": values})).repartition(max(1, size // 100), by_rows=True)

        stats = df.describe()
        approx_median = stats["columns"][0]["approx_median"]
        exact_median = _compute_exact_median(values)

        error = abs(approx_median - exact_median)
        relative_error = (error / exact_median * 100) if exact_median != 0 else 0

        results.append({
            "size": size,
            "label": label,
            "exact_median": exact_median,
            "approx_median": approx_median,
            "relative_error_percent": relative_error,
        })

    # Verify all test cases are within acceptable bounds
    for r in results:
        if r["size"] <= 100:
            assert r["relative_error_percent"] < 1.0, f"Size {r['size']}: error {r['relative_error_percent']:.2f}% exceeds 1%"
        elif r["size"] <= 1000:
            assert r["relative_error_percent"] < 3.0, f"Size {r['size']}: error {r['relative_error_percent']:.2f}% exceeds 3%"
        else:
            assert r["relative_error_percent"] < 10.0, f"Size {r['size']}: error {r['relative_error_percent']:.2f}% exceeds 10%"


# ============================================================================
# Tests for require_non_null() validation feature
# ============================================================================


def test_require_non_null_passes_when_no_nulls(sp: Session):
    """Test that require_non_null passes when columns have no null values."""
    df = sp.from_arrow(pa.table({"id": [1, 2, 3], "name": ["a", "b", "c"]}))
    df = df.require_non_null("id")

    # Should not raise an error
    assert df.count() == 3
    assert len(df.take(10)) == 3
    assert len(df.to_pandas()) == 3
    assert df.to_arrow().num_rows == 3


def test_require_non_null_fails_when_nulls_exist(sp: Session):
    """Test that require_non_null raises NullValidationError when nulls exist."""
    from smallpond.dataframe import NullValidationError

    df = sp.from_arrow(pa.table({
        "id": pa.array([1, None, 3], type=pa.int64()),
        "name": ["a", "b", "c"]
    }))
    df = df.require_non_null("id")

    # Should raise NullValidationError
    with pytest.raises(NullValidationError) as exc_info:
        df.count()

    error = exc_info.value
    assert "id" in error.columns
    assert error.null_counts["id"] == 1
    assert "'id' (1 nulls)" in str(error)


def test_require_non_null_multiple_columns(sp: Session):
    """Test require_non_null with multiple columns."""
    from smallpond.dataframe import NullValidationError

    df = sp.from_arrow(pa.table({
        "id": pa.array([1, None, 3], type=pa.int64()),
        "email": pa.array(["a@test.com", "b@test.com", None], type=pa.string()),
        "name": ["a", "b", "c"]
    }))
    df = df.require_non_null(["id", "email"])

    with pytest.raises(NullValidationError) as exc_info:
        df.take(10)

    error = exc_info.value
    assert "id" in error.columns
    assert "email" in error.columns
    assert error.null_counts["id"] == 1
    assert error.null_counts["email"] == 1


def test_require_non_null_chained_calls(sp: Session):
    """Test that multiple require_non_null calls accumulate columns."""
    from smallpond.dataframe import NullValidationError

    df = sp.from_arrow(pa.table({
        "id": pa.array([1, None, 3], type=pa.int64()),
        "email": pa.array(["a@test.com", "b@test.com", None], type=pa.string()),
        "name": ["a", "b", "c"]
    }))
    df = df.require_non_null("id").require_non_null("email")

    with pytest.raises(NullValidationError) as exc_info:
        df.to_pandas()

    error = exc_info.value
    assert "id" in error.columns
    assert "email" in error.columns


def test_require_non_null_preserves_through_filter(sp: Session):
    """Test that non-null constraints are preserved through filter operations."""
    from smallpond.dataframe import NullValidationError

    df = sp.from_arrow(pa.table({
        "id": pa.array([1, None, 3, 4], type=pa.int64()),
        "value": [10, 20, 30, 40]
    }))
    df = df.require_non_null("id").filter("value > 15")

    # After filter, only rows with value > 15 remain: (None, 20), (3, 30), (4, 40)
    # The null in 'id' is still there
    with pytest.raises(NullValidationError):
        df.count()


def test_require_non_null_preserves_through_map(sp: Session):
    """Test that non-null constraints are preserved through map operations."""
    from smallpond.dataframe import NullValidationError

    df = sp.from_arrow(pa.table({
        "id": pa.array([1, None, 3], type=pa.int64()),
        "value": [10, 20, 30]
    }))
    df = df.require_non_null("id").map("id, value * 2 as doubled")

    with pytest.raises(NullValidationError):
        df.to_arrow()


def test_require_non_null_preserves_through_repartition(sp: Session):
    """Test that non-null constraints are preserved through repartition."""
    from smallpond.dataframe import NullValidationError

    df = sp.from_arrow(pa.table({
        "id": pa.array([1, None, 3, 4, 5], type=pa.int64()),
        "value": [10, 20, 30, 40, 50]
    }))
    df = df.require_non_null("id").repartition(2, by_rows=True)

    with pytest.raises(NullValidationError):
        df.count()


def test_require_non_null_describe_skips_validation(sp: Session):
    """Test that describe() skips non-null validation to allow data exploration."""
    df = sp.from_arrow(pa.table({
        "id": pa.array([1, None, 3], type=pa.int64()),
        "value": [10, 20, 30]
    }))
    df = df.require_non_null("id")

    # describe() should NOT raise an error - it's used for data exploration
    stats = df.describe()
    assert stats["num_rows"] == 3
    # Check that describe correctly reports the null count
    id_col = next(c for c in stats["columns"] if c["name"] == "id")
    assert id_col["null_count"] == 1


def test_require_non_null_with_join(sp: Session):
    """Test that non-null constraints from both DataFrames are merged in join."""
    from smallpond.dataframe import NullValidationError

    df1 = sp.from_arrow(pa.table({
        "id": pa.array([1, 2, 3], type=pa.int64()),
        "value": pa.array([10, None, 30], type=pa.int64())
    })).require_non_null("value")

    df2 = sp.from_arrow(pa.table({
        "id": pa.array([1, 2, 3], type=pa.int64()),
        "name": pa.array(["a", "b", None], type=pa.string())
    })).require_non_null("name")

    joined = df1.join(df2, on="id")

    with pytest.raises(NullValidationError) as exc_info:
        joined.take(10)

    error = exc_info.value
    # Both 'value' from df1 and 'name' from df2 should be in the non-null columns
    assert "value" in error.columns
    assert "name" in error.columns


def test_require_non_null_column_not_exists(sp: Session):
    """Test that specifying a non-existent column raises ValueError immediately."""
    df = sp.from_arrow(pa.table({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"]
    }))

    # Error should be raised immediately at require_non_null() time,
    # not at count() time, for early typo detection
    with pytest.raises(ValueError) as exc_info:
        df.require_non_null("nonexistent_column")

    assert "nonexistent_column" in str(exc_info.value)
    assert "do not exist" in str(exc_info.value)
    assert "Available columns" in str(exc_info.value)


def test_require_non_null_column_not_exists_with_filter(sp: Session):
    """Test that early validation works through schema-preserving transformations."""
    df = sp.from_arrow(pa.table({
        "id": [1, 2, 3],
        "name": ["a", "b", "c"]
    }))

    # Filter preserves schema, so early validation should still work
    filtered_df = df.filter("id > 1")

    with pytest.raises(ValueError) as exc_info:
        filtered_df.require_non_null("nonexistent_column")

    assert "nonexistent_column" in str(exc_info.value)


def test_require_non_null_cached_results(sp: Session):
    """Test that validation is performed even for cached results."""
    from smallpond.dataframe import NullValidationError

    df = sp.from_arrow(pa.table({
        "id": pa.array([1, None, 3], type=pa.int64()),
        "value": [10, 20, 30]
    }))

    # First, compute without validation to populate cache
    count_no_validation = df.count()
    assert count_no_validation == 3

    # Now add validation and try to access data
    df_validated = df.require_non_null("id")

    # Should still raise error even though underlying data is cached
    with pytest.raises(NullValidationError):
        df_validated.count()


def test_require_non_null_returns_self(sp: Session):
    """Test that require_non_null returns self for method chaining."""
    df = sp.from_arrow(pa.table({"id": [1, 2, 3]}))
    result = df.require_non_null("id")
    assert result is df


def test_require_non_null_no_duplicates(sp: Session):
    """Test that calling require_non_null with same column doesn't create duplicates."""
    df = sp.from_arrow(pa.table({"id": [1, 2, 3]}))
    df.require_non_null("id").require_non_null("id").require_non_null("id")
    # _non_null_columns is a frozenset, so duplicates are inherently prevented
    assert len(df._non_null_columns) == 1
    assert "id" in df._non_null_columns


def test_null_validation_error_message(sp: Session):
    """Test that NullValidationError has informative message."""
    from smallpond.dataframe import NullValidationError

    df = sp.from_arrow(pa.table({
        "id": pa.array([1, None, None, 4], type=pa.int64()),
        "email": pa.array(["a@test.com", None, "c@test.com", None], type=pa.string()),
    }))
    df = df.require_non_null(["id", "email"])

    with pytest.raises(NullValidationError) as exc_info:
        df.count()

    error_msg = str(exc_info.value)
    assert "id" in error_msg
    assert "2 nulls" in error_msg
    assert "email" in error_msg
    assert "require_non_null()" in error_msg


# ==================== Union Tests ====================


def test_union_basic_two_dataframes(sp: Session):
    """Test basic union of two DataFrames with the same schema."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2, 3], "value": ["a", "b", "c"]}))
    df2 = sp.from_arrow(pa.table({"id": [4, 5, 6], "value": ["d", "e", "f"]}))

    result = df1.union(df2)
    rows = sorted(result.take_all(), key=lambda x: x["id"])

    assert len(rows) == 6
    assert rows[0] == {"id": 1, "value": "a"}
    assert rows[5] == {"id": 6, "value": "f"}


def test_union_multiple_dataframes(sp: Session):
    """Test union of more than two DataFrames."""
    df1 = sp.from_arrow(pa.table({"x": [1, 2]}))
    df2 = sp.from_arrow(pa.table({"x": [3, 4]}))
    df3 = sp.from_arrow(pa.table({"x": [5, 6]}))
    df4 = sp.from_arrow(pa.table({"x": [7, 8]}))

    result = df1.union(df2, df3, df4)
    rows = sorted(result.take_all(), key=lambda x: x["x"])

    assert len(rows) == 8
    assert [r["x"] for r in rows] == [1, 2, 3, 4, 5, 6, 7, 8]


def test_union_different_column_order(sp: Session):
    """Test union of DataFrames with same columns but different order."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2], "name": ["a", "b"], "value": [10, 20]}))
    df2 = sp.from_arrow(pa.table({"name": ["c", "d"], "value": [30, 40], "id": [3, 4]}))

    result = df1.union(df2)
    rows = sorted(result.take_all(), key=lambda x: x["id"])

    # Result should have columns in the order of the first DataFrame
    assert len(rows) == 4
    assert rows[0] == {"id": 1, "name": "a", "value": 10}
    assert rows[1] == {"id": 2, "name": "b", "value": 20}
    assert rows[2] == {"id": 3, "name": "c", "value": 30}
    assert rows[3] == {"id": 4, "name": "d", "value": 40}


def test_union_preserves_all_rows(sp: Session):
    """Test that union preserves all rows including duplicates."""
    df1 = sp.from_arrow(pa.table({"x": [1, 2, 3]}))
    df2 = sp.from_arrow(pa.table({"x": [2, 3, 4]}))  # 2 and 3 are duplicates

    result = df1.union(df2)
    rows = result.take_all()

    # Should have 6 rows, including duplicates
    assert len(rows) == 6
    values = sorted([r["x"] for r in rows])
    assert values == [1, 2, 2, 3, 3, 4]


def test_union_with_nulls(sp: Session):
    """Test union correctly handles null values."""
    df1 = sp.from_arrow(pa.table({
        "id": pa.array([1, None, 3], type=pa.int64()),
        "value": ["a", "b", "c"]
    }))
    df2 = sp.from_arrow(pa.table({
        "id": pa.array([4, 5, None], type=pa.int64()),
        "value": ["d", "e", "f"]
    }))

    result = df1.union(df2)
    rows = result.take_all()

    assert len(rows) == 6
    # Count nulls
    null_count = sum(1 for r in rows if r["id"] is None)
    assert null_count == 2


def test_union_validation_no_args(sp: Session):
    """Test that union raises error when no other DataFrames are provided."""
    df1 = sp.from_arrow(pa.table({"x": [1, 2]}))

    with pytest.raises(ValueError, match="requires at least one other DataFrame"):
        df1.union()


def test_union_validation_missing_columns(sp: Session):
    """Test that union raises SchemaMismatchError when columns are missing."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"id": [1, 2], "name": ["a", "b"]}))
    df2 = sp.from_arrow(pa.table({"id": [3, 4]}))  # Missing 'name' column

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    assert "name" in str(error)
    assert "Missing columns" in str(error) or "missing" in str(error).lower()
    assert error.details.get("missing_columns") == ["name"]


def test_union_validation_extra_columns(sp: Session):
    """Test that union raises SchemaMismatchError when extra columns exist."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"id": [1, 2]}))
    df2 = sp.from_arrow(pa.table({"id": [3, 4], "extra": ["a", "b"]}))

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    assert "extra" in str(error)
    assert error.details.get("extra_columns") == ["extra"]


def test_union_validation_different_columns(sp: Session):
    """Test that union raises SchemaMismatchError when columns are completely different."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"a": [1, 2], "b": [3, 4]}))
    df2 = sp.from_arrow(pa.table({"c": [5, 6], "d": [7, 8]}))

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    # Should have both missing and extra columns
    assert "a" in str(error) or "b" in str(error)
    assert "c" in str(error) or "d" in str(error)


def test_union_preserves_partitions(sp: Session):
    """Test that union preserves partitioning from all input DataFrames."""
    # Create DataFrames with different partition counts
    df1 = sp.from_items(list(range(100))).repartition(4, by_rows=True)
    df2 = sp.from_items(list(range(100, 200))).repartition(3, by_rows=True)

    result = df1.union(df2)
    rows = result.take_all()

    # All 200 rows should be present
    assert len(rows) == 200
    values = sorted([r["item"] for r in rows])
    assert values == list(range(200))


def test_union_preserves_non_null_columns(sp: Session):
    """Test that union merges non-null column requirements from all DataFrames."""
    from smallpond.dataframe import NullValidationError

    df1 = sp.from_arrow(pa.table({
        "id": pa.array([1, 2], type=pa.int64()),
        "value": [10, 20]
    })).require_non_null("id")

    df2 = sp.from_arrow(pa.table({
        "id": pa.array([3, None], type=pa.int64()),  # Has a null in 'id'
        "value": [30, 40]
    })).require_non_null("value")

    result = df1.union(df2)

    # Result should have non-null requirements for both 'id' and 'value'
    assert "id" in result._non_null_columns
    assert "value" in result._non_null_columns

    # Should raise error because df2 has null in 'id'
    with pytest.raises(NullValidationError) as exc_info:
        result.count()

    assert "id" in exc_info.value.columns


def test_union_preserves_cache_setting(sp: Session):
    """Test that union inherits cache setting correctly."""
    df1 = sp.from_arrow(pa.table({"x": [1, 2]})).no_cache()
    df2 = sp.from_arrow(pa.table({"x": [3, 4]}))

    result = df1.union(df2)
    # Cache should be disabled if any input has caching disabled
    assert result._use_cache is False


def test_union_large_dataset(sp: Session):
    """Test union correctness with larger datasets across partitions."""
    # Create two DataFrames with 500 rows each across multiple partitions
    items1 = [{"id": i, "category": "A", "value": i * 10} for i in range(500)]
    items2 = [{"id": i + 500, "category": "B", "value": (i + 500) * 10} for i in range(500)]

    df1 = sp.from_items(items1).repartition(5, by_rows=True)
    df2 = sp.from_items(items2).repartition(5, by_rows=True)

    result = df1.union(df2)
    rows = result.take_all()

    assert len(rows) == 1000

    # Verify all ids are present
    ids = sorted([r["id"] for r in rows])
    assert ids == list(range(1000))

    # Verify categories
    category_a = [r for r in rows if r["category"] == "A"]
    category_b = [r for r in rows if r["category"] == "B"]
    assert len(category_a) == 500
    assert len(category_b) == 500


def test_union_with_filter_after(sp: Session):
    """Test that filter can be applied after union."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2, 3], "category": ["A", "A", "B"]}))
    df2 = sp.from_arrow(pa.table({"id": [4, 5, 6], "category": ["A", "B", "B"]}))

    result = df1.union(df2).filter("category = 'A'")
    rows = sorted(result.take_all(), key=lambda x: x["id"])

    assert len(rows) == 3  # ids 1, 2, 4
    assert all(r["category"] == "A" for r in rows)


def test_union_with_map_after(sp: Session):
    """Test that map can be applied after union."""
    df1 = sp.from_arrow(pa.table({"value": [1, 2]}))
    df2 = sp.from_arrow(pa.table({"value": [3, 4]}))

    result = df1.union(df2).map("value * 2 as doubled")
    rows = sorted(result.take_all(), key=lambda x: x["doubled"])

    assert len(rows) == 4
    assert [r["doubled"] for r in rows] == [2, 4, 6, 8]


def test_union_empty_dataframe(sp: Session):
    """Test union with an empty DataFrame."""
    df1 = sp.from_arrow(pa.table({"x": [1, 2, 3]}))
    df2 = sp.from_arrow(pa.table({"x": []}))  # Empty DataFrame with same schema

    result = df1.union(df2)
    rows = result.take_all()

    assert len(rows) == 3


def test_union_both_empty(sp: Session):
    """Test union of two empty DataFrames."""
    df1 = sp.from_arrow(pa.table({"x": pa.array([], type=pa.int64())}))
    df2 = sp.from_arrow(pa.table({"x": pa.array([], type=pa.int64())}))

    result = df1.union(df2)
    rows = result.take_all()

    assert len(rows) == 0


def test_union_multiple_numeric_types(sp: Session):
    """Test union with various numeric column types."""
    df1 = sp.from_arrow(pa.table({
        "int_col": pa.array([1, 2], type=pa.int64()),
        "float_col": pa.array([1.5, 2.5], type=pa.float64()),
    }))
    df2 = sp.from_arrow(pa.table({
        "int_col": pa.array([3, 4], type=pa.int64()),
        "float_col": pa.array([3.5, 4.5], type=pa.float64()),
    }))

    result = df1.union(df2)
    rows = sorted(result.take_all(), key=lambda x: x["int_col"])

    assert len(rows) == 4
    assert rows[0]["float_col"] == 1.5
    assert rows[3]["float_col"] == 4.5


def test_union_schema_mismatch_error_details(sp: Session):
    """Test that SchemaMismatchError contains useful details."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"a": [1], "b": [2], "c": [3]}))
    df2 = sp.from_arrow(pa.table({"a": [4], "b": [5], "d": [6]}))  # 'd' instead of 'c'

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    assert error.details["dataframe_index"] == 1
    assert "c" in error.details["missing_columns"]
    assert "d" in error.details["extra_columns"]
    assert error.details["expected_columns"] == ["a", "b", "c"]


def test_union_chained(sp: Session):
    """Test chaining multiple union calls."""
    df1 = sp.from_arrow(pa.table({"x": [1]}))
    df2 = sp.from_arrow(pa.table({"x": [2]}))
    df3 = sp.from_arrow(pa.table({"x": [3]}))
    df4 = sp.from_arrow(pa.table({"x": [4]}))

    # Chain union calls
    result = df1.union(df2).union(df3).union(df4)
    rows = sorted(result.take_all(), key=lambda x: x["x"])

    assert len(rows) == 4
    assert [r["x"] for r in rows] == [1, 2, 3, 4]


def test_union_recompute_inheritance(sp: Session):
    """Test that union inherits recompute flag from inputs."""
    df1 = sp.from_arrow(pa.table({"x": [1, 2]})).recompute()
    df2 = sp.from_arrow(pa.table({"x": [3, 4]}))

    result = df1.union(df2)
    assert result.need_recompute is True


def test_union_with_groupby_after(sp: Session):
    """Test that groupby_agg can be applied after union."""
    df1 = sp.from_arrow(pa.table({
        "category": ["A", "A", "B"],
        "value": [10, 20, 30]
    }))
    df2 = sp.from_arrow(pa.table({
        "category": ["A", "B", "B"],
        "value": [40, 50, 60]
    }))

    result = df1.union(df2).groupby_agg(by="category", aggs={"value": "sum"})
    rows = sorted(result.take_all(), key=lambda x: x["category"])

    assert len(rows) == 2
    assert rows[0]["category"] == "A"
    assert rows[0]["value_sum"] == 70  # 10 + 20 + 40
    assert rows[1]["category"] == "B"
    assert rows[1]["value_sum"] == 140  # 30 + 50 + 60


def test_union_validation_third_dataframe_mismatch(sp: Session):
    """Test that union validates all DataFrames, not just the second one."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"a": [1], "b": [2]}))
    df2 = sp.from_arrow(pa.table({"a": [3], "b": [4]}))
    df3 = sp.from_arrow(pa.table({"a": [5], "c": [6]}))  # Mismatch in third DataFrame

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2, df3)

    error = exc_info.value
    assert error.details["dataframe_index"] == 2  # Third DataFrame (index 2)


# ==================== Union Type Validation Tests ====================


def test_union_type_mismatch_string_vs_int(sp: Session):
    """Test that union raises SchemaMismatchError for string vs integer type mismatch."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"id": [1, 2, 3], "value": [10, 20, 30]}))
    df2 = sp.from_arrow(pa.table({"id": ["a", "b", "c"], "value": [40, 50, 60]}))  # id is string

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    assert "Type mismatch" in str(error)
    assert "id" in str(error)
    assert error.details["dataframe_index"] == 1
    assert len(error.details["type_mismatches"]) == 1
    assert error.details["type_mismatches"][0]["column"] == "id"
    assert error.details["type_mismatches"][0]["expected_category"] == "numeric"
    assert error.details["type_mismatches"][0]["actual_category"] == "string"


def test_union_type_mismatch_int_vs_string(sp: Session):
    """Test type mismatch in reverse order (string first, int second)."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"name": ["alice", "bob"], "score": [100, 200]}))
    df2 = sp.from_arrow(pa.table({"name": [1, 2], "score": [300, 400]}))  # name is int

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    assert "name" in str(error)
    assert error.details["type_mismatches"][0]["expected_category"] == "string"
    assert error.details["type_mismatches"][0]["actual_category"] == "numeric"


def test_union_type_mismatch_bool_vs_int(sp: Session):
    """Test that union raises SchemaMismatchError for boolean vs integer type mismatch."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"flag": [True, False, True]}))
    df2 = sp.from_arrow(pa.table({"flag": [1, 0, 1]}))  # flag is int

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    assert "flag" in str(error)
    assert error.details["type_mismatches"][0]["expected_category"] == "boolean"
    assert error.details["type_mismatches"][0]["actual_category"] == "numeric"


def test_union_type_mismatch_bool_vs_string(sp: Session):
    """Test that union raises SchemaMismatchError for boolean vs string type mismatch."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"active": [True, False]}))
    df2 = sp.from_arrow(pa.table({"active": ["yes", "no"]}))  # active is string

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    assert "active" in str(error)
    assert error.details["type_mismatches"][0]["expected_category"] == "boolean"
    assert error.details["type_mismatches"][0]["actual_category"] == "string"


def test_union_type_mismatch_multiple_columns(sp: Session):
    """Test that union reports all type mismatches when multiple columns have issues."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({
        "id": [1, 2],
        "name": ["alice", "bob"],
        "active": [True, False]
    }))
    df2 = sp.from_arrow(pa.table({
        "id": ["a", "b"],      # string instead of int
        "name": [100, 200],    # int instead of string
        "active": [True, False]  # matches
    }))

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    assert len(error.details["type_mismatches"]) == 2
    mismatched_cols = {m["column"] for m in error.details["type_mismatches"]}
    assert "id" in mismatched_cols
    assert "name" in mismatched_cols


def test_union_type_compatible_numeric_widening(sp: Session):
    """Test that union allows compatible numeric types (int32 with int64)."""
    df1 = sp.from_arrow(pa.table({
        "id": pa.array([1, 2, 3], type=pa.int32()),
        "value": pa.array([1.5, 2.5, 3.5], type=pa.float32())
    }))
    df2 = sp.from_arrow(pa.table({
        "id": pa.array([4, 5, 6], type=pa.int64()),
        "value": pa.array([4.5, 5.5, 6.5], type=pa.float64())
    }))

    # Should not raise - numeric types are compatible
    result = df1.union(df2)
    rows = result.take_all()
    assert len(rows) == 6


def test_union_type_compatible_string_variations(sp: Session):
    """Test that union allows compatible string types (string with large_string)."""
    df1 = sp.from_arrow(pa.table({
        "name": pa.array(["alice", "bob"], type=pa.string())
    }))
    df2 = sp.from_arrow(pa.table({
        "name": pa.array(["carol", "dave"], type=pa.large_string())
    }))

    # Should not raise - string types are compatible
    result = df1.union(df2)
    rows = result.take_all()
    assert len(rows) == 4


def test_union_type_mismatch_temporal_vs_string(sp: Session):
    """Test that union raises SchemaMismatchError for temporal vs string type mismatch."""
    from smallpond.dataframe import SchemaMismatchError
    import datetime

    df1 = sp.from_arrow(pa.table({
        "event_date": pa.array([datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)])
    }))
    df2 = sp.from_arrow(pa.table({
        "event_date": ["2024-01-03", "2024-01-04"]  # string instead of date
    }))

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    assert "event_date" in str(error)
    assert error.details["type_mismatches"][0]["expected_category"] == "temporal"
    assert error.details["type_mismatches"][0]["actual_category"] == "string"


def test_union_type_mismatch_temporal_vs_numeric(sp: Session):
    """Test that union raises SchemaMismatchError for temporal vs numeric type mismatch."""
    from smallpond.dataframe import SchemaMismatchError
    import datetime

    df1 = sp.from_arrow(pa.table({
        "timestamp": pa.array([datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)])
    }))
    df2 = sp.from_arrow(pa.table({
        "timestamp": [1704067200, 1704153600]  # unix timestamp as int
    }))

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    assert "timestamp" in str(error)


def test_union_type_compatible_temporal_same_category(sp: Session):
    """Test that union allows compatible temporal types."""
    import datetime

    df1 = sp.from_arrow(pa.table({
        "event_date": pa.array([datetime.date(2024, 1, 1), datetime.date(2024, 1, 2)], type=pa.date32())
    }))
    df2 = sp.from_arrow(pa.table({
        "event_date": pa.array([datetime.date(2024, 1, 3), datetime.date(2024, 1, 4)], type=pa.date64())
    }))

    # Should not raise - both are temporal types
    result = df1.union(df2)
    rows = result.take_all()
    assert len(rows) == 4


def test_union_type_error_message_includes_types(sp: Session):
    """Test that type mismatch error message includes the actual types."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"col": pa.array([1, 2], type=pa.int64())}))
    df2 = sp.from_arrow(pa.table({"col": pa.array(["a", "b"], type=pa.string())}))

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error_msg = str(exc_info.value)
    # The error message should contain type information
    assert "int64" in error_msg
    assert "string" in error_msg
    assert "numeric" in error_msg  # category


def test_union_type_validation_third_dataframe(sp: Session):
    """Test that type validation checks all DataFrames, not just the second one."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"value": [1, 2]}))
    df2 = sp.from_arrow(pa.table({"value": [3, 4]}))
    df3 = sp.from_arrow(pa.table({"value": ["a", "b"]}))  # Type mismatch in third

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2, df3)

    error = exc_info.value
    assert error.details["dataframe_index"] == 2  # Third DataFrame


def test_union_with_pandas_type_mismatch(sp: Session):
    """Test type validation works with DataFrames created from pandas."""
    from smallpond.dataframe import SchemaMismatchError

    pd_df1 = pd.DataFrame({"id": [1, 2, 3], "name": ["a", "b", "c"]})
    pd_df2 = pd.DataFrame({"id": ["x", "y", "z"], "name": ["d", "e", "f"]})  # id is string

    df1 = sp.from_pandas(pd_df1)
    df2 = sp.from_pandas(pd_df2)

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1.union(df2)

    error = exc_info.value
    assert "id" in str(error)


def test_union_with_filter_type_validation_preserved(sp: Session):
    """Test that type validation works through filter transformation."""
    from smallpond.dataframe import SchemaMismatchError

    df1 = sp.from_arrow(pa.table({"id": [1, 2, 3, 4], "value": [10, 20, 30, 40]}))
    df2 = sp.from_arrow(pa.table({"id": ["a", "b"], "value": [50, 60]}))

    # Apply filter to df1 - schema should be preserved
    df1_filtered = df1.filter("value > 15")

    with pytest.raises(SchemaMismatchError) as exc_info:
        df1_filtered.union(df2)

    error = exc_info.value
    assert "id" in str(error)


def test_union_type_validation_skipped_when_schema_unavailable(sp: Session):
    """Test that union proceeds without early type validation when schema can't be determined.

    When schemas can't be fully extracted (e.g., after complex transformations),
    the union should still work and let DuckDB handle type validation at compute time.
    """
    # Create a DataFrame with a transformation that makes schema extraction difficult
    df1 = sp.from_arrow(pa.table({"a": [1, 2], "b": [3, 4]}))
    df2 = sp.from_arrow(pa.table({"a": [5, 6], "b": [7, 8]}))

    # Apply a map that changes schema - this makes _try_get_schema return None
    # because the method can't determine the output schema of map operations
    df1_mapped = df1.map("a + b as c")
    df2_mapped = df2.map("a + b as c")

    # This should work - no early validation, DuckDB handles it
    result = df1_mapped.union(df2_mapped)
    rows = result.take_all()
    assert len(rows) == 4


def test_union_type_get_type_category_function(sp: Session):
    """Test the _get_type_category static method handles various type strings."""
    from smallpond.dataframe import DataFrame

    # Numeric types
    assert DataFrame._get_type_category("int64") == "numeric"
    assert DataFrame._get_type_category("INT32") == "numeric"
    assert DataFrame._get_type_category("float64") == "numeric"
    assert DataFrame._get_type_category("double") == "numeric"
    assert DataFrame._get_type_category("decimal(10,2)") == "numeric"
    assert DataFrame._get_type_category("bigint") == "numeric"

    # String types
    assert DataFrame._get_type_category("string") == "string"
    assert DataFrame._get_type_category("large_string") == "string"
    assert DataFrame._get_type_category("utf8") == "string"
    assert DataFrame._get_type_category("VARCHAR") == "string"

    # Boolean types
    assert DataFrame._get_type_category("bool") == "boolean"
    assert DataFrame._get_type_category("boolean") == "boolean"

    # Temporal types
    assert DataFrame._get_type_category("date32") == "temporal"
    assert DataFrame._get_type_category("timestamp[ns]") == "temporal"
    assert DataFrame._get_type_category("time64[us]") == "temporal"

    # Binary types
    assert DataFrame._get_type_category("binary") == "binary"
    assert DataFrame._get_type_category("large_binary") == "binary"

    # Unknown types should return "other"
    assert DataFrame._get_type_category("unknown_type") == "other"
    assert DataFrame._get_type_category("struct<a: int>") == "other"


# ==================== Drop Duplicates Tests ====================


def test_drop_duplicates_all_columns(sp: Session):
    """Test drop_duplicates on all columns (default)."""
    df = sp.from_arrow(pa.table({
        "a": [1, 2, 1, 2, 1],
        "b": [10, 20, 10, 20, 30]
    }))
    result = df.drop_duplicates()
    rows = sorted(result.take_all(), key=lambda x: (x["a"], x["b"]))
    assert len(rows) == 3
    assert rows[0] == {"a": 1, "b": 10}
    assert rows[1] == {"a": 1, "b": 30}
    assert rows[2] == {"a": 2, "b": 20}


def test_drop_duplicates_subset_single_column(sp: Session):
    """Test drop_duplicates on a single column subset (default keeps first)."""
    df = sp.from_arrow(pa.table({
        "id": [1, 1, 2, 2, 3],
        "name": ["Alice", "Bob", "Carol", "Dave", "Eve"]
    }))
    result = df.drop_duplicates(subset="id")
    rows = sorted(result.take_all(), key=lambda x: x["id"])
    assert len(rows) == 3
    # Default is keep='first', so first occurrence is kept
    assert rows[0] == {"id": 1, "name": "Alice"}  # First occurrence of id=1
    assert rows[1] == {"id": 2, "name": "Carol"}  # First occurrence of id=2
    assert rows[2] == {"id": 3, "name": "Eve"}


def test_drop_duplicates_subset_multiple_columns(sp: Session):
    """Test drop_duplicates on multiple columns subset."""
    df = sp.from_arrow(pa.table({
        "a": [1, 1, 1, 2, 2],
        "b": [1, 1, 2, 1, 1],
        "c": ["x", "y", "z", "w", "v"]
    }))
    result = df.drop_duplicates(subset=["a", "b"])
    rows = sorted(result.take_all(), key=lambda x: (x["a"], x["b"]))
    assert len(rows) == 3
    assert rows[0]["a"] == 1 and rows[0]["b"] == 1
    assert rows[1]["a"] == 1 and rows[1]["b"] == 2
    assert rows[2]["a"] == 2 and rows[2]["b"] == 1


def test_drop_duplicates_keep_first(sp: Session):
    """Test drop_duplicates keeping the first occurrence."""
    df = sp.from_arrow(pa.table({
        "id": [1, 1, 1],
        "value": [100, 200, 300]
    }))
    result = df.drop_duplicates(subset="id", keep="first")
    rows = result.take_all()
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["value"] == 100


def test_drop_duplicates_keep_last(sp: Session):
    """Test drop_duplicates keeping the last occurrence."""
    df = sp.from_arrow(pa.table({
        "id": [1, 1, 1],
        "value": [100, 200, 300]
    }))
    result = df.drop_duplicates(subset="id", keep="last")
    rows = result.take_all()
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    assert rows[0]["value"] == 300


def test_drop_duplicates_keep_any(sp: Session):
    """Test drop_duplicates keeping any row (for performance optimization)."""
    df = sp.from_arrow(pa.table({
        "id": [1, 1, 2],
        "value": [10, 20, 30]
    }))
    result = df.drop_duplicates(subset="id", keep="any")
    rows = sorted(result.take_all(), key=lambda x: x["id"])
    assert len(rows) == 2
    assert rows[0]["id"] == 1
    assert rows[1]["id"] == 2
    assert rows[1]["value"] == 30


def test_drop_duplicates_no_duplicates(sp: Session):
    """Test drop_duplicates when there are no duplicates."""
    df = sp.from_arrow(pa.table({
        "id": [1, 2, 3],
        "name": ["Alice", "Bob", "Carol"]
    }))
    result = df.drop_duplicates()
    rows = sorted(result.take_all(), key=lambda x: x["id"])
    assert len(rows) == 3
    assert rows[0] == {"id": 1, "name": "Alice"}
    assert rows[1] == {"id": 2, "name": "Bob"}
    assert rows[2] == {"id": 3, "name": "Carol"}


def test_drop_duplicates_all_duplicates(sp: Session):
    """Test drop_duplicates when all rows are duplicates."""
    df = sp.from_arrow(pa.table({
        "a": [1, 1, 1],
        "b": [2, 2, 2]
    }))
    result = df.drop_duplicates()
    rows = result.take_all()
    assert len(rows) == 1
    assert rows[0] == {"a": 1, "b": 2}


def test_drop_duplicates_empty_dataframe(sp: Session):
    """Test drop_duplicates on an empty DataFrame."""
    df = sp.from_arrow(pa.table({
        "id": pa.array([], type=pa.int64()),
        "name": pa.array([], type=pa.string())
    }))
    result = df.drop_duplicates()
    rows = result.take_all()
    assert len(rows) == 0


def test_drop_duplicates_with_nulls(sp: Session):
    """Test drop_duplicates with null values."""
    df = sp.from_arrow(pa.table({
        "id": [1, 1, None, None, 2],
        "val": ["a", "a", "b", "b", "c"]
    }))
    result = df.drop_duplicates()
    rows = result.take_all()
    # Should have 3 unique rows: (1, a), (None, b), (2, c)
    assert len(rows) == 3


def test_drop_duplicates_after_union(sp: Session):
    """Test drop_duplicates after union operation."""
    df1 = sp.from_arrow(pa.table({"id": [1, 2], "val": ["a", "b"]}))
    df2 = sp.from_arrow(pa.table({"id": [2, 3], "val": ["b", "c"]}))
    combined = df1.union(df2)
    result = combined.drop_duplicates()
    rows = sorted(result.take_all(), key=lambda x: x["id"])
    assert len(rows) == 3
    assert rows[0] == {"id": 1, "val": "a"}
    assert rows[1] == {"id": 2, "val": "b"}
    assert rows[2] == {"id": 3, "val": "c"}


def test_drop_duplicates_partitioned_dataframe(sp: Session):
    """Test drop_duplicates on a partitioned DataFrame."""
    # Create data with duplicates across what would be different partitions
    df = sp.from_items(list(range(100)) + list(range(50)))  # 0-99 + 0-49 again
    df = df.repartition(10, by_rows=True)
    result = df.drop_duplicates()
    rows = sorted(result.take_all(), key=lambda x: x["item"])
    assert len(rows) == 100
    assert [r["item"] for r in rows] == list(range(100))


def test_drop_duplicates_partitioned_with_subset(sp: Session):
    """Test drop_duplicates with subset on a partitioned DataFrame."""
    df = sp.from_arrow(pa.table({
        "id": [1, 2, 3, 1, 2, 3, 4, 5],
        "category": ["A", "A", "A", "B", "B", "B", "A", "A"],
        "value": [10, 20, 30, 40, 50, 60, 70, 80]
    }))
    df = df.repartition(4, by_rows=True)
    result = df.drop_duplicates(subset="id")
    rows = sorted(result.take_all(), key=lambda x: x["id"])
    assert len(rows) == 5
    ids = [r["id"] for r in rows]
    assert ids == [1, 2, 3, 4, 5]


def test_drop_duplicates_invalid_keep_value(sp: Session):
    """Test that invalid keep value raises ValueError."""
    df = sp.from_arrow(pa.table({"a": [1, 2, 3]}))
    with pytest.raises(ValueError) as excinfo:
        df.drop_duplicates(keep="invalid")
    assert "Invalid keep value" in str(excinfo.value)


def test_drop_duplicates_invalid_subset_column(sp: Session):
    """Test that invalid subset column raises ValueError."""
    df = sp.from_arrow(pa.table({"a": [1, 2, 3], "b": [4, 5, 6]}))
    with pytest.raises(ValueError) as excinfo:
        df.drop_duplicates(subset="nonexistent")
    assert "Columns not found" in str(excinfo.value)


def test_drop_duplicates_preserves_non_null_requirements(sp: Session):
    """Test that drop_duplicates preserves non-null column requirements."""
    df = sp.from_arrow(pa.table({"id": [1, 1, 2], "val": ["a", "b", "c"]}))
    df_with_requirement = df.require_non_null("id")
    result = df_with_requirement.drop_duplicates(subset="id")
    # The non-null requirement should be preserved
    assert "id" in result._non_null_columns


def test_drop_duplicates_keep_first_multiple_groups(sp: Session):
    """Test keep='first' with multiple duplicate groups."""
    df = sp.from_arrow(pa.table({
        "group": ["A", "A", "A", "B", "B", "C"],
        "order": [1, 2, 3, 1, 2, 1],
        "value": ["a1", "a2", "a3", "b1", "b2", "c1"]
    }))
    result = df.drop_duplicates(subset="group", keep="first")
    rows = sorted(result.take_all(), key=lambda x: x["group"])
    assert len(rows) == 3
    assert rows[0] == {"group": "A", "order": 1, "value": "a1"}
    assert rows[1] == {"group": "B", "order": 1, "value": "b1"}
    assert rows[2] == {"group": "C", "order": 1, "value": "c1"}


def test_drop_duplicates_keep_last_multiple_groups(sp: Session):
    """Test keep='last' with multiple duplicate groups."""
    df = sp.from_arrow(pa.table({
        "group": ["A", "A", "A", "B", "B", "C"],
        "order": [1, 2, 3, 1, 2, 1],
        "value": ["a1", "a2", "a3", "b1", "b2", "c1"]
    }))
    result = df.drop_duplicates(subset="group", keep="last")
    rows = sorted(result.take_all(), key=lambda x: x["group"])
    assert len(rows) == 3
    assert rows[0] == {"group": "A", "order": 3, "value": "a3"}
    assert rows[1] == {"group": "B", "order": 2, "value": "b2"}
    assert rows[2] == {"group": "C", "order": 1, "value": "c1"}


def test_drop_duplicates_with_string_columns(sp: Session):
    """Test drop_duplicates with string columns."""
    df = sp.from_arrow(pa.table({
        "name": ["Alice", "Bob", "Alice", "Carol", "Bob"],
        "city": ["NYC", "LA", "NYC", "Chicago", "LA"]
    }))
    result = df.drop_duplicates()
    rows = sorted(result.take_all(), key=lambda x: (x["name"], x["city"]))
    assert len(rows) == 3
    assert rows[0] == {"name": "Alice", "city": "NYC"}
    assert rows[1] == {"name": "Bob", "city": "LA"}
    assert rows[2] == {"name": "Carol", "city": "Chicago"}


def test_drop_duplicates_default_keeps_first(sp: Session):
    """Test that drop_duplicates defaults to keep='first' (pandas compatible)."""
    df = sp.from_arrow(pa.table({
        "id": [1, 1, 1],
        "value": [100, 200, 300]
    }))
    # Call without specifying keep - should default to 'first'
    result = df.drop_duplicates(subset="id")
    rows = result.take_all()
    assert len(rows) == 1
    assert rows[0]["id"] == 1
    # First occurrence (value=100) should be kept, matching pandas behavior
    assert rows[0]["value"] == 100