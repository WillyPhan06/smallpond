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
