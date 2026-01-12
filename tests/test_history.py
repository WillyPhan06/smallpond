"""
Tests for DataFrame operation history tracking feature.

This module tests the OperationRecord class and the history() method that
tracks dataset-level operations performed on DataFrames.
"""

from dataclasses import FrozenInstanceError

import pandas as pd
import pyarrow as pa
import pytest

from smallpond.dataframe import OperationRecord, Session


# =============================================================================
# OperationRecord Tests
# =============================================================================


class TestOperationRecord:
    """Tests for the OperationRecord dataclass."""

    def test_operation_record_creation(self):
        """Test basic OperationRecord creation with operation and params."""
        record = OperationRecord(operation="filter", params=(("predicate", "a > 1"),))
        assert record.operation == "filter"
        assert record.params == (("predicate", "a > 1"),)
        assert record.timestamp is not None

    def test_operation_record_empty_params(self):
        """Test OperationRecord with empty params."""
        record = OperationRecord(operation="random_shuffle", params=())
        assert record.operation == "random_shuffle"
        assert record.params == ()

    def test_operation_record_default_params(self):
        """Test OperationRecord uses empty tuple as default params."""
        record = OperationRecord(operation="test")
        assert record.params == ()

    def test_operation_record_repr(self):
        """Test OperationRecord string representation."""
        record = OperationRecord(operation="filter", params=(("predicate", "x > 0"),))
        repr_str = repr(record)
        assert "filter" in repr_str
        assert "predicate" in repr_str

    def test_operation_record_timestamp_ordering(self):
        """Test that timestamps are created in order."""
        record1 = OperationRecord(operation="op1", params=())
        record2 = OperationRecord(operation="op2", params=())
        assert record1.timestamp <= record2.timestamp

    def test_operation_record_get_params(self):
        """Test get_params() returns a dictionary."""
        record = OperationRecord(
            operation="filter",
            params=(("predicate", "a > 1"), ("extra", 42))
        )
        params_dict = record.get_params()
        assert isinstance(params_dict, dict)
        assert params_dict == {"predicate": "a > 1", "extra": 42}

    def test_operation_record_get_params_empty(self):
        """Test get_params() with empty params returns empty dict."""
        record = OperationRecord(operation="test", params=())
        assert record.get_params() == {}


# =============================================================================
# OperationRecord Immutability Tests
# =============================================================================


class TestOperationRecordImmutability:
    """Tests to ensure OperationRecord is immutable."""

    def test_cannot_modify_operation(self):
        """Test that operation attribute cannot be modified."""
        record = OperationRecord(operation="filter", params=())
        with pytest.raises(FrozenInstanceError):
            record.operation = "map"

    def test_cannot_modify_params(self):
        """Test that params attribute cannot be modified."""
        record = OperationRecord(operation="filter", params=(("predicate", "a > 1"),))
        with pytest.raises(FrozenInstanceError):
            record.params = (("predicate", "a > 2"),)

    def test_cannot_modify_timestamp(self):
        """Test that timestamp attribute cannot be modified."""
        from datetime import datetime
        record = OperationRecord(operation="filter", params=())
        with pytest.raises(FrozenInstanceError):
            record.timestamp = datetime.now()

    def test_cannot_add_new_attribute(self):
        """Test that new attributes cannot be added."""
        record = OperationRecord(operation="filter", params=())
        with pytest.raises(FrozenInstanceError):
            record.new_attr = "value"

    def test_cannot_delete_attribute(self):
        """Test that attributes cannot be deleted."""
        record = OperationRecord(operation="filter", params=())
        with pytest.raises(FrozenInstanceError):
            del record.operation

    def test_params_tuple_is_immutable(self):
        """Test that the params tuple cannot be modified (tuples are inherently immutable)."""
        record = OperationRecord(operation="filter", params=(("predicate", "a > 1"),))
        # Tuples don't have append, extend, etc. - they are immutable
        assert not hasattr(record.params, 'append')
        assert not hasattr(record.params, 'extend')
        assert not hasattr(record.params, '__setitem__')

    def test_record_can_be_hashed(self):
        """Test that frozen dataclass can be hashed (useful for sets/dicts)."""
        record = OperationRecord(operation="filter", params=(("predicate", "a > 1"),))
        # Should not raise - frozen dataclasses are hashable
        hash_value = hash(record)
        assert isinstance(hash_value, int)

    def test_records_with_same_data_are_equal(self):
        """Test that records with identical data are equal."""
        from datetime import datetime
        fixed_time = datetime(2024, 1, 1, 12, 0, 0)
        record1 = OperationRecord(operation="filter", params=(("predicate", "a > 1"),), timestamp=fixed_time)
        record2 = OperationRecord(operation="filter", params=(("predicate", "a > 1"),), timestamp=fixed_time)
        assert record1 == record2


# =============================================================================
# Data Source History Tests
# =============================================================================


class TestDataSourceHistory:
    """Tests for history tracking of data source operations."""

    def test_from_pandas_history(self, sp: Session):
        """Test history is recorded for from_pandas."""
        pandas_df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df = sp.from_pandas(pandas_df)

        history = df.history()
        assert len(history) == 1
        assert history[0].operation == "from_pandas"
        params = history[0].get_params()
        assert params["shape"] == (3, 2)
        assert params["columns"] == ("a", "b")

    def test_from_arrow_history(self, sp: Session):
        """Test history is recorded for from_arrow."""
        arrow_table = pa.table({"x": [1, 2], "y": [3, 4]})
        df = sp.from_arrow(arrow_table)

        history = df.history()
        assert len(history) == 1
        assert history[0].operation == "from_arrow"
        params = history[0].get_params()
        assert params["num_rows"] == 2
        assert params["columns"] == ("x", "y")

    def test_from_items_history(self, sp: Session):
        """Test history is recorded for from_items."""
        df = sp.from_items([{"a": 1}, {"a": 2}, {"a": 3}])

        history = df.history()
        assert len(history) == 1
        assert history[0].operation == "from_items"
        params = history[0].get_params()
        assert params["num_items"] == 3

    def test_read_parquet_history(self, sp: Session):
        """Test history is recorded for read_parquet."""
        df = sp.read_parquet("tests/data/mock_urls/*.parquet")

        history = df.history()
        assert len(history) == 1
        assert history[0].operation == "read_parquet"
        params = history[0].get_params()
        assert "tests/data/mock_urls/*.parquet" in params["paths"]

    def test_read_csv_history(self, sp: Session):
        """Test history is recorded for read_csv."""
        df = sp.read_csv(
            "tests/data/mock_urls/*.tsv",
            schema={"urlstr": "varchar", "valstr": "varchar"},
            delim=r"\t",
        )

        history = df.history()
        assert len(history) == 1
        assert history[0].operation == "read_csv"
        params = history[0].get_params()
        assert params["delim"] == r"\t"


# =============================================================================
# Single Operation History Tests
# =============================================================================


class TestSingleOperationHistory:
    """Tests for history tracking of individual transformation operations."""

    def test_filter_sql_history(self, sp: Session):
        """Test filter with SQL predicate records history correctly."""
        df = sp.from_items([{"a": 1}, {"a": 2}, {"a": 3}])
        filtered = df.filter("a > 1")

        history = filtered.history()
        assert len(history) == 2
        assert history[0].operation == "from_items"
        assert history[1].operation == "filter"
        assert history[1].get_params()["predicate"] == "a > 1"

    def test_filter_lambda_history(self, sp: Session):
        """Test filter with lambda function records history correctly."""
        df = sp.from_items([{"a": 1}, {"a": 2}, {"a": 3}])
        filtered = df.filter(lambda r: r["a"] > 1)

        history = filtered.history()
        assert len(history) == 2
        assert history[1].operation == "filter"
        # Lambda should be recorded as "<function>"
        assert history[1].get_params()["predicate"] == "<function>"

    def test_map_sql_history(self, sp: Session):
        """Test map with SQL expression records history correctly."""
        df = sp.from_arrow(pa.table({"a": [1, 2], "b": [3, 4]}))
        mapped = df.map("a + b as c")

        history = mapped.history()
        assert len(history) == 2
        assert history[1].operation == "map"
        assert history[1].get_params()["expression"] == "a + b as c"

    def test_map_lambda_history(self, sp: Session):
        """Test map with lambda function records history correctly."""
        df = sp.from_arrow(pa.table({"a": [1, 2], "b": [3, 4]}))
        mapped = df.map(lambda r: {"c": r["a"] + r["b"]})

        history = mapped.history()
        assert len(history) == 2
        assert history[1].operation == "map"
        # Lambda should be recorded as "<function>"
        assert history[1].get_params()["expression"] == "<function>"

    def test_flat_map_sql_history(self, sp: Session):
        """Test flat_map with SQL expression records history."""
        df = sp.from_arrow(pa.table({"a": [1, 2], "b": [3, 4]}))
        flat_mapped = df.flat_map("unnest(array[a, b]) as c")

        history = flat_mapped.history()
        assert len(history) == 2
        assert history[1].operation == "flat_map"
        assert history[1].get_params()["expression"] == "unnest(array[a, b]) as c"

    def test_flat_map_lambda_history(self, sp: Session):
        """Test flat_map with lambda function records history."""
        df = sp.from_arrow(pa.table({"a": [1, 2]}))
        flat_mapped = df.flat_map(lambda r: [{"c": r["a"]}, {"c": r["a"] + 1}])

        history = flat_mapped.history()
        assert len(history) == 2
        assert history[1].operation == "flat_map"
        assert history[1].get_params()["expression"] == "<function>"

    def test_limit_history(self, sp: Session):
        """Test limit records history correctly."""
        df = sp.from_items([{"a": 1}, {"a": 2}, {"a": 3}])
        limited = df.limit(2)

        history = limited.history()
        assert len(history) == 2
        assert history[1].operation == "limit"
        assert history[1].get_params()["limit"] == 2

    def test_repartition_history(self, sp: Session):
        """Test repartition records history correctly."""
        df = sp.from_items([{"a": i} for i in range(100)])
        repartitioned = df.repartition(4, hash_by="a")

        history = repartitioned.history()
        assert len(history) == 2
        assert history[1].operation == "repartition"
        params = history[1].get_params()
        assert params["npartitions"] == 4
        assert params["hash_by"] == "a"

    def test_random_shuffle_history(self, sp: Session):
        """Test random_shuffle records history correctly."""
        df = sp.from_items([{"a": i} for i in range(10)]).repartition(2, by_rows=True)
        shuffled = df.random_shuffle()

        history = shuffled.history()
        assert history[-1].operation == "random_shuffle"
        assert history[-1].get_params() == {}

    def test_partial_sort_history(self, sp: Session):
        """Test partial_sort records history correctly."""
        df = sp.from_items([{"a": 3}, {"a": 1}, {"a": 2}])
        sorted_df = df.partial_sort(by="a")

        history = sorted_df.history()
        assert len(history) == 2
        assert history[1].operation == "partial_sort"
        assert history[1].get_params()["by"] == ["a"]

    def test_map_batches_history(self, sp: Session):
        """Test map_batches records history correctly."""
        df = sp.from_arrow(pa.table({"a": [1, 2, 3]}))
        batched = df.map_batches(lambda batch: batch, batch_size=100)

        history = batched.history()
        assert len(history) == 2
        assert history[1].operation == "map_batches"
        params = history[1].get_params()
        assert params["func"] == "<function>"
        assert params["batch_size"] == 100


# =============================================================================
# Column Operation History Tests
# =============================================================================


class TestColumnOperationHistory:
    """Tests for history tracking of column operations."""

    def test_rename_columns_history(self, sp: Session):
        """Test rename_columns records history correctly."""
        df = sp.from_arrow(pa.table({"old_name": [1, 2, 3]}))
        renamed = df.rename_columns({"old_name": "new_name"})

        history = renamed.history()
        assert len(history) == 2
        assert history[1].operation == "rename_columns"
        assert history[1].get_params()["mapping"] == {"old_name": "new_name"}

    def test_drop_columns_history(self, sp: Session):
        """Test drop_columns records history correctly."""
        df = sp.from_arrow(pa.table({"a": [1, 2], "b": [3, 4], "c": [5, 6]}))
        dropped = df.drop_columns(["b", "c"])

        history = dropped.history()
        assert len(history) == 2
        assert history[1].operation == "drop_columns"
        assert history[1].get_params()["columns"] == ["b", "c"]

    def test_select_columns_history(self, sp: Session):
        """Test select_columns records history correctly."""
        df = sp.from_arrow(pa.table({"a": [1, 2], "b": [3, 4], "c": [5, 6]}))
        selected = df.select_columns(["c", "a"])

        history = selected.history()
        assert len(history) == 2
        assert history[1].operation == "select_columns"
        assert history[1].get_params()["columns"] == ["c", "a"]


# =============================================================================
# Aggregation History Tests
# =============================================================================


class TestAggregationHistory:
    """Tests for history tracking of aggregation operations."""

    def test_groupby_agg_history(self, sp: Session):
        """Test groupby_agg records history correctly."""
        df = sp.from_items([
            {"category": "A", "value": 10},
            {"category": "A", "value": 20},
            {"category": "B", "value": 30},
        ])
        aggregated = df.groupby_agg(by="category", aggs={"value": ["sum", "count"]})

        history = aggregated.history()
        # from_items + groupby_agg (internally may have repartitions, but the recorded op is groupby_agg)
        assert history[-1].operation == "groupby_agg"
        params = history[-1].get_params()
        assert params["by"] == "category"
        assert params["aggs"] == {"value": ["sum", "count"]}

    def test_drop_duplicates_history(self, sp: Session):
        """Test drop_duplicates records history correctly."""
        df = sp.from_items([{"a": 1, "b": 2}, {"a": 1, "b": 3}, {"a": 2, "b": 4}])
        deduped = df.drop_duplicates(subset="a", keep="first")

        history = deduped.history()
        assert history[-1].operation == "drop_duplicates"
        params = history[-1].get_params()
        assert params["subset"] == "a"
        assert params["keep"] == "first"


# =============================================================================
# Metadata Operation History Tests
# =============================================================================


class TestMetadataOperationHistory:
    """Tests for history tracking of metadata operations."""

    def test_require_non_null_history(self, sp: Session):
        """Test require_non_null records history correctly."""
        df = sp.from_arrow(pa.table({"a": [1, 2, 3]}))
        df = df.require_non_null("a")

        history = df.history()
        assert len(history) == 2
        assert history[1].operation == "require_non_null"
        assert history[1].get_params()["columns"] == ("a",)

    def test_recompute_history(self, sp: Session):
        """Test recompute records history correctly."""
        df = sp.from_arrow(pa.table({"a": [1, 2, 3]}))
        df = df.recompute()

        history = df.history()
        assert len(history) == 2
        assert history[1].operation == "recompute"
        assert history[1].get_params() == {}

    def test_no_cache_history(self, sp: Session):
        """Test no_cache records history correctly."""
        df = sp.from_arrow(pa.table({"a": [1, 2, 3]}))
        df = df.no_cache()

        history = df.history()
        assert len(history) == 2
        assert history[1].operation == "no_cache"
        assert history[1].get_params() == {}


# =============================================================================
# Multi-DataFrame Operation History Tests
# =============================================================================


class TestMultiDataFrameHistory:
    """Tests for history tracking of operations involving multiple DataFrames."""

    def test_join_preserves_left_history_only(self, sp: Session):
        """Test that join only preserves the left (calling) DataFrame's history."""
        left_df = sp.from_items([{"id": 1, "a": 10}, {"id": 2, "a": 20}])
        left_df = left_df.filter("a > 5")  # Add another operation to left

        right_df = sp.from_items([{"id": 1, "b": 100}, {"id": 2, "b": 200}])
        right_df = right_df.filter("b > 50")  # Add another operation to right

        joined = left_df.join(right_df, on="id")

        history = joined.history()
        # Should have: from_items (left) -> filter (left) -> join
        # Should NOT have: from_items (right) or filter (right)
        assert len(history) == 3
        assert history[0].operation == "from_items"
        assert history[1].operation == "filter"
        assert history[1].get_params()["predicate"] == "a > 5"  # Left's filter
        assert history[2].operation == "join"
        assert history[2].get_params()["on"] == "id"

    def test_join_records_parameters(self, sp: Session):
        """Test that join records all its parameters correctly."""
        left = sp.from_items([{"user_id": 1}, {"user_id": 2}])
        right = sp.from_items([{"id": 1}, {"id": 2}])

        joined = left.join(right, left_on="user_id", right_on="id", how="left", npartitions=2)

        history = joined.history()
        join_record = history[-1]
        assert join_record.operation == "join"
        params = join_record.get_params()
        assert params["left_on"] == "user_id"
        assert params["right_on"] == "id"
        assert params["how"] == "left"
        assert params["npartitions"] == 2

    def test_union_preserves_calling_df_history_only(self, sp: Session):
        """Test that union only preserves the calling DataFrame's history."""
        df1 = sp.from_items([{"a": 1}])
        df1 = df1.filter("a > 0")  # Add another operation

        df2 = sp.from_items([{"a": 2}])
        df2 = df2.filter("a > 0")  # This history should NOT be in result

        df3 = sp.from_items([{"a": 3}])
        df3 = df3.filter("a > 0")  # This history should NOT be in result

        unioned = df1.union(df2, df3)

        history = unioned.history()
        # Should have: from_items (df1) -> filter (df1) -> union
        # Should NOT have histories from df2 or df3
        assert len(history) == 3
        assert history[0].operation == "from_items"
        assert history[1].operation == "filter"
        assert history[2].operation == "union"
        assert history[2].get_params()["num_dataframes"] == 2

    def test_union_records_num_dataframes(self, sp: Session):
        """Test that union records the number of DataFrames being unioned."""
        df1 = sp.from_items([{"a": 1}])
        df2 = sp.from_items([{"a": 2}])
        df3 = sp.from_items([{"a": 3}])
        df4 = sp.from_items([{"a": 4}])

        # Union 3 DataFrames with df1
        unioned = df1.union(df2, df3, df4)

        history = unioned.history()
        union_record = history[-1]
        assert union_record.operation == "union"
        assert union_record.get_params()["num_dataframes"] == 3  # df2, df3, df4

    def test_partial_sql_preserves_first_input_history(self, sp: Session):
        """Test that partial_sql only preserves the first input DataFrame's history."""
        df1 = sp.from_items([{"id": 1, "a": 10}])
        df1 = df1.filter("a > 5")

        df2 = sp.from_items([{"id": 1, "b": 20}])
        df2 = df2.filter("b > 10")  # This should NOT be in history

        # Need to repartition for partial_sql join
        df1 = df1.repartition(1)
        df2 = df2.repartition(1)

        result = sp.partial_sql("SELECT * FROM {0} JOIN {1} ON {0}.id = {1}.id", df1, df2)

        history = result.history()
        # Should have: from_items -> filter -> repartition -> partial_sql
        # Should NOT have df2's history
        assert history[-1].operation == "partial_sql"
        # Check that df1's operations are present
        ops = [h.operation for h in history]
        assert "from_items" in ops
        assert "filter" in ops
        assert "repartition" in ops


# =============================================================================
# Chained Operation History Tests
# =============================================================================


class TestChainedOperationHistory:
    """Tests for history tracking across multiple chained operations."""

    def test_simple_chain(self, sp: Session):
        """Test history is correct for a simple chain of operations."""
        df = (sp.from_items([{"a": 1}, {"a": 2}, {"a": 3}])
              .filter("a > 1")
              .map("a * 2 as b")
              .limit(1))

        history = df.history()
        assert len(history) == 4
        assert history[0].operation == "from_items"
        assert history[1].operation == "filter"
        assert history[2].operation == "map"
        assert history[3].operation == "limit"

    def test_complex_chain(self, sp: Session):
        """Test history is correct for a complex chain of operations."""
        df = (sp.from_arrow(pa.table({"id": [1, 2, 3], "value": [10, 20, 30]}))
              .filter("value > 5")
              .repartition(2, hash_by="id")
              .map("id, value * 2 as doubled")
              .partial_sort(by="id")
              .rename_columns({"doubled": "new_value"})
              .select_columns(["id", "new_value"]))

        history = df.history()
        assert len(history) == 7
        assert [h.operation for h in history] == [
            "from_arrow",
            "filter",
            "repartition",
            "map",
            "partial_sort",
            "rename_columns",
            "select_columns",
        ]

    def test_chain_with_multiple_filters(self, sp: Session):
        """Test history tracks multiple filter operations."""
        df = (sp.from_items([{"a": 1, "b": 10}, {"a": 2, "b": 20}, {"a": 3, "b": 30}])
              .filter("a > 1")
              .filter("b < 30")
              .filter("a + b > 20"))

        history = df.history()
        assert len(history) == 4
        assert history[1].get_params()["predicate"] == "a > 1"
        assert history[2].get_params()["predicate"] == "b < 30"
        assert history[3].get_params()["predicate"] == "a + b > 20"

    def test_chain_with_metadata_operations(self, sp: Session):
        """Test history tracks metadata operations in chain."""
        df = (sp.from_items([{"a": 1}, {"a": 2}])
              .require_non_null("a")
              .no_cache()
              .recompute()
              .filter("a > 0"))

        history = df.history()
        assert len(history) == 5
        assert [h.operation for h in history] == [
            "from_items",
            "require_non_null",
            "no_cache",
            "recompute",
            "filter",
        ]


# =============================================================================
# Edge Cases
# =============================================================================


class TestHistoryEdgeCases:
    """Tests for edge cases in history tracking."""

    def test_history_returns_copy(self, sp: Session):
        """Test that history() returns a copy, not the internal list."""
        df = sp.from_items([{"a": 1}])
        history1 = df.history()
        history2 = df.history()

        # Should be equal but not the same object
        assert history1 == history2
        assert history1 is not history2

        # Modifying the returned list should not affect the DataFrame
        history1.append(OperationRecord(operation="fake", params=()))
        assert len(df.history()) == 1

    def test_operation_record_in_history_is_immutable(self, sp: Session):
        """Test that OperationRecord objects in history cannot be modified."""
        df = sp.from_items([{"a": 1}]).filter("a > 0")
        history = df.history()

        # Try to modify the operation record
        with pytest.raises(FrozenInstanceError):
            history[0].operation = "modified"

        with pytest.raises(FrozenInstanceError):
            history[1].params = (("fake", "value"),)

    def test_empty_history_for_new_df_without_source(self, sp: Session):
        """Test that a DataFrame created without tracking has empty history."""
        # This tests internal behavior - DataFrames should always have history from source
        df = sp.from_items([{"a": 1}])
        assert len(df.history()) >= 1  # At least the from_items operation

    def test_filter_with_complex_sql(self, sp: Session):
        """Test filter with complex SQL expression."""
        df = sp.from_items([{"a": 1, "b": 2}])
        filtered = df.filter("a > 0 AND b < 10 OR (a = 1 AND b = 2)")

        history = filtered.history()
        assert history[-1].get_params()["predicate"] == "a > 0 AND b < 10 OR (a = 1 AND b = 2)"

    def test_map_with_complex_sql(self, sp: Session):
        """Test map with complex SQL expression."""
        df = sp.from_items([{"a": 1, "b": 2}])
        mapped = df.map("a + b as sum, a * b as product, CASE WHEN a > 0 THEN 'pos' ELSE 'neg' END as sign")

        history = mapped.history()
        assert "CASE WHEN" in history[-1].get_params()["expression"]

    def test_nested_lambda_recorded_as_function(self, sp: Session):
        """Test that nested lambda functions are recorded as <function>."""
        df = sp.from_items([{"a": 1}])

        # Create a closure
        multiplier = 2
        filtered = df.filter(lambda r: r["a"] * multiplier > 0)

        history = filtered.history()
        assert history[-1].get_params()["predicate"] == "<function>"

    def test_branching_dataframes_have_independent_histories(self, sp: Session):
        """Test that branching DataFrames have independent histories."""
        base = sp.from_items([{"a": 1}, {"a": 2}, {"a": 3}])

        branch1 = base.filter("a > 1")
        branch2 = base.filter("a < 3")

        history1 = branch1.history()
        history2 = branch2.history()

        # Both should have from_items + filter
        assert len(history1) == 2
        assert len(history2) == 2

        # But different filter predicates
        assert history1[-1].get_params()["predicate"] == "a > 1"
        assert history2[-1].get_params()["predicate"] == "a < 3"

    def test_history_after_action_operations(self, sp: Session):
        """Test that history is unchanged after action operations like count(), take()."""
        df = sp.from_items([{"a": 1}, {"a": 2}]).filter("a > 0")

        history_before = df.history()
        _ = df.count()  # Action operation
        history_after = df.history()

        assert history_before == history_after

    def test_join_different_how_types(self, sp: Session):
        """Test join records different join types correctly."""
        left = sp.from_items([{"id": 1}])
        right = sp.from_items([{"id": 1}])

        for how in ["inner", "left", "right", "outer", "cross", "semi", "anti"]:
            if how == "cross":
                joined = left.join(right, how=how)
            else:
                joined = left.join(right, on="id", how=how)

            history = joined.history()
            assert history[-1].get_params()["how"] == how


# =============================================================================
# History Consistency Tests
# =============================================================================


class TestHistoryConsistency:
    """Tests to ensure history tracking doesn't affect DataFrame functionality."""

    def test_filter_result_unchanged_with_history(self, sp: Session):
        """Test that filter results are correct with history tracking."""
        df = sp.from_items([{"a": 1}, {"a": 2}, {"a": 3}])
        filtered = df.filter("a > 1")

        result = filtered.take_all()
        assert result == [{"a": 2}, {"a": 3}]

    def test_map_result_unchanged_with_history(self, sp: Session):
        """Test that map results are correct with history tracking."""
        df = sp.from_arrow(pa.table({"a": [1, 2, 3]}))
        mapped = df.map("a * 2 as doubled")

        result = mapped.take_all()
        assert result == [{"doubled": 2}, {"doubled": 4}, {"doubled": 6}]

    def test_join_result_unchanged_with_history(self, sp: Session):
        """Test that join results are correct with history tracking."""
        left = sp.from_items([{"id": 1, "a": 10}, {"id": 2, "a": 20}])
        right = sp.from_items([{"id": 1, "b": 100}, {"id": 2, "b": 200}])

        joined = left.join(right, on="id")

        result = sorted(joined.take_all(), key=lambda x: x["id"])
        assert result == [
            {"id": 1, "a": 10, "b": 100},
            {"id": 2, "a": 20, "b": 200},
        ]

    def test_union_result_unchanged_with_history(self, sp: Session):
        """Test that union results are correct with history tracking."""
        df1 = sp.from_items([{"a": 1}])
        df2 = sp.from_items([{"a": 2}])

        unioned = df1.union(df2)

        result = sorted(unioned.take_all(), key=lambda x: x["a"])
        assert result == [{"a": 1}, {"a": 2}]
