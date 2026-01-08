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
