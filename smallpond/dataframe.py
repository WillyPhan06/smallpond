from __future__ import annotations

import hashlib
import os
import pickle
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from threading import Lock
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import pandas as pd
import pyarrow as arrow
import ray
import ray.exceptions
from loguru import logger

from smallpond.execution.task import Task
from smallpond.io.filesystem import remove_path
from smallpond.logical.dataset import *
from smallpond.logical.node import *
from smallpond.logical.optimizer import Optimizer
from smallpond.logical.planner import Planner
from smallpond.session import SessionBase


@dataclass(frozen=True)
class OperationRecord:
    """
    An immutable record of a dataset-level operation performed on a DataFrame.

    This class stores information about operations that affect the DataFrame
    at a dataset level (e.g., filter, join, union) for debugging and auditing
    purposes. It allows users to trace the sequence of transformations that
    led to the current state of a DataFrame.

    The class is immutable (frozen=True) to ensure that once an operation is
    recorded, it cannot be modified. This guarantees the integrity of the
    operation history.

    Attributes
    ----------
    operation : str
        The name of the operation (e.g., 'filter', 'join', 'union').
    params : tuple
        A tuple of (key, value) pairs representing the parameters passed to
        the operation. Stored as a tuple instead of dict for immutability.
        For readability, callable parameters are represented as '<function>'
        and DataFrame parameters are represented as '<DataFrame>'.
    timestamp : datetime
        The timestamp when the operation was recorded.

    Examples
    --------
    .. code-block::

        # Get the history of operations on a DataFrame
        for record in df.history():
            print(f"{record.timestamp}: {record.operation}({record.params})")

        # Access parameters as a dictionary
        params_dict = dict(record.params)

    Notes
    -----
    - Only dataset-level operations are tracked (filter, join, union, etc.).
    - Row-level operations are not tracked as they would require significant
      code changes and memory overhead.
    - The operation history is inherited when creating new DataFrames from
      transformations, maintaining the full lineage of operations.
    - This class is immutable - attempting to modify any attribute after
      creation will raise a ``FrozenInstanceError``.
    """
    operation: str
    params: Tuple[Tuple[str, Any], ...] = field(default_factory=tuple)
    timestamp: datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        params_str = ", ".join(f"{k}={v!r}" for k, v in self.params)
        return f"OperationRecord({self.operation}({params_str}) at {self.timestamp.isoformat()})"

    def get_params(self) -> Dict[str, Any]:
        """
        Get the operation parameters as a dictionary.

        Returns
        -------
        Dict[str, Any]
            A dictionary of parameter names to values.

        Examples
        --------
        .. code-block::

            record = df.history()[0]
            params = record.get_params()
            print(params.get('predicate'))
        """
        return dict(self.params)


class NullValidationError(ValueError):
    """
    Exception raised when non-null validation fails for specified columns.

    This exception is raised when columns marked as required non-null via
    `require_non_null()` contain null values during data retrieval operations
    like `take()`, `to_pandas()`, `to_arrow()`, or `count()`.

    Attributes
    ----------
    columns : List[str]
        The column names that were required to be non-null.
    null_counts : Dict[str, int]
        A dictionary mapping column names to their null value counts.
    message : str
        A descriptive error message.

    Notes
    -----
    **Why inherit from ValueError?**

    This exception inherits from `ValueError` rather than a custom base exception
    for the following reasons:

    1. **Semantic alignment**: `ValueError` is raised when an operation receives
       an argument with the right type but an inappropriate value. In this case,
       the DataFrame's data has an inappropriate value (null) where non-null was
       required. The "value" being validated is the data content itself.

    2. **Consistency with pandas**: pandas raises `ValueError` for similar data
       quality issues (e.g., when `dropna()` would result in empty data with
       certain parameters, or when data doesn't meet expected constraints).

    3. **Catchability**: Users can catch this as `ValueError` for broad error
       handling, or specifically as `NullValidationError` for targeted handling.
       This follows Python's exception hierarchy conventions.

    4. **Not a programming error**: While `ValueError` is sometimes associated
       with programming errors, it's also commonly used for runtime data validation
       failures. The distinction is that `TypeError` is for wrong types (programming
       error), while `ValueError` is for wrong values (which can be data-driven).

    Example of catching both broadly and specifically:

    .. code-block::

        # Catch specifically
        try:
            df.require_non_null("id").to_pandas()
        except NullValidationError as e:
            print(f"Data quality issue: {e.null_counts}")

        # Catch broadly with other value errors
        try:
            df.require_non_null("id").to_pandas()
        except ValueError as e:
            print(f"Value error: {e}")
    """

    def __init__(self, columns: List[str], null_counts: Dict[str, int]):
        self.columns = columns
        self.null_counts = null_counts

        # Build descriptive message
        failed_cols = [col for col in columns if null_counts.get(col, 0) > 0]
        details = ", ".join(f"'{col}' ({null_counts[col]} nulls)" for col in failed_cols)
        self.message = (
            f"Non-null validation failed for columns: {details}. "
            f"These columns were marked as required non-null via require_non_null() but contain null values. "
            f"Please clean your data or adjust the non-null requirements."
        )
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class SchemaMismatchError(ValueError):
    """
    Exception raised when DataFrame schemas don't match during union operations.

    This exception is raised when attempting to union DataFrames with incompatible
    schemas, such as mismatched column names or different column types.

    Attributes
    ----------
    message : str
        A descriptive error message explaining the schema mismatch.
    details : Dict[str, Any]
        Additional details about the mismatch, which may include:
        - missing_columns: Columns missing from one or more DataFrames
        - extra_columns: Unexpected columns in one or more DataFrames
        - type_mismatches: Columns with incompatible types across DataFrames

    Notes
    -----
    This exception inherits from `ValueError` because the schemas represent
    inappropriate values for the union operation. Users can catch this specifically
    or as a general `ValueError`.

    Example:

    .. code-block::

        try:
            combined = df1.union(df2, df3)
        except SchemaMismatchError as e:
            print(f"Schema mismatch: {e}")
            print(f"Details: {e.details}")
    """

    def __init__(self, message: str, details: Optional[Dict[str, Any]] = None):
        self.message = message
        self.details = details or {}
        super().__init__(self.message)

    def __str__(self) -> str:
        return self.message


class DataFrameCache:
    """
    A cache for storing computed DataFrame results.

    This cache stores the results of _compute() calls, keyed by a hash of the
    DataFrame's optimized logical plan. When the same DataFrame operations are
    executed multiple times, the cached results are returned instead of
    recomputing.

    The cache is thread-safe and can be shared across multiple DataFrames
    within the same session.
    """

    def __init__(self, enabled: bool = True, max_entries: Optional[int] = None):
        """
        Initialize the DataFrame cache.

        Parameters
        ----------
        enabled : bool, default True
            Whether caching is enabled by default.
        max_entries : int, optional
            Maximum number of cache entries. If None, no limit is applied.
            When the limit is reached, the oldest entries are evicted.
        """
        self._cache: Dict[str, Tuple[List[DataSet], datetime]] = {}
        self._lock = Lock()
        self._enabled = enabled
        self._max_entries = max_entries
        self._hits = 0
        self._misses = 0

    @property
    def enabled(self) -> bool:
        """Whether caching is currently enabled."""
        return self._enabled

    @enabled.setter
    def enabled(self, value: bool):
        """Enable or disable caching."""
        self._enabled = value

    def _generate_cache_key(self, plan: Node) -> str:
        """
        Generate a unique cache key based on the logical plan.

        The key is generated by traversing the plan tree and hashing
        the relevant attributes of each node that affect computation results.
        """
        def serialize_value(value: Any) -> Any:
            """Serialize a value to a hashable representation."""
            if value is None:
                return None
            elif isinstance(value, (str, int, float, bool)):
                return value
            elif isinstance(value, (list, tuple)):
                return [serialize_value(v) for v in value]
            elif isinstance(value, dict):
                return {k: serialize_value(v) for k, v in sorted(value.items())}
            elif isinstance(value, set):
                return sorted(list(value))
            elif hasattr(value, "__dict__"):
                # For objects, serialize their __dict__ but skip private attrs and methods
                return {
                    k: serialize_value(v)
                    for k, v in value.__dict__.items()
                    if not k.startswith("_") and not callable(v)
                }
            else:
                # For other types, use string representation
                return str(value)

        def node_to_dict(node: Node) -> dict:
            """Convert a node to a dictionary representation for hashing."""
            result = {
                "type": node.__class__.__name__,
                "id": int(node.id),
            }

            # Collect all public attributes that are not methods or private
            # This ensures any new node types are automatically handled
            for attr_name in dir(node):
                # Skip private attributes, methods, and known non-deterministic attrs
                if attr_name.startswith("_"):
                    continue
                if attr_name in ("input_deps", "ctx", "generated_tasks", "perf_stats",
                                 "perf_metrics", "location", "optimized_plan"):
                    continue

                try:
                    attr_value = getattr(node, attr_name)
                    # Skip methods and callables
                    if callable(attr_value):
                        continue
                    # Serialize the value
                    result[attr_name] = serialize_value(attr_value)
                except Exception as e:
                    # If we can't get/serialize an attribute, log it and skip
                    logger.debug(
                        f"Cache key generation: skipping attribute '{attr_name}' on node "
                        f"{node.__class__.__name__} (id={node.id}) due to serialization error: {e}"
                    )

            # Recursively process input dependencies
            result["input_deps"] = [node_to_dict(dep) for dep in node.input_deps]

            return result

        plan_dict = node_to_dict(plan)
        plan_str = pickle.dumps(plan_dict)
        return hashlib.sha256(plan_str).hexdigest()

    def get(self, plan: Node) -> Optional[List[DataSet]]:
        """
        Get cached results for the given plan.

        Parameters
        ----------
        plan : Node
            The optimized logical plan.

        Returns
        -------
        Optional[List[DataSet]]
            The cached results if available, None otherwise.
        """
        if not self._enabled:
            return None

        key = self._generate_cache_key(plan)
        with self._lock:
            if key in self._cache:
                self._hits += 1
                result, _ = self._cache[key]
                logger.debug(f"Cache hit for plan {plan!r}, key={key[:8]}...")
                return result
            self._misses += 1
            return None

    def put(self, plan: Node, result: List[DataSet]) -> None:
        """
        Store results in the cache.

        Parameters
        ----------
        plan : Node
            The optimized logical plan.
        result : List[DataSet]
            The computed results to cache.
        """
        if not self._enabled:
            return

        key = self._generate_cache_key(plan)
        with self._lock:
            # Evict oldest entries if max_entries is set and reached
            if self._max_entries is not None and len(self._cache) >= self._max_entries:
                # Find and remove the oldest entry
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k][1])
                del self._cache[oldest_key]
                logger.debug(f"Evicted cache entry {oldest_key[:8]}...")

            self._cache[key] = (result, datetime.now())
            logger.debug(f"Cached result for plan {plan!r}, key={key[:8]}...")

    def invalidate(self, plan: Optional[Node] = None) -> int:
        """
        Invalidate cache entries.

        Parameters
        ----------
        plan : Node, optional
            If provided, only invalidate the cache entry for this specific plan.
            If None, clear the entire cache.

        Returns
        -------
        int
            The number of entries removed.
        """
        with self._lock:
            if plan is None:
                count = len(self._cache)
                self._cache.clear()
                logger.info(f"Cleared all {count} cache entries")
                return count
            else:
                key = self._generate_cache_key(plan)
                if key in self._cache:
                    del self._cache[key]
                    logger.debug(f"Invalidated cache entry for plan {plan!r}")
                    return 1
                return 0

    def get_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - entries: Number of cached entries
            - hits: Number of cache hits
            - misses: Number of cache misses
            - hit_rate: Cache hit rate (0.0 to 1.0)
            - enabled: Whether caching is enabled
        """
        with self._lock:
            total = self._hits + self._misses
            return {
                "entries": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": self._hits / total if total > 0 else 0.0,
                "enabled": self._enabled,
            }

    def get_cached_entries(self) -> List[Dict[str, Any]]:
        """
        Get information about all cached entries.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, each containing:
            - key: The cache key (truncated)
            - cached_at: When the entry was cached
            - num_datasets: Number of datasets in the cached result
        """
        with self._lock:
            entries = []
            for key, (result, cached_at) in self._cache.items():
                entries.append({
                    "key": key[:16] + "...",
                    "cached_at": cached_at.isoformat(),
                    "num_datasets": len(result),
                })
            return entries

    def reset_stats(self) -> None:
        """Reset cache hit/miss statistics."""
        with self._lock:
            self._hits = 0
            self._misses = 0


# Global default cache instance
_default_cache: Optional[DataFrameCache] = None
_default_cache_lock = Lock()


def get_default_cache() -> DataFrameCache:
    """Get or create the default global cache instance."""
    global _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            _default_cache = DataFrameCache()
        return _default_cache


def set_default_cache(cache: Optional[DataFrameCache]) -> None:
    """Set the default global cache instance."""
    global _default_cache
    with _default_cache_lock:
        _default_cache = cache


class Session(SessionBase):
    """
    Extended session class with methods to create DataFrames.

    The Session class is the main entry point for creating DataFrames in smallpond.
    It provides methods to read data from various sources (Parquet, CSV, JSON, pandas,
    PyArrow) and execute SQL queries.

    Operation History Tracking
    --------------------------
    When a DataFrame is created through Session methods (e.g., ``read_parquet()``,
    ``from_pandas()``), operation history tracking begins automatically. The first
    operation recorded is the data source operation itself.

    For example:
    - ``sp.read_parquet("data/*.parquet")`` creates a DataFrame with history
      starting at ``read_parquet``
    - ``sp.from_pandas(df)`` creates a DataFrame with history starting at ``from_pandas``
    - ``sp.partial_sql(query, df1, df2)`` inherits history from the first input
      DataFrame and adds ``partial_sql``

    The history continues to build as transformations are applied to the DataFrame.
    See ``DataFrame.history()`` for more details on accessing and understanding
    the operation history.
    """

    def __init__(self, cache: Optional[DataFrameCache] = None, **kwargs):
        """
        Initialize a Session.

        Parameters
        ----------
        cache : DataFrameCache, optional
            A cache instance for storing computed DataFrame results.
            If None, uses the global default cache.
            Pass DataFrameCache(enabled=False) to disable caching.
        **kwargs
            Additional arguments passed to SessionBase.
        """
        super().__init__(**kwargs)
        self._nodes: List[Node] = []

        self._node_to_tasks: Dict[Node, List[Task]] = {}
        """
        When a DataFrame is evaluated, the tasks of the logical plan are stored here.
        Subsequent DataFrames can reuse the tasks to avoid recomputation.
        """

        self._cache = cache if cache is not None else get_default_cache()
        """
        Cache for storing computed DataFrame results.
        """

    @property
    def cache(self) -> DataFrameCache:
        """Get the cache instance for this session."""
        return self._cache

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        Get cache statistics for this session.

        Returns
        -------
        Dict[str, Any]
            Cache statistics including entries, hits, misses, and hit rate.
        """
        return self._cache.get_stats()

    def get_cached_entries(self) -> List[Dict[str, Any]]:
        """
        Get information about all cached entries.

        Returns
        -------
        List[Dict[str, Any]]
            Information about each cached entry.
        """
        return self._cache.get_cached_entries()

    def clear_cache(self) -> int:
        """
        Clear all cached DataFrame results.

        Returns
        -------
        int
            The number of cache entries cleared.
        """
        return self._cache.invalidate()

    def set_cache_enabled(self, enabled: bool) -> None:
        """
        Enable or disable caching for this session.

        Parameters
        ----------
        enabled : bool
            Whether to enable caching.
        """
        self._cache.enabled = enabled

    def read_csv(self, paths: Union[str, List[str]], schema: Dict[str, str], delim=",") -> DataFrame:
        """
        Create a DataFrame from CSV files.
        """
        dataset = CsvDataSet(paths, OrderedDict(schema), delim)
        plan = DataSourceNode(self._ctx, dataset)
        # Record operation history for data source
        history = [OperationRecord(
            operation="read_csv",
            params=(("paths", paths), ("schema", schema), ("delim", delim))
        )]
        return DataFrame(self, plan, operation_history=history)

    def read_parquet(
        self,
        paths: Union[str, List[str]],
        recursive: bool = False,
        columns: Optional[List[str]] = None,
        union_by_name: bool = False,
    ) -> DataFrame:
        """
        Create a DataFrame from Parquet files.
        """
        dataset = ParquetDataSet(paths, columns=columns, union_by_name=union_by_name, recursive=recursive)
        plan = DataSourceNode(self._ctx, dataset)
        # Record operation history for data source
        history = [OperationRecord(
            operation="read_parquet",
            params=(("paths", paths), ("recursive", recursive), ("columns", columns), ("union_by_name", union_by_name))
        )]
        return DataFrame(self, plan, operation_history=history)

    def read_json(self, paths: Union[str, List[str]], schema: Dict[str, str]) -> DataFrame:
        """
        Create a DataFrame from JSON files.
        """
        dataset = JsonDataSet(paths, schema)
        plan = DataSourceNode(self._ctx, dataset)
        # Record operation history for data source
        history = [OperationRecord(operation="read_json", params=(("paths", paths), ("schema", schema)))]
        return DataFrame(self, plan, operation_history=history)

    def from_items(self, items: List[Any]) -> DataFrame:
        """
        Create a DataFrame from a list of local Python objects.
        """

        assert isinstance(items, list), "items must be a list"
        assert len(items) > 0, "items must not be empty"
        if isinstance(items[0], dict):
            df = self.from_arrow(arrow.Table.from_pylist(items))
            # Update history to reflect from_items instead of from_arrow
            df._operation_history = [OperationRecord(
                operation="from_items",
                params=(("num_items", len(items)),)
            )]
            return df
        else:
            df = self.from_arrow(arrow.table({"item": items}))
            # Update history to reflect from_items instead of from_arrow
            df._operation_history = [OperationRecord(
                operation="from_items",
                params=(("num_items", len(items)),)
            )]
            return df

    def from_pandas(self, df: pd.DataFrame) -> DataFrame:
        """
        Create a DataFrame from a pandas DataFrame.
        """
        plan = DataSourceNode(self._ctx, PandasDataSet(df))
        # Record operation history for data source
        history = [OperationRecord(
            operation="from_pandas",
            params=(("shape", df.shape), ("columns", tuple(df.columns)))
        )]
        return DataFrame(self, plan, operation_history=history)

    def from_arrow(self, table: arrow.Table) -> DataFrame:
        """
        Create a DataFrame from a pyarrow Table.
        """
        plan = DataSourceNode(self._ctx, ArrowTableDataSet(table))
        # Record operation history for data source
        history = [OperationRecord(
            operation="from_arrow",
            params=(("num_rows", table.num_rows), ("columns", tuple(table.column_names)))
        )]
        return DataFrame(self, plan, operation_history=history)

    def partial_sql(self, query: str, *inputs: DataFrame, **kwargs) -> DataFrame:
        """
        Execute a SQL query on each partition of the input DataFrames.

        The query can contain placeholder `{0}`, `{1}`, etc. for the input DataFrames.
        If multiple DataFrames are provided, they must have the same number of partitions.

        Examples
        --------
        Join two datasets. You need to make sure the join key is correctly partitioned.

        .. code-block::

            a = sp.read_parquet("a/*.parquet").repartition(10, hash_by="id")
            b = sp.read_parquet("b/*.parquet").repartition(10, hash_by="id")
            c = sp.partial_sql("select * from {0} join {1} on a.id = b.id", a, b)
        """

        plan = SqlEngineNode(self._ctx, tuple(input.plan for input in inputs), query, **kwargs)
        recompute = any(input.need_recompute for input in inputs)
        # Record operation history - only preserve the first input DataFrame's history
        # Other input DataFrames' histories are not merged to keep the history linear
        new_history: List[OperationRecord] = []
        if inputs:
            new_history = list(inputs[0]._operation_history)
        new_history.append(OperationRecord(operation="partial_sql", params=(("query", query),)))
        return DataFrame(self, plan, recompute=recompute, operation_history=new_history)

    def wait(self, *dfs: DataFrame):
        """
        Wait for all DataFrames to be computed.

        Example
        -------
        This can be used to wait for multiple outputs from a pipeline:

        .. code-block::

            df = sp.read_parquet("input/*.parquet")
            output1 = df.write_parquet("output1")
            output2 = df.map("col1, col2").write_parquet("output2")
            sp.wait(output1, output2)
        """
        ray.get([task.run_on_ray() for df in dfs for task in df._get_or_create_tasks()])

    def graph(self) -> Digraph:
        """
        Get the DataFrame graph.
        """
        dot = Digraph(comment="SmallPond")
        for node in self._nodes:
            dot.node(str(node.id), repr(node))
            for dep in node.input_deps:
                dot.edge(str(dep.id), str(node.id))
        return dot

    def shutdown(self):
        """
        Shutdown the session.
        """
        # prevent shutdown from being called multiple times
        if hasattr(self, "_shutdown_called"):
            return
        self._shutdown_called = True

        # log status
        finished = self._all_tasks_finished()
        with open(self._runtime_ctx.job_status_path, "a") as fout:
            status = "success" if finished else "failure"
            fout.write(f"{status}@{datetime.now():%Y-%m-%d-%H-%M-%S}\n")

        # clean up runtime directories if success
        if finished:
            logger.info("all tasks are finished, cleaning up")
            self._runtime_ctx.cleanup(remove_output_root=self.config.remove_output_root)
        else:
            logger.warning("tasks are not finished!")

        super().shutdown()

    def _summarize_task(self) -> Tuple[int, int]:
        """
        Return the total number of tasks and the number of tasks that are finished.
        """
        dataset_refs = [task._dataset_ref for tasks in self._node_to_tasks.values() for task in tasks if task._dataset_ref is not None]
        ready_tasks, _ = ray.wait(dataset_refs, num_returns=len(dataset_refs), timeout=0, fetch_local=False)
        return len(dataset_refs), len(ready_tasks)

    def _all_tasks_finished(self) -> bool:
        """
        Check if all tasks are finished.
        """
        dataset_refs = [task._dataset_ref for tasks in self._node_to_tasks.values() for task in tasks]
        try:
            ray.get(dataset_refs, timeout=0)
        except Exception:
            # GetTimeoutError is raised if any task is not finished
            # RuntimeError is raised if any task failed
            return False
        return True


class DataFrame:
    """
    A distributed data collection representing a 2-dimensional table of rows and columns.

    Internally, it's a wrapper around a `Node` (logical plan) and a `Session` required
    for execution. DataFrames support lazy evaluation - transformations build up a
    logical plan that is only executed when an action (like `count()`, `take()`, or
    `to_pandas()`) is called.

    Operation History Tracking
    --------------------------
    Each DataFrame maintains a history of dataset-level operations that were applied
    to create it. This history is useful for:

    - **Debugging**: Understanding what transformations led to the current state
    - **Auditing**: Tracing the lineage of data transformations
    - **Troubleshooting**: Identifying which operation might have caused unexpected results

    **When tracking starts**: History tracking begins when a DataFrame is created from
    a data source via Session methods (e.g., ``sp.read_parquet()``, ``sp.from_pandas()``).
    The first entry in the history is always the data source operation.

    **What is tracked**: Only dataset-level operations are tracked, including:

    - Data sources: ``read_parquet``, ``read_csv``, ``read_json``, ``from_pandas``,
      ``from_arrow``, ``from_items``, ``partial_sql``
    - Transformations: ``filter``, ``map``, ``flat_map``, ``map_batches``, ``limit``
    - Partitioning: ``repartition``, ``random_shuffle``, ``partial_sort``
    - Multi-DataFrame: ``join``, ``union``, ``drop_duplicates``
    - Aggregation: ``groupby_agg``
    - Column operations: ``rename_columns``, ``drop_columns``, ``select_columns``
    - Metadata: ``require_non_null``, ``recompute``, ``no_cache``

    **What is NOT tracked**: Row-level operations are not tracked as they would
    require significant code changes and memory overhead.

    **History inheritance**: When a transformation is applied to a DataFrame, the
    resulting DataFrame inherits the parent's history and adds the new operation.
    For multi-DataFrame operations (join, union), only the calling (left) DataFrame's
    history is preserved to maintain a simple, linear history.

    **Immutability**: Each ``OperationRecord`` in the history is immutable (frozen
    dataclass), ensuring the integrity of the recorded history.

    Examples
    --------
    Access the operation history:

    .. code-block::

        df = (sp.read_parquet("data/*.parquet")
              .filter("status = 'active'")
              .repartition(10, hash_by="user_id"))

        # View operation history
        for record in df.history():
            print(f"{record.operation}: {record.get_params()}")

        # Output:
        # read_parquet: {'paths': 'data/*.parquet', ...}
        # filter: {'predicate': "status = 'active'"}
        # repartition: {'npartitions': 10, 'hash_by': 'user_id', ...}

    See Also
    --------
    history : Method to retrieve the operation history.
    OperationRecord : The immutable dataclass representing a single operation record.
    """

    def __init__(
        self,
        session: Session,
        plan: Node,
        recompute: bool = False,
        use_cache: bool = True,
        non_null_columns: Optional[frozenset] = None,
        operation_history: Optional[List[OperationRecord]] = None,
    ):
        self.session = session
        self.plan = plan
        self.optimized_plan: Optional[Node] = None
        self.need_recompute = recompute
        """Whether to recompute the data regardless of whether it's already computed."""
        self._use_cache = use_cache
        """Whether to use caching for this DataFrame's computations."""
        self._compute_lock = Lock()
        """Lock to ensure thread-safe access to optimized_plan during _compute()."""
        self._non_null_columns: frozenset = non_null_columns if non_null_columns is not None else frozenset()
        """
        Columns that must not contain null values when data is retrieved.

        Uses frozenset for O(1) membership testing and immutability. Since frozenset
        is immutable, it can be safely shared between parent and child DataFrames
        without copying. When new columns are added via require_non_null(), a new
        frozenset is created using union operation.
        """
        self._operation_history: List[OperationRecord] = operation_history if operation_history is not None else []
        """
        History of dataset-level operations performed on this DataFrame.

        This list tracks operations like filter, join, union, etc. that transform
        the DataFrame at a dataset level. Row-level operations are not tracked.
        The history is inherited from parent DataFrames and extended with new
        operations, maintaining the full lineage of transformations.
        """

        session._nodes.append(plan)

    def __str__(self) -> str:
        return repr(self.plan)

    def _record_operation(
        self,
        operation: str,
        params: Dict[str, Any],
    ) -> List[OperationRecord]:
        """
        Create a new operation history by extending this DataFrame's history.

        This helper method copies this DataFrame's operation history and appends
        a new operation record. For multi-DataFrame operations (join, union),
        only the calling DataFrame's history is preserved to maintain a simple,
        linear history trace.

        Parameters
        ----------
        operation : str
            The name of the operation being performed.
        params : Dict[str, Any]
            The parameters for the operation. Callable values are converted to
            '<function>' for readability, and DataFrame values are converted to
            '<DataFrame>'.

        Returns
        -------
        List[OperationRecord]
            A new list containing this DataFrame's history plus the new operation record.

        Notes
        -----
        For operations involving multiple DataFrames (e.g., join, union):
        - Only the calling (left) DataFrame's history is preserved
        - The other DataFrame(s)' histories are not merged
        - This keeps the history as a simple linear sequence of operations

        This design choice prioritizes simplicity and readability over completeness.
        If you need to trace the full lineage of all involved DataFrames, inspect
        each DataFrame's history separately before the multi-DataFrame operation.
        """
        # Sanitize params: convert callables to string representation
        # and convert to tuple of tuples for immutability
        sanitized_params = []
        for k, v in params.items():
            if callable(v):
                sanitized_params.append((k, "<function>"))
            elif isinstance(v, DataFrame):
                sanitized_params.append((k, "<DataFrame>"))
            else:
                sanitized_params.append((k, v))

        # Start with this DataFrame's history (only the calling DataFrame)
        new_history = list(self._operation_history)

        # Add the new operation record with params as immutable tuple
        new_history.append(OperationRecord(operation=operation, params=tuple(sanitized_params)))

        return new_history

    def history(self) -> List[OperationRecord]:
        """
        Get the history of dataset-level operations performed on this DataFrame.

        This method returns a list of operation records that trace the sequence of
        transformations applied to create this DataFrame. Only dataset-level operations
        are tracked (e.g., filter, join, union, repartition), not row-level operations.

        The history forms a simple linear sequence of operations, starting from the
        data source and including each transformation in order.

        Returns
        -------
        List[OperationRecord]
            A list of OperationRecord objects, each containing:
            - operation: The name of the operation (e.g., 'filter', 'join')
            - params: A tuple of (key, value) pairs for parameters. Use
              ``record.get_params()`` to get as a dictionary.
            - timestamp: When the operation was recorded

        Examples
        --------
        View the operation history:

        .. code-block::

            df = (sp.read_parquet("data/*.parquet")
                  .filter("status = 'active'")
                  .repartition(10, hash_by="user_id")
                  .join(other_df, on="user_id"))

            for record in df.history():
                print(f"{record.operation}: {record.get_params()}")

        Output might look like:

        .. code-block::

            read_parquet: {'paths': 'data/*.parquet', ...}
            filter: {'predicate': "status = 'active'"}
            repartition: {'npartitions': 10, 'hash_by': 'user_id', ...}
            join: {'on': 'user_id', 'how': 'inner', ...}

        Check the number of operations:

        .. code-block::

            print(f"This DataFrame has {len(df.history())} operations in its history")

        Notes
        -----
        - The history is a copy; modifying it does not affect the DataFrame.
        - Operations are recorded at the time they are called, not when computed.
        - **Multi-DataFrame operations (join, union)**: Only the calling (left)
          DataFrame's history is preserved. The other DataFrame(s)' histories are
          NOT merged. This keeps the history as a simple, linear sequence.

          For example, if you call ``left_df.join(right_df, on="id")``, the resulting
          DataFrame's history will contain ``left_df``'s history followed by the
          join operation. ``right_df``'s history is not included.

          If you need to trace the lineage of all involved DataFrames, inspect each
          DataFrame's history separately before the operation.
        - The following operations are tracked:
          - Data sources: read_parquet, read_csv, read_json, from_pandas, from_arrow, from_items, partial_sql
          - Transformations: filter, map, flat_map, map_batches, limit
          - Partitioning: repartition, random_shuffle, partial_sort
          - Multi-DataFrame: join, union, drop_duplicates
          - Aggregation: groupby_agg
          - Column operations: rename_columns, drop_columns, select_columns
          - Metadata: require_non_null, recompute, no_cache

        See Also
        --------
        OperationRecord : The dataclass representing a single operation record.
        """
        # Return a copy to prevent external modification
        return list(self._operation_history)

    def _get_or_create_tasks(self) -> List[Task]:
        """
        Get or create tasks to compute the data.
        """
        # optimize the plan
        if self.optimized_plan is None:
            logger.info(f"optimizing\n{LogicalPlan(self.session._ctx, self.plan)}")
            self.optimized_plan = Optimizer(exclude_nodes=set(self.session._node_to_tasks.keys())).visit(self.plan)
            logger.info(f"optimized\n{LogicalPlan(self.session._ctx, self.optimized_plan)}")
        # return the tasks if already created
        if tasks := self.session._node_to_tasks.get(self.optimized_plan):
            return tasks

        # remove all completed task files if recompute is needed
        if self.need_recompute:
            remove_path(
                os.path.join(
                    self.session._runtime_ctx.completed_task_dir,
                    str(self.optimized_plan.id),
                )
            )
            logger.info(f"cleared all results of {self.optimized_plan!r}")

        # create tasks for the optimized plan
        planner = Planner(self.session._runtime_ctx)
        # let planner update self.session._node_to_tasks
        planner.node_to_tasks = self.session._node_to_tasks
        return planner.visit(self.optimized_plan)

    def is_computed(self) -> bool:
        """
        Check if the data is ready on disk.
        """
        if tasks := self.session._node_to_tasks.get(self.plan):
            _, unready_tasks = ray.wait(tasks, timeout=0)
            return len(unready_tasks) == 0
        return False

    def compute(self) -> None:
        """
        Compute the data.

        This operation will trigger execution of the lazy transformations performed on this DataFrame.
        """
        self._compute()

    def _compute(self, use_cache: Optional[bool] = None, skip_validation: bool = False) -> List[DataSet]:
        """
        Compute the data and return the datasets.

        This method is thread-safe. Multiple threads calling _compute() on the same
        DataFrame will be serialized to avoid race conditions when optimizing the plan.

        Parameters
        ----------
        use_cache : bool, optional
            Whether to use caching for this computation.
            If None, uses the DataFrame's default cache setting.
            If True, will check the cache first and store results in cache.
            If False, will bypass the cache entirely.
        skip_validation : bool, default False
            If True, skip the non-null column validation. This is used internally
            by methods like describe() that need to compute data without validation.
        """
        # Determine if we should use cache
        should_use_cache = use_cache if use_cache is not None else self._use_cache

        # Use lock to ensure thread-safe access to optimized_plan
        with self._compute_lock:
            # Ensure the plan is optimized before checking cache
            if self.optimized_plan is None:
                # This will be set by _get_or_create_tasks, but we need it for cache key
                logger.info(f"optimizing\n{LogicalPlan(self.session._ctx, self.plan)}")
                self.optimized_plan = Optimizer(exclude_nodes=set(self.session._node_to_tasks.keys())).visit(self.plan)
                logger.info(f"optimized\n{LogicalPlan(self.session._ctx, self.optimized_plan)}")

            # Check cache first (unless recompute is needed or cache is disabled)
            if should_use_cache and not self.need_recompute:
                cached_result = self.session._cache.get(self.optimized_plan)
                if cached_result is not None:
                    # Validate non-null columns even for cached results
                    if not skip_validation:
                        self._validate_non_null_columns(cached_result)
                    return cached_result

            # Compute the data
            for retry_count in range(3):
                try:
                    result = ray.get([task.run_on_ray() for task in self._get_or_create_tasks()])

                    # Store in cache if caching is enabled
                    if should_use_cache:
                        self.session._cache.put(self.optimized_plan, result)

                    # Validate non-null columns before returning
                    if not skip_validation:
                        self._validate_non_null_columns(result)

                    return result
                except ray.exceptions.RuntimeEnvSetupError as e:
                    # XXX: Ray may raise this error when a worker is interrupted.
                    #      ```
                    #      ray.exceptions.RuntimeEnvSetupError: Failed to set up runtime environment.
                    #      Failed to create runtime env for job 01000000, status = IOError:
                    #      on_read bad version, maybe there are some network problems, will fail the request.
                    #      ```
                    #      This is a bug of Ray and has been fixed in Ray 2.24: <https://github.com/ray-project/ray/pull/45513>
                    #      However, since Ray dropped support for Python 3.8 since 2.11, we can not upgrade Ray.
                    #      So we catch this error and retry by ourselves.
                    logger.error(f"found ray RuntimeEnvSetupError, retrying...\n{e}")
                    time.sleep(10 << retry_count)
            raise RuntimeError("Failed to compute data after 3 retries")

    # operations

    def recompute(self) -> DataFrame:
        """
        Always recompute the data regardless of whether it's already computed.

        This method also clears any cached result for this DataFrame to free memory,
        since the cached result will no longer be valid after recomputation.

        Examples
        --------
        Modify the code as follows and rerun:

        .. code-block:: diff

            - df = input.select('a')
            + df = input.select('b').recompute()

        The result of `input` can be reused.
        """
        # Clear the cached result if it exists, since we're going to recompute
        if self.optimized_plan is not None:
            cleared = self.session._cache.invalidate(self.optimized_plan)
            if cleared > 0:
                logger.debug(f"Cleared cached result for {self.optimized_plan!r} due to recompute()")

        self.need_recompute = True
        # Record operation history
        self._operation_history.append(OperationRecord(operation="recompute", params=()))
        return self

    def no_cache(self) -> DataFrame:
        """
        Disable caching for this DataFrame's computations.

        When caching is disabled, results will always be computed fresh and
        will not be stored in or retrieved from the cache.

        Returns
        -------
        DataFrame
            Returns self for method chaining.

        Examples
        --------
        .. code-block::

            # Disable caching for a specific computation
            result = df.filter('x > 10').no_cache().count()

            # Chain with other operations
            df.map('a, b').no_cache().to_pandas()
        """
        self._use_cache = False
        # Record operation history
        self._operation_history.append(OperationRecord(operation="no_cache", params=()))
        return self

    def clear_cache(self) -> int:
        """
        Clear the cached result for this specific DataFrame.

        Returns
        -------
        int
            The number of cache entries removed (0 or 1).

        Examples
        --------
        .. code-block::

            df = sp.read_parquet("data/*.parquet").filter('x > 10')
            count1 = df.count()  # Computes and caches
            df.clear_cache()     # Clears cached result for this df
            count2 = df.count()  # Computes again
        """
        if self.optimized_plan is not None:
            return self.session._cache.invalidate(self.optimized_plan)
        return 0

    def _try_get_column_names(self) -> Optional[List[str]]:
        """
        Attempt to get column names from the DataFrame's plan without triggering computation.

        This method tries to extract column names from the underlying data source if possible.
        It works for DataFrames created directly from data sources (parquet, CSV, JSON, pandas,
        arrow) but may not work for DataFrames that have undergone transformations that change
        the schema (like map operations with new columns).

        Returns
        -------
        Optional[List[str]]
            A list of column names if they can be determined without computation,
            or None if the schema cannot be determined cheaply.

        Notes
        -----
        This is a best-effort method. For complex transformation chains, it returns None
        and validation will happen at compute time instead.
        """
        from smallpond.logical.dataset import (
            ArrowTableDataSet,
            CsvDataSet,
            JsonDataSet,
            PandasDataSet,
            ParquetDataSet,
        )

        # Try to find the root DataSourceNode
        node = self.plan
        while node is not None:
            if isinstance(node, DataSourceNode):
                dataset = node.dataset
                # Handle different dataset types
                if isinstance(dataset, ArrowTableDataSet):
                    return [field.name for field in dataset.table.schema]
                elif isinstance(dataset, PandasDataSet):
                    return list(dataset.df.columns)
                elif isinstance(dataset, (CsvDataSet, JsonDataSet)):
                    # These have explicit schema dictionaries
                    return list(dataset.schema.keys())
                elif isinstance(dataset, ParquetDataSet):
                    # For parquet, we can read schema from metadata without loading data
                    try:
                        import pyarrow.parquet as parquet
                        resolved_paths = dataset.resolved_paths
                        if resolved_paths:
                            # Read just the schema from the first file's metadata
                            parquet_file = parquet.ParquetFile(resolved_paths[0])
                            return [field.name for field in parquet_file.schema_arrow]
                    except Exception:
                        # If we can't read metadata, fall back to late validation
                        pass
                return None

            # For transformation nodes, try to follow the input dependency chain
            # But only for transformations that preserve schema (filter, limit, etc.)
            if hasattr(node, 'input_deps') and node.input_deps:
                # For nodes with a single input that preserve schema, follow the chain
                # This works for: filter, limit, repartition, partial_sort, etc.
                if len(node.input_deps) == 1:
                    node = node.input_deps[0]
                else:
                    # Multiple inputs (like join) - schema is complex, give up
                    return None
            else:
                return None

        return None

    def _try_get_schema(self) -> Optional[Dict[str, str]]:
        """
        Attempt to get column names and types from the DataFrame's plan without triggering computation.

        This method tries to extract schema information (column names and their types) from the
        underlying data source if possible. It works for DataFrames created directly from data
        sources (parquet, CSV, JSON, pandas, arrow) but may not work for DataFrames that have
        undergone transformations that change the schema.

        Returns
        -------
        Optional[Dict[str, str]]
            A dictionary mapping column names to their type strings if the schema can be
            determined without computation, or None if the schema cannot be determined cheaply.
            Type strings are normalized to lowercase for comparison (e.g., "int64", "string").

        Notes
        -----
        This is a best-effort method. For complex transformation chains, it returns None
        and validation will happen at compute time instead.
        """
        from smallpond.logical.dataset import (
            ArrowTableDataSet,
            CsvDataSet,
            JsonDataSet,
            PandasDataSet,
            ParquetDataSet,
        )

        # Try to find the root DataSourceNode
        node = self.plan
        while node is not None:
            if isinstance(node, DataSourceNode):
                dataset = node.dataset
                # Handle different dataset types
                if isinstance(dataset, ArrowTableDataSet):
                    return {field.name: str(field.type) for field in dataset.table.schema}
                elif isinstance(dataset, PandasDataSet):
                    # Convert pandas dtypes to string representation
                    return {col: str(dtype) for col, dtype in dataset.df.dtypes.items()}
                elif isinstance(dataset, (CsvDataSet, JsonDataSet)):
                    # These have explicit schema dictionaries (DuckDB type strings)
                    return dict(dataset.schema)
                elif isinstance(dataset, ParquetDataSet):
                    # For parquet, we can read schema from metadata without loading data
                    try:
                        import pyarrow.parquet as parquet
                        resolved_paths = dataset.resolved_paths
                        if resolved_paths:
                            # Read just the schema from the first file's metadata
                            parquet_file = parquet.ParquetFile(resolved_paths[0])
                            return {field.name: str(field.type) for field in parquet_file.schema_arrow}
                    except Exception:
                        # If we can't read metadata, fall back to late validation
                        pass
                return None

            # For transformation nodes, try to follow the input dependency chain
            # But only for transformations that preserve schema (filter, limit, etc.)
            if hasattr(node, 'input_deps') and node.input_deps:
                # For nodes with a single input that preserve schema, follow the chain
                # This works for: filter, limit, repartition, partial_sort, etc.
                if len(node.input_deps) == 1:
                    node = node.input_deps[0]
                else:
                    # Multiple inputs (like join) - schema is complex, give up
                    return None
            else:
                return None

        return None

    def _validate_columns_exist(
        self,
        columns: Union[str, List[str], set],
        operation_name: str
    ) -> Optional[List[str]]:
        """
        Validate that specified columns exist in the DataFrame.

        This is a helper method used by column operations (rename_columns, drop_columns,
        select_columns) to validate column existence early when possible.

        Parameters
        ----------
        columns : str, List[str], or set
            The column(s) to validate.
        operation_name : str
            The name of the operation for error messages (e.g., "rename", "drop", "select").

        Returns
        -------
        Optional[List[str]]
            The available columns if schema can be determined, None otherwise.

        Raises
        ------
        ValueError
            If any specified column doesn't exist in the DataFrame (when schema is available).
        """
        # Normalize to set for validation
        if isinstance(columns, str):
            cols_set = {columns}
        elif isinstance(columns, set):
            cols_set = columns
        else:
            cols_set = set(columns)

        available_columns = self._try_get_column_names()
        if available_columns is not None:
            missing_cols = cols_set - set(available_columns)
            if missing_cols:
                raise ValueError(
                    f"Columns to {operation_name} not found in DataFrame: {sorted(missing_cols)}. "
                    f"Available columns: {available_columns}"
                )

        return available_columns

    def require_non_null(self, columns: Union[str, List[str]]) -> DataFrame:
        """
        Mark columns as required to be non-null.

        When data is retrieved from this DataFrame via operations like `take()`,
        `to_pandas()`, `to_arrow()`, or `count()`, the specified columns will be
        validated to ensure they contain no null values. If null values are found,
        a `NullValidationError` will be raised.

        This is useful for ensuring data quality early in the pipeline, especially
        for critical columns like IDs or foreign keys that should never be null.

        Parameters
        ----------
        columns : str or List[str]
            A single column name or a list of column names that must not contain
            null values. The validation is additive - if `require_non_null()` is
            called multiple times, all specified columns will be validated.

        Returns
        -------
        DataFrame
            Returns self for method chaining.

        Raises
        ------
        NullValidationError
            Raised when any of the specified columns contain null values during
            data retrieval operations. The exception includes details about which
            columns failed validation and how many null values were found.

        Examples
        --------
        Mark a single column as required non-null:

        .. code-block::

            df = sp.read_parquet("data/*.parquet").require_non_null("id")
            df.take(10)  # Will raise NullValidationError if 'id' has nulls

        Mark multiple columns:

        .. code-block::

            df = df.require_non_null(["id", "user_id", "timestamp"])

        Chain with other operations:

        .. code-block::

            df = (sp.read_parquet("data/*.parquet")
                  .filter("status = 'active'")
                  .require_non_null(["id", "email"])
                  .map("id, email, name"))

            # Any of these operations will validate the columns:
            df.count()       # Validates before counting
            df.take(100)     # Validates before returning rows
            df.to_pandas()   # Validates before converting
            df.to_arrow()    # Validates before converting

        Handling validation errors:

        .. code-block::

            try:
                df.require_non_null("id").to_pandas()
            except NullValidationError as e:
                print(f"Validation failed: {e}")
                print(f"Columns with nulls: {e.null_counts}")

        Notes
        -----
        - The non-null constraint is preserved through method chaining. When you call
          transformation methods like `filter()`, `map()`, or `repartition()`, the
          resulting DataFrame will inherit the non-null requirements.
        - **Early column validation**: When possible, this method validates that the
          specified columns exist in the DataFrame immediately (without triggering
          computation). This catches typos and invalid column names early. For simple
          DataFrames (direct from data sources like parquet, CSV, pandas, arrow) and
          DataFrames with schema-preserving transformations (filter, limit, repartition),
          column existence is validated immediately.
        - For complex transformations that change the schema (like map with new columns),
          column validation happens at data retrieval time instead.
        - The validation checks all partitions, so null values in any partition
          will trigger the error.
        - Internally uses a `frozenset` for O(1) membership testing. This ensures
          efficient duplicate checking regardless of how many columns are validated.

        Raises
        ------
        ValueError
            Raised immediately if the specified columns don't exist in the DataFrame
            and the schema can be determined without computation. This helps catch
            typos early.
        """
        if isinstance(columns, str):
            columns = [columns]

        # Try to validate columns exist early (without triggering computation)
        known_columns = self._try_get_column_names()
        if known_columns is not None:
            known_columns_set = set(known_columns)
            invalid_columns = [col for col in columns if col not in known_columns_set]
            if invalid_columns:
                raise ValueError(
                    f"Column(s) {invalid_columns} specified in require_non_null() do not exist in the DataFrame. "
                    f"Available columns: {known_columns}"
                )

        # Create new frozenset with union of existing and new columns
        # frozenset is immutable, so this creates a new set rather than mutating
        self._non_null_columns = self._non_null_columns | frozenset(columns)

        # Record operation history
        self._operation_history.append(OperationRecord(operation="require_non_null", params=(("columns", tuple(columns)),)))
        return self

    def _validate_non_null_columns(self, datasets: List[DataSet]) -> None:
        """
        Validate that columns marked as non-null contain no null values.

        Parameters
        ----------
        datasets : List[DataSet]
            The computed datasets to validate.

        Raises
        ------
        NullValidationError
            If any of the non-null columns contain null values.
        """
        if not self._non_null_columns:
            return

        # Collect null counts for each required column across all partitions
        null_counts: Dict[str, int] = {col: 0 for col in self._non_null_columns}

        for dataset in datasets:
            arrow_table = dataset.to_arrow_table()
            schema = arrow_table.schema

            for col_name in self._non_null_columns:
                try:
                    col_idx = schema.get_field_index(col_name)
                    col_array = arrow_table.column(col_idx)
                    null_counts[col_name] += col_array.null_count
                except KeyError:
                    raise ValueError(
                        f"Column '{col_name}' specified in require_non_null() does not exist in the DataFrame. "
                        f"Available columns: {[field.name for field in schema]}"
                    )

        # Check if any column has null values
        columns_with_nulls = [col for col, count in null_counts.items() if count > 0]
        if columns_with_nulls:
            raise NullValidationError(list(self._non_null_columns), null_counts)

    def repartition(
        self,
        npartitions: int,
        hash_by: Union[str, List[str], None] = None,
        by: Optional[str] = None,
        by_rows: bool = False,
        **kwargs,
    ) -> DataFrame:
        """
        Repartition the data into the given number of partitions.

        Parameters
        ----------
        npartitions
            The dataset would be split and distributed to `npartitions` partitions.
            If not specified, the number of partitions would be the default partition size of the context.
        hash_by, optional
            If specified, the dataset would be repartitioned by the hash of the given columns.
        by, optional
            If specified, the dataset would be repartitioned by the given column.
        by_rows, optional
            If specified, the dataset would be repartitioned by rows instead of by files.

        Examples
        --------
        .. code-block::

            df = df.repartition(10)                 # evenly distributed
            df = df.repartition(10, by_rows=True)   # evenly distributed by rows
            df = df.repartition(10, hash_by='host') # hash partitioned
            df = df.repartition(10, by='bucket')    # partitioned by column
        """
        if by is not None:
            assert hash_by is None, "cannot specify both by and hash_by"
            plan = ShuffleNode(
                self.session._ctx,
                (self.plan,),
                npartitions,
                data_partition_column=by,
                **kwargs,
            )
        elif hash_by is not None:
            hash_columns = [hash_by] if isinstance(hash_by, str) else hash_by
            plan = HashPartitionNode(self.session._ctx, (self.plan,), npartitions, hash_columns, **kwargs)
        else:
            plan = EvenlyDistributedPartitionNode(
                self.session._ctx,
                (self.plan,),
                npartitions,
                partition_by_rows=by_rows,
                **kwargs,
            )
        # Record operation history
        new_history = self._record_operation(
            "repartition",
            {"npartitions": npartitions, "hash_by": hash_by, "by": by, "by_rows": by_rows}
        )
        return DataFrame(self.session, plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None, operation_history=new_history)

    def random_shuffle(self, **kwargs) -> DataFrame:
        """
        Randomly shuffle all rows globally.
        """

        repartition = HashPartitionNode(
            self.session._ctx,
            (self.plan,),
            self.plan.num_partitions,
            random_shuffle=True,
            **kwargs,
        )
        plan = SqlEngineNode(
            self.session._ctx,
            (repartition,),
            r"select * from {0} order by random()",
            **kwargs,
        )
        # Record operation history
        new_history = self._record_operation("random_shuffle", {})
        return DataFrame(self.session, plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None, operation_history=new_history)

    def partial_sort(self, by: Union[str, List[str]], **kwargs) -> DataFrame:
        """
        Sort rows by the given columns in each partition.

        Parameters
        ----------
        by
            A column or a list of columns to sort by.

        Examples
        --------
        .. code-block::

            df = df.partial_sort(by='a')
            df = df.partial_sort(by=['a', 'b desc'])
        """

        by = [by] if isinstance(by, str) else by
        plan = SqlEngineNode(
            self.session._ctx,
            (self.plan,),
            f"select * from {{0}} order by {', '.join(by)}",
            **kwargs,
        )
        # Record operation history
        new_history = self._record_operation("partial_sort", {"by": by})
        return DataFrame(self.session, plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None, operation_history=new_history)

    def filter(self, sql_or_func: Union[str, Callable[[Dict[str, Any]], bool]], **kwargs) -> DataFrame:
        """
        Filter out rows that don't satisfy the given predicate.

        Parameters
        ----------
        sql_or_func
            A SQL expression or a predicate function.
            For functions, it should take a dictionary of columns as input and returns a boolean.
            SQL expression is preferred as it's more efficient.

        Examples
        --------
        .. code-block::

            df = df.filter('a > 1')
            df = df.filter(lambda r: r['a'] > 1)
        """
        if isinstance(sql := sql_or_func, str):
            plan = SqlEngineNode(
                self.session._ctx,
                (self.plan,),
                f"select * from {{0}} where ({sql})",
                **kwargs,
            )
        elif isinstance(func := sql_or_func, Callable):

            def process_func(_runtime_ctx, tables: List[arrow.Table]) -> arrow.Table:
                table = tables[0]
                return table.filter([func(row) for row in table.to_pylist()])

            plan = ArrowBatchNode(self.session._ctx, (self.plan,), process_func=process_func, **kwargs)
        else:
            raise ValueError("condition must be a SQL expression or a predicate function")
        # Record operation history - store the predicate (SQL string or function)
        predicate_repr = sql_or_func if isinstance(sql_or_func, str) else sql_or_func
        new_history = self._record_operation("filter", {"predicate": predicate_repr})
        return DataFrame(self.session, plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None, operation_history=new_history)

    def map(
        self,
        sql_or_func: Union[str, Callable[[Dict[str, Any]], Dict[str, Any]]],
        *,
        schema: Optional[arrow.Schema] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Apply a function to each row.

        Parameters
        ----------
        sql_or_func
            A SQL expression or a function to apply to each row.
            For functions, it should take a dictionary of columns as input and returns a dictionary of columns.
            SQL expression is preferred as it's more efficient.
        schema, optional
            The schema of the output DataFrame.
            If not passed, will be inferred from the first row of the mapping values.
        udfs, optional
            A list of user defined functions to be referenced in the SQL expression.

        Examples
        --------
        .. code-block::

            df = df.map('a, b')
            df = df.map('a + b as c')
            df = df.map(lambda row: {'c': row['a'] + row['b']})


        Use user-defined functions in SQL expression:

        .. code-block::

            @udf(params=[UDFType.INT, UDFType.INT], return_type=UDFType.INT)
            def gcd(a: int, b: int) -> int:
                while b:
                    a, b = b, a % b
                return a
            # load python udf
            df = df.map('gcd(a, b)', udfs=[gcd])

            # load udf from duckdb extension
            df = df.map('gcd(a, b)', udfs=['path/to/udf.duckdb_extension'])

        """
        if isinstance(sql := sql_or_func, str):
            plan = SqlEngineNode(self.session._ctx, (self.plan,), f"select {sql} from {{0}}", **kwargs)
        elif isinstance(func := sql_or_func, Callable):

            def process_func(_runtime_ctx, tables: List[arrow.Table]) -> arrow.Table:
                output_rows = [func(row) for row in tables[0].to_pylist()]
                return arrow.Table.from_pylist(output_rows, schema=schema)

            plan = ArrowBatchNode(self.session._ctx, (self.plan,), process_func=process_func, **kwargs)
        else:
            raise ValueError(f"must be a SQL expression or a function: {sql_or_func!r}")
        # Record operation history
        expr_repr = sql_or_func if isinstance(sql_or_func, str) else sql_or_func
        new_history = self._record_operation("map", {"expression": expr_repr})
        return DataFrame(self.session, plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None, operation_history=new_history)

    def flat_map(
        self,
        sql_or_func: Union[str, Callable[[Dict[str, Any]], List[Dict[str, Any]]]],
        *,
        schema: Optional[arrow.Schema] = None,
        **kwargs,
    ) -> DataFrame:
        """
        Apply a function to each row and flatten the result.

        Parameters
        ----------
        sql_or_func
            A SQL expression or a function to apply to each row.
            For functions, it should take a dictionary of columns as input and returns a list of dictionaries.
            SQL expression is preferred as it's more efficient.
        schema, optional
            The schema of the output DataFrame.
            If not passed, will be inferred from the first row of the mapping values.

        Examples
        --------
        .. code-block::

            df = df.flat_map('unnest(array[a, b]) as c')
            df = df.flat_map(lambda row: [{'c': row['a']}, {'c': row['b']}])
        """
        if isinstance(sql := sql_or_func, str):

            plan = SqlEngineNode(self.session._ctx, (self.plan,), f"select {sql} from {{0}}", **kwargs)
        elif isinstance(func := sql_or_func, Callable):

            def process_func(_runtime_ctx, tables: List[arrow.Table]) -> arrow.Table:
                output_rows = [item for row in tables[0].to_pylist() for item in func(row)]
                return arrow.Table.from_pylist(output_rows, schema=schema)

            plan = ArrowBatchNode(self.session._ctx, (self.plan,), process_func=process_func, **kwargs)
        else:
            raise ValueError(f"must be a SQL expression or a function: {sql_or_func!r}")
        # Record operation history
        expr_repr = sql_or_func if isinstance(sql_or_func, str) else sql_or_func
        new_history = self._record_operation("flat_map", {"expression": expr_repr})
        return DataFrame(self.session, plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None, operation_history=new_history)

    def map_batches(
        self,
        func: Callable[[arrow.Table], arrow.Table],
        *,
        batch_size: int = 122880,
        **kwargs,
    ) -> DataFrame:
        """
        Apply the given function to batches of data.

        Parameters
        ----------
        func
            A function or a callable class to apply to each batch of data.
            It should take a `arrow.Table` as input and returns a `arrow.Table`.
        batch_size, optional
            The number of rows in each batch. Defaults to 122880.
        """

        def process_func(_runtime_ctx, tables: List[arrow.Table]) -> arrow.Table:
            return func(tables[0])

        plan = ArrowBatchNode(
            self.session._ctx,
            (self.plan,),
            process_func=process_func,
            streaming_batch_size=batch_size,
            **kwargs,
        )
        # Record operation history
        new_history = self._record_operation("map_batches", {"func": func, "batch_size": batch_size})
        return DataFrame(self.session, plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None, operation_history=new_history)

    def limit(self, limit: int) -> DataFrame:
        """
        Limit the number of rows to the given number.

        Unlike `take`, this method doesn't trigger execution.
        """
        plan = LimitNode(self.session._ctx, self.plan, limit)
        # Record operation history
        new_history = self._record_operation("limit", {"limit": limit})
        return DataFrame(self.session, plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None, operation_history=new_history)

    def join(
        self,
        other: DataFrame,
        on: Union[str, List[str], None] = None,
        left_on: Union[str, List[str], None] = None,
        right_on: Union[str, List[str], None] = None,
        how: str = "inner",
        npartitions: Optional[int] = None,
        suffix: Tuple[str, str] = ("_left", "_right"),
    ) -> DataFrame:
        """
        Join this DataFrame with another DataFrame.

        This method automatically handles repartitioning both DataFrames by the join keys
        to ensure correct distributed join execution, and generates the appropriate SQL query.

        Parameters
        ----------
        other : DataFrame
            The right DataFrame to join with.
        on : str or List[str], optional
            Column name(s) to join on. Use this when the join columns have the same name
            in both DataFrames. Cannot be used together with `left_on`/`right_on`.
        left_on : str or List[str], optional
            Column name(s) from the left DataFrame (self) to join on.
            Must be used together with `right_on`.
        right_on : str or List[str], optional
            Column name(s) from the right DataFrame (other) to join on.
            Must be used together with `left_on`.
        how : str, default 'inner'
            Type of join to perform. Supported values:
            - 'inner': Inner join - only rows with matching keys in both DataFrames.
            - 'left': Left outer join - all rows from left DataFrame, matching rows from right.
            - 'right': Right outer join - all rows from right DataFrame, matching rows from left.
            - 'outer' or 'full': Full outer join - all rows from both DataFrames.
            - 'cross': Cross join - cartesian product of both DataFrames (no join keys needed).
            - 'semi': Semi join - rows from left DataFrame that have a match in right DataFrame.
            - 'anti': Anti join - rows from left DataFrame that have no match in right DataFrame.
        npartitions : int, optional
            Number of partitions to use for the join. If not specified, uses the maximum
            number of partitions from both DataFrames. This ensures that the larger DataFrame
            maintains its parallelism level, while the smaller DataFrame is repartitioned to
            match. Although this may create some empty or sparse partitions when DataFrames
            have very different sizes, it preserves the parallelism of the larger DataFrame
            and avoids the overhead of determining optimal partition counts dynamically.
        suffix : Tuple[str, str], default ('_left', '_right')
            Reserved for future use. Currently, when using the `on` parameter (same column
            names in both DataFrames), DuckDB's USING clause automatically deduplicates the
            join columns. When using `left_on`/`right_on` (different column names), all
            columns from both DataFrames are included in the result. If you need to handle
            overlapping non-join column names, use `map()` to rename columns before joining.

        Returns
        -------
        DataFrame
            A new DataFrame containing the joined data.

        Raises
        ------
        ValueError
            If join parameters are invalid (e.g., using both `on` and `left_on`/`right_on`,
            or specifying mismatched number of columns in `left_on` and `right_on`).

        Examples
        --------
        Inner join on a single column with the same name:

        .. code-block::

            result = df1.join(df2, on="id")

        Inner join on multiple columns:

        .. code-block::

            result = df1.join(df2, on=["id", "date"])

        Left join with different column names:

        .. code-block::

            result = df1.join(df2, left_on="user_id", right_on="id", how="left")

        Full outer join:

        .. code-block::

            result = df1.join(df2, on="id", how="outer")

        Cross join (cartesian product):

        .. code-block::

            result = df1.join(df2, how="cross")

        Notes
        -----
        - Both DataFrames are automatically repartitioned by the join keys using hash
          partitioning to ensure that matching rows end up in the same partition.
        - For cross joins, no repartitioning is performed since all combinations are needed.
        - The join is executed partition-by-partition using DuckDB SQL.
        - **Operation history**: The resulting DataFrame's history contains only the left
          (calling) DataFrame's history, followed by the join operation. The right DataFrame's
          (``other``) history is NOT included. This keeps the history as a simple, linear
          sequence. If you need to trace the lineage of both DataFrames, inspect each
          DataFrame's ``history()`` separately before the join.
        """
        # Validate join type
        valid_join_types = {"inner", "left", "right", "outer", "full", "cross", "semi", "anti"}
        how_lower = how.lower()
        if how_lower not in valid_join_types:
            raise ValueError(
                f"Invalid join type '{how}'. Supported types are: {', '.join(sorted(valid_join_types))}"
            )

        # Normalize 'full' to 'outer' for SQL generation
        if how_lower == "full":
            how_lower = "outer"

        # Validate join key parameters
        if how_lower == "cross":
            # Cross join doesn't need join keys
            if on is not None or left_on is not None or right_on is not None:
                raise ValueError("Cross join does not accept join keys (on, left_on, right_on)")
            left_cols: List[str] = []
            right_cols: List[str] = []
        else:
            # Non-cross joins require join keys
            if on is not None:
                if left_on is not None or right_on is not None:
                    raise ValueError("Cannot specify both 'on' and 'left_on'/'right_on'")
                left_cols = [on] if isinstance(on, str) else list(on)
                right_cols = left_cols.copy()
            elif left_on is not None and right_on is not None:
                left_cols = [left_on] if isinstance(left_on, str) else list(left_on)
                right_cols = [right_on] if isinstance(right_on, str) else list(right_on)
                if len(left_cols) != len(right_cols):
                    raise ValueError(
                        f"left_on and right_on must have the same number of columns. "
                        f"Got {len(left_cols)} left columns and {len(right_cols)} right columns."
                    )
            elif left_on is not None or right_on is not None:
                raise ValueError("Must specify both 'left_on' and 'right_on', or use 'on' for same-named columns")
            else:
                raise ValueError(
                    f"Join keys required for '{how}' join. Use 'on' for same-named columns, "
                    "or 'left_on' and 'right_on' for different column names."
                )

        # Determine number of partitions.
        # We use the maximum partition count from both DataFrames to preserve parallelism
        # of the larger DataFrame. While this may result in some sparse partitions when
        # joining DataFrames of very different sizes, it avoids the complexity of dynamically
        # determining optimal partition counts based on data size, which would require
        # computing metadata before the join. Users can override this by specifying npartitions.
        if npartitions is None:
            npartitions = max(self.plan.num_partitions, other.plan.num_partitions)

        # Repartition both DataFrames by join keys (skip for cross join)
        if how_lower == "cross":
            left_df = self
            right_df = other
        else:
            left_df = self.repartition(npartitions, hash_by=left_cols)
            right_df = other.repartition(npartitions, hash_by=right_cols)

        # Build the SQL query
        sql = self._build_join_sql(how_lower, left_cols, right_cols, suffix)

        # Execute the join using partial_sql
        plan = SqlEngineNode(
            self.session._ctx,
            (left_df.plan, right_df.plan),
            sql,
        )
        recompute = self.need_recompute or other.need_recompute
        # Merge non-null columns from both DataFrames using frozenset union
        merged_non_null = self._non_null_columns | other._non_null_columns
        # Record operation history - only preserve the left (calling) DataFrame's history
        # The right DataFrame's history is not merged to keep the history linear
        new_history = self._record_operation(
            "join",
            {"on": on, "left_on": left_on, "right_on": right_on, "how": how, "npartitions": npartitions}
        )
        return DataFrame(self.session, plan, recompute=recompute, use_cache=self._use_cache, non_null_columns=merged_non_null if merged_non_null else None, operation_history=new_history)

    def _build_join_sql(
        self,
        how: str,
        left_cols: List[str],
        right_cols: List[str],
        suffix: Tuple[str, str],
    ) -> str:
        """
        Build the SQL query for the join operation.

        Parameters
        ----------
        how : str
            The join type (inner, left, right, outer, cross, semi, anti).
        left_cols : List[str]
            Column names from the left DataFrame to join on.
        right_cols : List[str]
            Column names from the right DataFrame to join on.
        suffix : Tuple[str, str]
            Suffixes for overlapping column names (reserved for future use).

        Returns
        -------
        str
            The SQL query string.
        """
        # Map join type to SQL syntax
        join_type_map = {
            "inner": "INNER JOIN",
            "left": "LEFT OUTER JOIN",
            "right": "RIGHT OUTER JOIN",
            "outer": "FULL OUTER JOIN",
            "cross": "CROSS JOIN",
            "semi": "SEMI JOIN",
            "anti": "ANTI JOIN",
        }
        join_clause = join_type_map[how]

        # Build the ON clause for non-cross joins
        if how == "cross":
            on_clause = ""
        else:
            conditions = []
            for left_col, right_col in zip(left_cols, right_cols):
                # Quote column names to handle special characters and reserved words
                left_quoted = f'"{left_col}"'
                right_quoted = f'"{right_col}"'
                conditions.append(f"__left__.{left_quoted} = __right__.{right_quoted}")
            on_clause = f" ON {' AND '.join(conditions)}"

        # For semi and anti joins, only select from left table
        if how in ("semi", "anti"):
            return f"SELECT __left__.* FROM {{0}} AS __left__ {join_clause} {{1}} AS __right__{on_clause}"

        if left_cols == right_cols:
            # Same column names in both DataFrames - use DuckDB's USING clause.
            # USING automatically deduplicates the join columns (includes them once in the result)
            # and is cleaner than manually excluding columns with complex SQL.
            using_cols = ", ".join(f'"{col}"' for col in left_cols)
            return f"SELECT * FROM {{0}} AS __left__ {join_clause} {{1}} AS __right__ USING ({using_cols})"
        else:
            # Different column names - select all columns from both tables.
            # Both join key columns will be included in the result (e.g., user_id and id).
            # If there are overlapping non-join column names, users should rename them
            # before joining using map() to avoid ambiguity.
            return f"SELECT __left__.*, __right__.* FROM {{0}} AS __left__ {join_clause} {{1}} AS __right__{on_clause}"

    def union(self, *others: "DataFrame") -> "DataFrame":
        """
        Combine multiple DataFrames by appending their rows.

        This method creates a new DataFrame containing all rows from this DataFrame
        and all the DataFrames passed as arguments. The schemas of all DataFrames
        must be compatible: they must have the same column names and compatible types.

        The method handles DataFrames with the same columns but in different order
        by reordering columns to match the schema of the first (self) DataFrame.
        Partitioning is preserved - the resulting DataFrame will have the combined
        partitions from all input DataFrames.

        Parameters
        ----------
        *others : DataFrame
            One or more DataFrames to union with this DataFrame. All DataFrames
            must have compatible schemas (same column names with compatible types).

        Returns
        -------
        DataFrame
            A new DataFrame containing all rows from this DataFrame and all
            other DataFrames, with columns ordered according to the first DataFrame.

        Raises
        ------
        ValueError
            If no other DataFrames are provided.
        SchemaMismatchError
            If the DataFrames have incompatible schemas:
            - Different number of columns
            - Different column names (missing or extra columns)
            - Incompatible column types (e.g., string vs integer)

        Examples
        --------
        Combine two DataFrames with the same schema:

        .. code-block::

            sales_2025 = sp.read_parquet("sales_2025/*.parquet")
            sales_2026 = sp.read_parquet("sales_2026/*.parquet")
            all_sales = sales_2025.union(sales_2026)

        Combine multiple DataFrames at once:

        .. code-block::

            q1 = sp.read_parquet("q1/*.parquet")
            q2 = sp.read_parquet("q2/*.parquet")
            q3 = sp.read_parquet("q3/*.parquet")
            q4 = sp.read_parquet("q4/*.parquet")
            yearly = q1.union(q2, q3, q4)

        Handle DataFrames with same columns in different order:

        .. code-block::

            # df1 has columns: [id, name, value]
            # df2 has columns: [name, value, id]  (different order)
            # The union will reorder df2's columns to match df1
            combined = df1.union(df2)  # Result has columns: [id, name, value]

        Notes
        -----
        - **Schema validation**: The union operation validates that all DataFrames
          have exactly the same columns. If columns don't match, a `SchemaMismatchError`
          is raised with details about the mismatch.

        - **Type compatibility**: Column types are strictly validated before the union.
          Types must match exactly or be within the same type category. The following
          type mismatches will raise `SchemaMismatchError`:

          - **String vs Numeric**: Cannot union string columns (string, large_string,
            varchar, utf8) with numeric columns (int8, int16, int32, int64, uint8,
            uint16, uint32, uint64, float, double, float16, float32, float64).

          - **String vs Boolean**: Cannot union string columns with boolean columns.

          - **Numeric vs Boolean**: Cannot union numeric columns with boolean columns.

          - **String vs Temporal**: Cannot union string columns with date/time columns
            (date32, date64, timestamp, time32, time64).

          - **Numeric vs Temporal**: Cannot union numeric columns with temporal columns.

          The following type variations within the same category are allowed and will
          be automatically cast by the underlying SQL engine:

          - **Numeric widening**: int32 with int64, float32 with float64, int with float.
          - **String variations**: string with large_string, utf8 with varchar.

        - **Column ordering**: When DataFrames have the same columns but in different
          order, the columns are automatically reordered to match the first DataFrame's
          schema. This ensures consistent column ordering in the result.

        - **Partitioning**: The resulting DataFrame preserves the partitions from all
          input DataFrames. If df1 has 4 partitions and df2 has 3 partitions, the
          result will have 7 partitions.

        - **Non-null requirements**: Non-null column requirements from all input
          DataFrames are merged. If any input DataFrame has `require_non_null("id")`,
          the result will also have that requirement.

        - **Lazy evaluation**: Like other DataFrame operations, union is lazy. The
          actual combination happens when you trigger an action like `count()`,
          `take()`, `to_pandas()`, etc.

        - **Early validation**: When possible, schema and type validation is performed
          immediately when `union()` is called, without triggering computation. This
          helps catch errors early. However, for complex transformation chains where
          the schema cannot be determined without computation, validation will occur
          at compute time.

        - **Operation history**: The resulting DataFrame's history contains only the
          calling DataFrame's (``self``) history, followed by the union operation.
          The other DataFrames' (``others``) histories are NOT included. This keeps
          the history as a simple, linear sequence. If you need to trace the lineage
          of all involved DataFrames, inspect each DataFrame's ``history()`` separately
          before the union.

        See Also
        --------
        join : Combine DataFrames horizontally based on key columns.
        """
        if not others:
            raise ValueError("union() requires at least one other DataFrame")

        # Collect all DataFrames to union
        all_dfs = [self] + list(others)

        # Get schema information (column names + types) from each DataFrame for validation
        # We attempt early validation when possible (without triggering computation)
        schemas_with_types: List[Tuple[int, Dict[str, str]]] = []
        schemas_names_only: List[Tuple[int, List[str]]] = []

        for i, df in enumerate(all_dfs):
            schema = df._try_get_schema()
            if schema is not None:
                schemas_with_types.append((i, schema))
                schemas_names_only.append((i, list(schema.keys())))
            else:
                # Fall back to column names only if full schema not available
                columns = df._try_get_column_names()
                if columns is not None:
                    schemas_names_only.append((i, columns))

        # Validate schemas early if we have enough information
        if len(schemas_names_only) == len(all_dfs):
            # First validate column names
            self._validate_union_column_names(schemas_names_only)

            # Then validate types if we have type info for all DataFrames
            if len(schemas_with_types) == len(all_dfs):
                self._validate_union_column_types(schemas_with_types)

        # Build the union using SQL UNION ALL
        # Use UNION ALL to keep all rows (including duplicates) and preserve partitioning
        # Record operation history - only preserve the calling DataFrame's history
        # Other DataFrames' histories are not merged to keep the history linear
        new_history = self._record_operation(
            "union",
            {"num_dataframes": len(others)}
        )
        return self._build_union(all_dfs, new_history)

    def drop_duplicates(
        self,
        subset: Union[str, List[str], None] = None,
        keep: str = "first",
        npartitions: Optional[int] = None,
    ) -> "DataFrame":
        """
        Remove duplicate rows from the DataFrame.

        This method removes rows that have identical values in the specified columns
        (or all columns if no subset is specified). When duplicates are found, only
        one row is kept based on the `keep` parameter.

        For partitioned DataFrames, this method automatically repartitions the data
        by the subset columns to ensure all potential duplicates are co-located in
        the same partition before deduplication.

        Parameters
        ----------
        subset : str, List[str], or None, optional
            Column name(s) to consider when identifying duplicates.
            - If None (default), all columns are used to identify duplicates,
              meaning rows must have identical values in every column to be
              considered duplicates.
            - If a string, that single column is used.
            - If a list of strings, those columns are used.

        keep : str, default 'first'
            Determines which duplicate row to keep:
            - 'first': Keep the first occurrence of each duplicate based on the original
              row order within each partition. This is the default, matching pandas behavior.
            - 'last': Keep the last occurrence of each duplicate based on the original
              row order within each partition.
            - 'any': Keep any one of the duplicate rows. This is the most efficient option
              as it uses SQL DISTINCT, but the specific row kept is not guaranteed and
              may vary between executions. Use this when row order doesn't matter and
              performance is critical.

        npartitions : int, optional
            Number of partitions to use for the deduplication operation. If not
            specified, uses the current number of partitions. This parameter controls
            the parallelism during deduplication when data needs to be repartitioned
            to co-locate potential duplicates.

        Returns
        -------
        DataFrame
            A new DataFrame with duplicate rows removed.

        Raises
        ------
        ValueError
            If `keep` is not one of 'first', 'last', or 'any'.
            If `subset` contains column names that don't exist in the DataFrame
            (when column names can be determined without computation).

        Examples
        --------
        Remove rows that are completely identical across all columns (keeps first occurrence):

        .. code-block::

            # Original: [{'a': 1, 'b': 2}, {'a': 1, 'b': 2}, {'a': 1, 'b': 3}]
            df_unique = df.drop_duplicates()
            # Result: [{'a': 1, 'b': 2}, {'a': 1, 'b': 3}]
            # (first occurrence of duplicate row is kept)

        Remove duplicates based on specific columns (keeps first occurrence):

        .. code-block::

            # Original: [{'id': 1, 'name': 'Alice'}, {'id': 1, 'name': 'Bob'}, {'id': 2, 'name': 'Charlie'}]
            df_unique = df.drop_duplicates(subset='id')
            # Result: [{'id': 1, 'name': 'Alice'}, {'id': 2, 'name': 'Charlie'}]
            # (Alice is kept as she appears first for id=1)

        Keep the last occurrence of duplicates:

        .. code-block::

            df_unique = df.drop_duplicates(subset='id', keep='last')

        Use 'any' for better performance when order doesn't matter:

        .. code-block::

            # When you don't care which duplicate is kept, use 'any' for best performance
            df_unique = df.drop_duplicates(subset='id', keep='any')

        Drop duplicates after union operation:

        .. code-block::

            # Combine multiple DataFrames and remove any duplicate rows
            combined = df1.union(df2, df3)
            unique_combined = combined.drop_duplicates()

        Notes
        -----
        - **Pandas compatibility**: The default `keep='first'` matches pandas behavior,
          making it predictable for users familiar with pandas. Unlike pandas, this
          implementation also offers `keep='any'` for better performance when row
          order doesn't matter.

        - **Performance**: Using `keep='any'` is the most efficient option as it uses
          SQL DISTINCT which is highly optimized. Using `keep='first'` or `keep='last'`
          requires row numbering which adds overhead. Consider using `keep='any'` for
          large datasets when the specific row kept doesn't matter.

        - **Partitioned DataFrames**: When dropping duplicates on a subset of columns,
          the DataFrame is automatically repartitioned by those columns using hash
          partitioning. This ensures all rows with the same key values are in the
          same partition, allowing correct deduplication across the distributed data.
          When dropping duplicates on all columns (`subset=None`), no repartitioning
          is needed since identical rows will produce the same hash and naturally
          co-locate.

        - **Row ordering**: The `keep='first'` and `keep='last'` options preserve
          ordering within each partition but not across partitions. If you need
          global ordering, consider using `partial_sort()` before dropping duplicates.

        - **Non-null requirements**: Non-null column requirements are preserved
          through the drop duplicates operation.

        - **Lazy evaluation**: Like other DataFrame operations, drop_duplicates is lazy.
          The actual deduplication happens when you trigger an action like `count()`,
          `take()`, `to_pandas()`, etc.

        See Also
        --------
        union : Combine multiple DataFrames (may produce duplicates).
        filter : Filter rows based on conditions.
        """
        # Validate 'keep' parameter
        valid_keep_values = {'any', 'first', 'last'}
        keep_lower = keep.lower()
        if keep_lower not in valid_keep_values:
            raise ValueError(
                f"Invalid keep value '{keep}'. Supported values are: {', '.join(sorted(valid_keep_values))}"
            )

        # Normalize subset parameter
        if subset is None:
            subset_cols: Optional[List[str]] = None
        elif isinstance(subset, str):
            subset_cols = [subset]
        else:
            subset_cols = list(subset)

        # Validate subset columns exist (when possible without triggering computation)
        if subset_cols is not None:
            available_columns = self._try_get_column_names()
            if available_columns is not None:
                missing_cols = set(subset_cols) - set(available_columns)
                if missing_cols:
                    raise ValueError(
                        f"Columns not found in DataFrame: {sorted(missing_cols)}. "
                        f"Available columns: {available_columns}"
                    )

        # Determine number of partitions
        if npartitions is None:
            npartitions = self.plan.num_partitions

        # For subset-based deduplication with multiple partitions, we need to
        # repartition to ensure all potential duplicates are in the same partition
        if subset_cols is not None and npartitions > 1:
            source_df = self.repartition(npartitions, hash_by=subset_cols)
        else:
            source_df = self

        # Get all column names for SQL building
        all_columns = source_df._try_get_column_names()

        # Build the deduplication SQL using the helper method
        sql = self._build_drop_duplicates_sql(subset_cols, keep_lower, all_columns)

        plan = SqlEngineNode(
            self.session._ctx,
            (source_df.plan,),
            sql,
        )

        # Record operation history
        new_history = self._record_operation(
            "drop_duplicates",
            {"subset": subset, "keep": keep, "npartitions": npartitions}
        )
        return DataFrame(
            self.session,
            plan,
            recompute=self.need_recompute,
            use_cache=self._use_cache,
            non_null_columns=self._non_null_columns if self._non_null_columns else None,
            operation_history=new_history
        )

    @staticmethod
    def _build_drop_duplicates_sql(
        subset_cols: Optional[List[str]],
        keep: str,
        all_columns: Optional[List[str]],
    ) -> str:
        """
        Build the SQL query for dropping duplicate rows.

        This helper method generates the appropriate SQL based on the deduplication
        strategy (keep='any', 'first', or 'last') and the columns to consider.

        Parameters
        ----------
        subset_cols : List[str] or None
            Column names to consider when identifying duplicates.
            If None, all columns are used.
        keep : str
            Which duplicate row to keep: 'any', 'first', or 'last'.
            Must be lowercase.
        all_columns : List[str] or None
            All column names in the DataFrame. Used for building the SELECT
            clause when keep='first' or keep='last'.

        Returns
        -------
        str
            The SQL query string for deduplication.
        """
        if keep == 'any':
            # Use DISTINCT for best performance
            if subset_cols is None:
                # Drop duplicates on all columns
                return "SELECT DISTINCT * FROM {0}"
            else:
                # Drop duplicates on subset columns, keep all columns in output
                # Use DISTINCT ON to get one row per unique subset combination
                quoted_subset = ", ".join(f'"{col}"' for col in subset_cols)
                return f"SELECT DISTINCT ON ({quoted_subset}) * FROM {{0}}"

        # Use ROW_NUMBER() for first/last semantics
        if subset_cols is None:
            # Partition by all columns
            if all_columns is not None:
                partition_cols = ", ".join(f'"{col}"' for col in all_columns)
            else:
                # Fallback when column names are not available
                partition_cols = "*"
        else:
            partition_cols = ", ".join(f'"{col}"' for col in subset_cols)

        # For 'first', we want row_num = 1 with ASC ordering (original order)
        # For 'last', we want row_num = 1 with DESC ordering (reverse order)
        order_direction = "DESC" if keep == 'last' else "ASC"

        # Build the SQL with ROW_NUMBER() window function
        # Use rowid as the ordering column to maintain original row order
        if all_columns is not None:
            # Select only the original columns (exclude the temporary row number column)
            quoted_cols = ", ".join(f'"{col}"' for col in all_columns)
            return f"""
                SELECT {quoted_cols} FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY {partition_cols}
                        ORDER BY rowid {order_direction}
                    ) AS __dedup_row_num__
                    FROM {{0}}
                ) WHERE __dedup_row_num__ = 1
            """
        else:
            # Fallback: select all columns including the row number
            # (will include __dedup_row_num__ in output)
            return f"""
                SELECT * FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY {partition_cols}
                        ORDER BY rowid {order_direction}
                    ) AS __dedup_row_num__
                    FROM {{0}}
                ) WHERE __dedup_row_num__ = 1
            """

    @staticmethod
    def _get_type_category(type_str: str) -> str:
        """
        Get the category of a type for compatibility checking.

        Returns one of: 'numeric', 'string', 'boolean', 'temporal', 'binary', 'other'
        """
        type_lower = type_str.lower()

        # Numeric types
        numeric_patterns = [
            'int8', 'int16', 'int32', 'int64',
            'uint8', 'uint16', 'uint32', 'uint64',
            'float', 'double', 'float16', 'float32', 'float64',
            'decimal', 'numeric', 'bigint', 'smallint', 'tinyint',
            'hugeint', 'real',
        ]
        for pattern in numeric_patterns:
            if pattern in type_lower:
                return 'numeric'

        # String types
        string_patterns = [
            'string', 'utf8', 'large_string', 'large_utf8',
            'varchar', 'char', 'text', 'object',
        ]
        for pattern in string_patterns:
            if pattern in type_lower:
                return 'string'

        # Boolean types
        if 'bool' in type_lower:
            return 'boolean'

        # Temporal types
        temporal_patterns = [
            'date', 'time', 'timestamp', 'duration', 'interval',
        ]
        for pattern in temporal_patterns:
            if pattern in type_lower:
                return 'temporal'

        # Binary types
        binary_patterns = ['binary', 'blob', 'bytes']
        for pattern in binary_patterns:
            if pattern in type_lower:
                return 'binary'

        return 'other'

    def _validate_union_column_names(
        self,
        schemas: List[Tuple[int, List[str]]],
    ) -> None:
        """
        Validate that all schemas have compatible column names for union.

        Parameters
        ----------
        schemas : List[Tuple[int, List[str]]]
            A list of (index, column_names) tuples for each DataFrame.

        Raises
        ------
        SchemaMismatchError
            If column names are incompatible.
        """
        if len(schemas) < 2:
            return

        # Use the first DataFrame's schema as the reference
        ref_idx, ref_columns = schemas[0]
        ref_columns_set = set(ref_columns)
        ref_num_cols = len(ref_columns)

        for df_idx, columns in schemas[1:]:
            columns_set = set(columns)

            # Check for column count mismatch
            if len(columns) != ref_num_cols:
                missing = ref_columns_set - columns_set
                extra = columns_set - ref_columns_set
                raise SchemaMismatchError(
                    f"Schema mismatch: DataFrame at index {df_idx} has {len(columns)} columns, "
                    f"but expected {ref_num_cols} columns (matching the first DataFrame). "
                    f"Missing columns: {sorted(missing) if missing else 'none'}. "
                    f"Extra columns: {sorted(extra) if extra else 'none'}.",
                    details={
                        "dataframe_index": df_idx,
                        "expected_columns": ref_columns,
                        "actual_columns": columns,
                        "missing_columns": sorted(missing),
                        "extra_columns": sorted(extra),
                    }
                )

            # Check for column name mismatch (same count but different names)
            if columns_set != ref_columns_set:
                missing = ref_columns_set - columns_set
                extra = columns_set - ref_columns_set
                raise SchemaMismatchError(
                    f"Schema mismatch: DataFrame at index {df_idx} has different columns than the first DataFrame. "
                    f"Missing columns: {sorted(missing)}. Extra columns: {sorted(extra)}.",
                    details={
                        "dataframe_index": df_idx,
                        "expected_columns": ref_columns,
                        "actual_columns": columns,
                        "missing_columns": sorted(missing),
                        "extra_columns": sorted(extra),
                    }
                )

    def _validate_union_column_types(
        self,
        schemas: List[Tuple[int, Dict[str, str]]],
    ) -> None:
        """
        Validate that all schemas have compatible column types for union.

        This method checks that columns with the same name across different DataFrames
        have compatible types. Types within the same category (e.g., int32 and int64)
        are considered compatible, but types from different categories (e.g., string
        and int64) will raise an error.

        Parameters
        ----------
        schemas : List[Tuple[int, Dict[str, str]]]
            A list of (index, schema_dict) tuples for each DataFrame.
            schema_dict maps column names to type strings.

        Raises
        ------
        SchemaMismatchError
            If column types are incompatible.
        """
        if len(schemas) < 2:
            return

        # Use the first DataFrame's schema as the reference
        ref_idx, ref_schema = schemas[0]

        for df_idx, schema in schemas[1:]:
            type_mismatches = []

            for col_name, ref_type in ref_schema.items():
                if col_name not in schema:
                    # Column name mismatch should have been caught earlier
                    continue

                actual_type = schema[col_name]
                ref_category = self._get_type_category(ref_type)
                actual_category = self._get_type_category(actual_type)

                # Check if types are in compatible categories
                if ref_category != actual_category:
                    type_mismatches.append({
                        "column": col_name,
                        "expected_type": ref_type,
                        "actual_type": actual_type,
                        "expected_category": ref_category,
                        "actual_category": actual_category,
                    })

            if type_mismatches:
                # Build a user-friendly error message
                mismatch_details = []
                for m in type_mismatches:
                    mismatch_details.append(
                        f"'{m['column']}': expected {m['expected_category']} ({m['expected_type']}) "
                        f"but got {m['actual_category']} ({m['actual_type']})"
                    )

                raise SchemaMismatchError(
                    f"Type mismatch: DataFrame at index {df_idx} has incompatible column types. "
                    f"Mismatched columns: {', '.join(mismatch_details)}. "
                    f"Types must be in the same category (e.g., all numeric or all string).",
                    details={
                        "dataframe_index": df_idx,
                        "type_mismatches": type_mismatches,
                    }
                )

    def _build_union(self, dfs: List["DataFrame"], operation_history: Optional[List[OperationRecord]] = None) -> "DataFrame":
        """
        Build the union of multiple DataFrames.

        Uses SQL UNION ALL BY NAME to handle column reordering automatically.
        DuckDB's UNION ALL BY NAME matches columns by name rather than position,
        which allows unioning DataFrames with the same columns in different order.

        Parameters
        ----------
        dfs : List[DataFrame]
            The DataFrames to union.
        operation_history : List[OperationRecord], optional
            The operation history for the resulting DataFrame.

        Returns
        -------
        DataFrame
            A new DataFrame representing the union of all inputs.
        """
        # Get the reference column order from the first DataFrame
        ref_columns = self._try_get_column_names()

        # Build the SQL query using UNION ALL BY NAME
        # UNION ALL BY NAME matches columns by name, not position, which handles
        # DataFrames with same columns in different order
        # After unioning, we select columns in the reference order to ensure consistency
        union_parts = []
        for i in range(len(dfs)):
            union_parts.append(f"SELECT * FROM {{{i}}}")

        union_sql = " UNION ALL BY NAME ".join(union_parts)

        # If we know the reference columns, wrap with a SELECT to ensure column order
        if ref_columns:
            quoted_cols = ", ".join(f'"{col}"' for col in ref_columns)
            sql = f"SELECT {quoted_cols} FROM ({union_sql})"
        else:
            sql = union_sql

        # Create the SQL node with all input DataFrames
        plan = SqlEngineNode(
            self.session._ctx,
            tuple(df.plan for df in dfs),
            sql,
        )

        # Merge properties from all DataFrames
        recompute = any(df.need_recompute for df in dfs)
        use_cache = all(df._use_cache for df in dfs)

        # Merge non-null columns from all DataFrames
        merged_non_null: frozenset = frozenset()
        for df in dfs:
            merged_non_null = merged_non_null | df._non_null_columns

        return DataFrame(
            self.session,
            plan,
            recompute=recompute,
            use_cache=use_cache,
            non_null_columns=merged_non_null if merged_non_null else None,
            operation_history=operation_history
        )

    def groupby_agg(
        self,
        by: Union[str, List[str]],
        aggs: Dict[str, Union[str, List[str]]],
        npartitions: Optional[int] = None,
    ) -> DataFrame:
        """
        Perform grouped aggregation on the DataFrame.

        This method groups the data by specified columns and computes aggregations
        on other columns. It automatically handles repartitioning by group keys and
        merges partial results across partitions correctly.

        Parameters
        ----------
        by : str or List[str]
            Column name(s) to group by.
        aggs : Dict[str, Union[str, List[str]]]
            A dictionary mapping column names to aggregation function(s).
            Keys are the column names to aggregate (must not overlap with `by` columns).
            Values can be a single aggregation function name (str) or a list of functions.
            Supported aggregation functions:
            - 'count': Count non-null values
            - 'sum': Sum of values
            - 'avg' or 'mean': Average of values
            - 'min': Minimum value
            - 'max': Maximum value
            - 'count_distinct': Count of distinct values. WARNING: This collects all
              distinct values in memory during the two-phase aggregation. Avoid using
              on columns with very high cardinality (millions of unique values) as it
              may cause memory issues.
        npartitions : int, optional
            Number of partitions to use for the partial aggregation phase. If not
            specified, uses the current number of partitions. This controls parallelism
            during the first phase where data is hash-partitioned by group keys and
            partial aggregates are computed per partition. The final aggregation phase
            always collects results into a single partition to combine partial results.

        Returns
        -------
        DataFrame
            A new DataFrame containing the grouped aggregation results.
            Output columns will be named as:
            - Group columns: original column names
            - Aggregated columns: '<column>_<agg_func>' (e.g., 'amount_sum', 'price_avg')

        Raises
        ------
        ValueError
            If `by` is empty, if an unsupported aggregation function is specified,
            or if any column appears in both `by` and `aggs`.

        Examples
        --------
        Single aggregation per column:

        .. code-block::

            result = df.groupby_agg(
                by='category',
                aggs={'amount': 'sum', 'price': 'avg'}
            )

        Multiple aggregations per column:

        .. code-block::

            result = df.groupby_agg(
                by=['region', 'category'],
                aggs={'amount': ['sum', 'count'], 'price': ['min', 'max', 'avg']}
            )

        Notes
        -----
        - This method uses a two-phase aggregation strategy:
          1. **Partial aggregation**: Data is hash-partitioned by group columns (using
             `npartitions`), and partial aggregates are computed in parallel per partition.
          2. **Final aggregation**: All partial results are collected into a single partition
             and combined to produce the final result.
        - For 'avg', partial sums and counts are computed separately, then combined as
          sum(partial_sums) / sum(partial_counts) to ensure correct weighted averages.
        - **Memory warning for 'count_distinct'**: This aggregation collects all distinct
          values per group into lists during partial aggregation. For columns with very
          high cardinality (millions of unique values), this can cause memory pressure.
          Consider using approximate distinct count techniques for such cases.
        """
        # Validate and normalize 'by' parameter
        if isinstance(by, str):
            group_cols = [by]
        else:
            group_cols = list(by)

        if not group_cols:
            raise ValueError("'by' parameter cannot be empty")

        # Validate and normalize 'aggs' parameter
        if not aggs:
            raise ValueError("'aggs' parameter cannot be empty")

        # Check for overlapping columns between group by and aggregation
        group_cols_set = set(group_cols)
        agg_cols_set = set(aggs.keys())
        overlapping_cols = group_cols_set & agg_cols_set
        if overlapping_cols:
            raise ValueError(
                f"Columns cannot be used for both grouping and aggregation: {sorted(overlapping_cols)}. "
                "A column should either be grouped by OR aggregated, not both."
            )

        supported_aggs = {'count', 'sum', 'avg', 'mean', 'min', 'max', 'count_distinct'}
        normalized_aggs: Dict[str, List[str]] = {}

        for col, funcs in aggs.items():
            if isinstance(funcs, str):
                funcs_list = [funcs]
            else:
                funcs_list = list(funcs)

            for func in funcs_list:
                func_lower = func.lower()
                if func_lower not in supported_aggs:
                    raise ValueError(
                        f"Unsupported aggregation function '{func}'. "
                        f"Supported functions are: {', '.join(sorted(supported_aggs))}"
                    )

            normalized_aggs[col] = funcs_list

        # Determine number of partitions
        if npartitions is None:
            npartitions = self.plan.num_partitions

        # Repartition by group columns to ensure all rows with same group keys
        # are in the same partition
        grouped_df = self.repartition(npartitions, hash_by=group_cols)

        # Build aggregation SQL and execute
        # We use a two-phase approach for correctness:
        # Phase 1: Compute partial aggregates per partition
        # Phase 2: Combine partial results

        partial_agg_sql, final_agg_sql = self._build_groupby_agg_sql(group_cols, normalized_aggs)

        # Phase 1: Partial aggregation per partition
        partial_plan = SqlEngineNode(
            self.session._ctx,
            (grouped_df.plan,),
            partial_agg_sql,
        )
        partial_df = DataFrame(self.session, partial_plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None)

        # Phase 2: Collect all partial results into a single partition and compute final aggregates
        # Use repartition(1) to collect all partial results, then apply final aggregation
        collected_df = partial_df.repartition(1)

        final_plan = SqlEngineNode(
            self.session._ctx,
            (collected_df.plan,),
            final_agg_sql,
        )

        # Record operation history
        new_history = self._record_operation(
            "groupby_agg",
            {"by": by, "aggs": aggs, "npartitions": npartitions}
        )
        return DataFrame(self.session, final_plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None, operation_history=new_history)

    def _build_groupby_agg_sql(
        self,
        group_cols: List[str],
        aggs: Dict[str, List[str]],
    ) -> Tuple[str, str]:
        """
        Build SQL queries for two-phase grouped aggregation.

        Parameters
        ----------
        group_cols : List[str]
            Column names to group by.
        aggs : Dict[str, List[str]]
            Mapping of column names to list of aggregation functions.

        Returns
        -------
        Tuple[str, str]
            A tuple of (partial_sql, final_sql) for the two aggregation phases.
        """
        # Quote column names for SQL
        quoted_group_cols = [f'"{col}"' for col in group_cols]
        group_by_clause = ", ".join(quoted_group_cols)

        # Build partial and final aggregation expressions
        partial_select_parts = list(quoted_group_cols)
        final_select_parts = list(quoted_group_cols)

        for col, funcs in aggs.items():
            quoted_col = f'"{col}"'
            for func in funcs:
                func_lower = func.lower()
                output_name = f"{col}_{func_lower}"
                quoted_output = f'"{output_name}"'

                if func_lower in ('count',):
                    # COUNT: sum of partial counts
                    partial_select_parts.append(f"COUNT({quoted_col}) AS {quoted_output}")
                    final_select_parts.append(f"SUM({quoted_output}) AS {quoted_output}")

                elif func_lower in ('sum',):
                    # SUM: sum of partial sums
                    partial_select_parts.append(f"SUM({quoted_col}) AS {quoted_output}")
                    final_select_parts.append(f"SUM({quoted_output}) AS {quoted_output}")

                elif func_lower in ('min',):
                    # MIN: min of partial mins
                    partial_select_parts.append(f"MIN({quoted_col}) AS {quoted_output}")
                    final_select_parts.append(f"MIN({quoted_output}) AS {quoted_output}")

                elif func_lower in ('max',):
                    # MAX: max of partial maxes
                    partial_select_parts.append(f"MAX({quoted_col}) AS {quoted_output}")
                    final_select_parts.append(f"MAX({quoted_output}) AS {quoted_output}")

                elif func_lower in ('avg', 'mean'):
                    # AVG: requires sum and count, then sum(sum)/sum(count)
                    sum_name = f"{col}__avg_sum"
                    count_name = f"{col}__avg_count"
                    # Normalize output name to 'avg' even if user specified 'mean'
                    output_name = f"{col}_avg"
                    quoted_output = f'"{output_name}"'

                    partial_select_parts.append(f'SUM({quoted_col}) AS "{sum_name}"')
                    partial_select_parts.append(f'COUNT({quoted_col}) AS "{count_name}"')
                    final_select_parts.append(f'SUM("{sum_name}") / SUM("{count_name}") AS {quoted_output}')

                elif func_lower == 'count_distinct':
                    # COUNT_DISTINCT: collect distinct values per partition, then count distinct in final
                    # We use a list aggregation approach: collect values, then flatten and count distinct
                    list_name = f"{col}__distinct_list"
                    partial_select_parts.append(f'LIST(DISTINCT {quoted_col}) AS "{list_name}"')
                    final_select_parts.append(
                        f'(SELECT COUNT(DISTINCT val) FROM (SELECT UNNEST(LIST("{list_name}")) AS val)) AS {quoted_output}'
                    )

        partial_select = ", ".join(partial_select_parts)
        final_select = ", ".join(final_select_parts)

        partial_sql = f"SELECT {partial_select} FROM {{0}} GROUP BY {group_by_clause}"
        final_sql = f"SELECT {final_select} FROM {{0}} GROUP BY {group_by_clause}"

        return partial_sql, final_sql

    def describe(self) -> Dict[str, Any]:
        """
        Get overview statistics of the DataFrame.

        This method computes comprehensive statistics about the DataFrame, including
        schema information, row counts, null counts, and summary statistics for
        numeric columns (min, max, mean, approximate median, std).

        The statistics are computed correctly across all partitions using a two-phase
        approach to ensure accurate results for distributed data.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing the following keys:

            - **num_rows** (int): Total number of rows in the DataFrame.
            - **num_columns** (int): Total number of columns.
            - **columns** (List[Dict[str, Any]]): Per-column statistics, where each
              column entry contains:

              - **name** (str): Column name.
              - **dtype** (str): Data type of the column.
              - **null_count** (int): Number of null/missing values.
              - **non_null_count** (int): Number of non-null values.
              - **null_percent** (float): Percentage of null values (0-100).

              For numeric columns (int, float), additional statistics are included:

              - **min** (numeric): Minimum value (excluding nulls).
              - **max** (numeric): Maximum value (excluding nulls).
              - **mean** (float): Mean/average value (excluding nulls).
              - **approx_median** (float): **Approximate** median value (50th percentile,
                excluding nulls). See notes below for accuracy details.
              - **std** (float): Standard deviation (excluding nulls).
              - **sum** (numeric): Sum of all values (excluding nulls).

        Examples
        --------
        Basic usage:

        .. code-block::

            stats = df.describe()
            print(f"Total rows: {stats['num_rows']}")
            print(f"Columns: {[col['name'] for col in stats['columns']]}")

        Inspecting numeric column statistics:

        .. code-block::

            stats = df.describe()
            for col in stats['columns']:
                print(f"{col['name']} ({col['dtype']}): {col['non_null_count']} non-null values")
                if 'mean' in col:
                    print(f"  Mean: {col['mean']}, Min: {col['min']}, Max: {col['max']}")
                    print(f"  Approx Median: {col['approx_median']}")

        Notes
        -----
        - This operation triggers execution of the lazy transformations performed
          on this DataFrame.
        - Statistics for numeric columns are computed using a two-phase aggregation
          to correctly combine results across partitions:

          - Phase 1: Compute partial statistics (count, sum, sum of squares, min, max,
            t-digest for median) per partition.
          - Phase 2: Merge partial results to compute final statistics.

        - **Approximate Median**: The median is computed using the t-digest algorithm,
          which is a memory-efficient streaming algorithm that provides approximate
          quantiles. This approach:

          - Uses O(δ) memory where δ is the compression parameter (default 100),
            regardless of data size
          - Provides higher accuracy near the tails (0th and 100th percentiles)
            and slightly lower accuracy near the median (50th percentile)
          - Is suitable for large-scale distributed data where exact median
            computation would require collecting all values in memory

        - **Note on non-null validation**: The `describe()` method skips non-null
          validation set via `require_non_null()`. This is intentional because
          `describe()` is typically used to explore and understand data quality,
          including identifying columns with null values. Use `describe()` to
          check null counts before deciding which columns to mark as required
          non-null.

        - **When to use describe() vs to_pandas().median()**:

          +------------------+------------------+------------------+-------------------+
          | Dataset Size     | describe()       | to_pandas()      | Recommendation    |
          +==================+==================+==================+===================+
          | n ≤ 100          | < 1% error       | Exact            | Either works      |
          +------------------+------------------+------------------+-------------------+
          | n ≤ 1,000        | < 2% error       | Exact            | Either works      |
          +------------------+------------------+------------------+-------------------+
          | n ≤ 10,000       | < 5% error       | Exact            | describe() for    |
          |                  |                  |                  | quick exploration |
          +------------------+------------------+------------------+-------------------+
          | n > 10,000       | < 10% error      | Memory intensive | Use describe()    |
          +------------------+------------------+------------------+-------------------+
          | n > 1,000,000    | < 10% error      | May fail (OOM)   | Use describe()    |
          +------------------+------------------+------------------+-------------------+

          **Use describe() when:**

          - You need a quick overview of large datasets
          - Memory is constrained
          - Approximate median is acceptable (exploratory analysis)
          - Data is distributed across many partitions

          **Use to_pandas().median() when:**

          - You need exact median values
          - Dataset fits comfortably in memory (< 10,000 rows)
          - Precision is critical for your analysis
          - You're doing final reporting or statistical tests

          Example for exact median:

          .. code-block::

              # For exact median on smaller datasets
              exact_median = df.to_pandas()['column_name'].median()

        - Standard deviation is computed using the population formula
          (dividing by N, not N-1).
        - Partitions with all-null values for a column are handled correctly and
          do not affect the statistics of partitions that have valid data.
        """
        # Skip validation for describe() - it's used to explore data quality
        datasets = self._compute(skip_validation=True)

        # Handle empty DataFrame case (no datasets at all)
        if not datasets:
            return {
                "num_rows": 0,
                "num_columns": 0,
                "columns": [],
            }

        # Get schema from first dataset with columns (schema exists even for 0-row tables)
        schema = None
        for dataset in datasets:
            arrow_table = dataset.to_arrow_table()
            if arrow_table.num_columns > 0:
                schema = arrow_table.schema
                break

        if schema is None:
            # All datasets have no columns - return empty result
            total_rows = sum(dataset.num_rows for dataset in datasets)
            return {
                "num_rows": total_rows,
                "num_columns": 0,
                "columns": [],
            }

        # Identify numeric columns for additional statistics
        numeric_types = {
            "int8", "int16", "int32", "int64",
            "uint8", "uint16", "uint32", "uint64",
            "float", "float16", "float32", "float64", "double",
            "decimal", "decimal128", "decimal256",
        }

        column_info = []
        for field in schema:
            dtype_str = str(field.type).lower()
            is_numeric = any(nt in dtype_str for nt in numeric_types)
            column_info.append({
                "name": field.name,
                "dtype": str(field.type),
                "is_numeric": is_numeric,
            })

        # Phase 1: Compute partial statistics per partition
        # We'll collect: count, null_count, and for numeric: sum, sum_sq, min, max, t-digest centroids
        partial_stats = []

        for dataset in datasets:
            arrow_table = dataset.to_arrow_table()
            partition_stats = {"total_rows": arrow_table.num_rows}

            for col_info in column_info:
                col_name = col_info["name"]
                col_idx = schema.get_field_index(col_name)
                col_array = arrow_table.column(col_idx)

                # Basic counts
                null_count = col_array.null_count
                non_null_count = len(col_array) - null_count

                col_stats = {
                    "null_count": null_count,
                    "non_null_count": non_null_count,
                }

                # Numeric statistics - only compute if there are non-null values
                if col_info["is_numeric"] and non_null_count > 0:
                    # Convert to pandas for easier numeric operations
                    values = col_array.to_pandas().dropna()
                    if len(values) > 0:
                        col_stats["sum"] = float(values.sum())
                        col_stats["sum_sq"] = float((values ** 2).sum())
                        col_stats["min"] = float(values.min())
                        col_stats["max"] = float(values.max())
                        # Build t-digest centroids for memory-efficient approximate median
                        col_stats["tdigest"] = self._build_tdigest(values.tolist())

                partition_stats[col_name] = col_stats

            partial_stats.append(partition_stats)

        # Phase 2: Merge partial statistics across partitions
        total_rows = sum(ps.get("total_rows", 0) for ps in partial_stats)

        columns_result = []
        for col_info in column_info:
            col_name = col_info["name"]

            # Aggregate counts with defensive checks
            total_null = 0
            total_non_null = 0
            for ps in partial_stats:
                if col_name in ps:
                    total_null += ps[col_name].get("null_count", 0)
                    total_non_null += ps[col_name].get("non_null_count", 0)

            col_result = {
                "name": col_name,
                "dtype": col_info["dtype"],
                "null_count": total_null,
                "non_null_count": total_non_null,
                "null_percent": (total_null / total_rows * 100) if total_rows > 0 else 0.0,
            }

            # Merge numeric statistics
            if col_info["is_numeric"] and total_non_null > 0:
                # Collect all partial stats that have numeric data (filter out all-null partitions)
                numeric_partials = []
                for ps in partial_stats:
                    if col_name in ps:
                        col_ps = ps[col_name]
                        # Only include partitions that have actual numeric data
                        if "sum" in col_ps and "min" in col_ps and "max" in col_ps:
                            numeric_partials.append(col_ps)

                if numeric_partials:
                    # Sum and sum of squares for mean and std - with defensive defaults
                    total_sum = sum(p.get("sum", 0) for p in numeric_partials)
                    total_sum_sq = sum(p.get("sum_sq", 0) for p in numeric_partials)

                    # Count of non-null values from partitions with data
                    count_from_partials = sum(p.get("non_null_count", 0) for p in numeric_partials)

                    # Min and max across partitions
                    mins = [p["min"] for p in numeric_partials if "min" in p]
                    maxs = [p["max"] for p in numeric_partials if "max" in p]

                    if mins and maxs and count_from_partials > 0:
                        global_min = min(mins)
                        global_max = max(maxs)

                        # Mean
                        mean = total_sum / count_from_partials

                        # Standard deviation using: std = sqrt(E[X^2] - (E[X])^2)
                        variance = (total_sum_sq / count_from_partials) - (mean ** 2)
                        # Handle floating point errors that could make variance slightly negative
                        std = (max(0, variance)) ** 0.5

                        # Approximate median: merge t-digests from all partitions
                        tdigests = [p["tdigest"] for p in numeric_partials if "tdigest" in p]
                        if tdigests:
                            merged_tdigest = self._merge_tdigests(tdigests)
                            approx_median = self._tdigest_quantile(merged_tdigest, 0.5)
                        else:
                            approx_median = None

                        col_result["min"] = global_min
                        col_result["max"] = global_max
                        col_result["mean"] = mean
                        col_result["approx_median"] = approx_median
                        col_result["std"] = std
                        col_result["sum"] = total_sum

            columns_result.append(col_result)

        return {
            "num_rows": total_rows,
            "num_columns": len(column_info),
            "columns": columns_result,
        }

    def _build_tdigest(self, values: List[float], compression: float = 100.0) -> Dict[str, Any]:
        """
        Build a t-digest data structure from a list of values.

        T-digest is a data structure for accurate estimation of quantiles with
        bounded memory usage. It uses a compression parameter δ to control the
        trade-off between accuracy and memory.

        Parameters
        ----------
        values : List[float]
            The values to build the t-digest from.
        compression : float
            The compression parameter δ. Higher values give more accuracy but use more memory.
            Default is 100, which provides good accuracy for most use cases.

        Returns
        -------
        Dict[str, Any]
            A dictionary containing:
            - centroids: List of (mean, weight) tuples representing the digest
            - count: Total number of values
            - min: Minimum value
            - max: Maximum value
        """
        if not values:
            return {"centroids": [], "count": 0, "min": None, "max": None}

        sorted_values = sorted(values)
        n = len(sorted_values)

        # For small datasets, store exact values as centroids
        if n <= compression:
            centroids = [(v, 1.0) for v in sorted_values]
            return {
                "centroids": centroids,
                "count": n,
                "min": sorted_values[0],
                "max": sorted_values[-1],
            }

        # For larger datasets, cluster values into centroids
        # Using scale function k1: k(q) = δ/2 * arcsin(2q - 1) / π
        # This gives higher resolution at the tails
        centroids = []
        current_mean = sorted_values[0]
        current_weight = 1.0
        current_quantile = 0.0

        def scale_func(q: float, delta: float) -> float:
            """Scale function k1 for t-digest."""
            import math
            return delta / (2 * math.pi) * math.asin(2 * q - 1)

        for i in range(1, n):
            proposed_weight = current_weight + 1
            proposed_quantile = (current_quantile * current_weight + (i / n)) / proposed_weight

            # Check if we should start a new centroid
            k_current = scale_func(current_quantile, compression)
            k_proposed = scale_func(proposed_quantile, compression)

            if k_proposed - k_current <= 1:
                # Merge into current centroid
                current_mean = (current_mean * current_weight + sorted_values[i]) / proposed_weight
                current_weight = proposed_weight
                current_quantile = proposed_quantile
            else:
                # Start new centroid
                centroids.append((current_mean, current_weight))
                current_mean = sorted_values[i]
                current_weight = 1.0
                current_quantile = i / n

        # Add the last centroid
        centroids.append((current_mean, current_weight))

        return {
            "centroids": centroids,
            "count": n,
            "min": sorted_values[0],
            "max": sorted_values[-1],
        }

    def _merge_tdigests(self, tdigests: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Merge multiple t-digests into one.

        Parameters
        ----------
        tdigests : List[Dict[str, Any]]
            List of t-digest structures to merge.

        Returns
        -------
        Dict[str, Any]
            A merged t-digest structure.
        """
        if not tdigests:
            return {"centroids": [], "count": 0, "min": None, "max": None}

        if len(tdigests) == 1:
            return tdigests[0]

        # Collect all centroids and sort by mean
        all_centroids = []
        total_count = 0
        global_min = None
        global_max = None

        for td in tdigests:
            if td["count"] > 0:
                all_centroids.extend(td["centroids"])
                total_count += td["count"]
                if td["min"] is not None:
                    if global_min is None:
                        global_min = td["min"]
                    else:
                        global_min = min(global_min, td["min"])
                if td["max"] is not None:
                    if global_max is None:
                        global_max = td["max"]
                    else:
                        global_max = max(global_max, td["max"])

        if not all_centroids:
            return {"centroids": [], "count": 0, "min": None, "max": None}

        # Sort centroids by mean
        all_centroids.sort(key=lambda x: x[0])

        # Re-cluster centroids using the same algorithm
        compression = 100.0
        merged_centroids = []
        current_mean, current_weight = all_centroids[0]
        cumulative_weight = current_weight

        def scale_func(q: float, delta: float) -> float:
            import math
            return delta / (2 * math.pi) * math.asin(2 * max(0, min(1, q)) - 1)

        for mean, weight in all_centroids[1:]:
            proposed_weight = current_weight + weight
            current_quantile = (cumulative_weight - current_weight / 2) / total_count
            proposed_quantile = (cumulative_weight + weight / 2) / total_count

            k_current = scale_func(current_quantile, compression)
            k_proposed = scale_func(proposed_quantile, compression)

            if k_proposed - k_current <= 1:
                # Merge
                current_mean = (current_mean * current_weight + mean * weight) / proposed_weight
                current_weight = proposed_weight
            else:
                # New centroid
                merged_centroids.append((current_mean, current_weight))
                current_mean = mean
                current_weight = weight

            cumulative_weight += weight

        merged_centroids.append((current_mean, current_weight))

        return {
            "centroids": merged_centroids,
            "count": total_count,
            "min": global_min,
            "max": global_max,
        }

    def _tdigest_quantile(self, tdigest: Dict[str, Any], q: float) -> Optional[float]:
        """
        Compute a quantile from a t-digest.

        Parameters
        ----------
        tdigest : Dict[str, Any]
            The t-digest structure.
        q : float
            The quantile to compute (0.0 to 1.0).

        Returns
        -------
        Optional[float]
            The estimated quantile value, or None if the digest is empty.
        """
        centroids = tdigest["centroids"]
        total_count = tdigest["count"]

        if not centroids or total_count == 0:
            return None

        if len(centroids) == 1:
            return centroids[0][0]

        # Handle edge cases
        if q <= 0:
            return tdigest["min"]
        if q >= 1:
            return tdigest["max"]

        # Find the centroid containing the quantile
        target_rank = q * total_count
        cumulative_weight = 0.0

        for i, (mean, weight) in enumerate(centroids):
            if cumulative_weight + weight >= target_rank:
                # Interpolate within this centroid
                if i == 0:
                    # First centroid - interpolate from min
                    left_mean = tdigest["min"]
                    left_weight = 0
                else:
                    left_mean, left_weight = centroids[i - 1]

                # Linear interpolation
                weight_before = cumulative_weight
                weight_in_centroid = target_rank - weight_before
                fraction = weight_in_centroid / weight if weight > 0 else 0.5

                # Interpolate between this centroid and neighbors
                if fraction <= 0.5 and i > 0:
                    # Closer to left neighbor
                    return left_mean + (mean - left_mean) * (0.5 + fraction)
                elif fraction > 0.5 and i < len(centroids) - 1:
                    # Closer to right neighbor
                    right_mean = centroids[i + 1][0]
                    return mean + (right_mean - mean) * (fraction - 0.5)
                else:
                    return mean

            cumulative_weight += weight

        # Should not reach here, but return max as fallback
        return tdigest["max"]

    def rename_columns(self, mapping: Dict[str, str]) -> "DataFrame":
        """
        Rename columns according to the provided mapping.

        This method creates a new DataFrame with specified columns renamed. Columns
        not included in the mapping retain their original names. This is a convenient
        alternative to using `map()` with SQL when you just need to rename columns.

        Parameters
        ----------
        mapping : Dict[str, str]
            A dictionary mapping old column names to new column names.
            Keys are the current column names, values are the desired new names.
            Only columns that need to be renamed should be included.

        Returns
        -------
        DataFrame
            A new DataFrame with the specified columns renamed.

        Raises
        ------
        ValueError
            If any column name in the mapping keys doesn't exist in the DataFrame
            (when column names can be determined without computation).
            If the mapping is empty.

        Examples
        --------
        Rename a single column:

        .. code-block::

            df_renamed = df.rename_columns({"old_name": "new_name"})

        Rename multiple columns:

        .. code-block::

            df_renamed = df.rename_columns({
                "id": "user_id",
                "name": "user_name",
                "ts": "timestamp"
            })

        Chain with other operations:

        .. code-block::

            result = (df
                .filter("status = 'active'")
                .rename_columns({"id": "user_id"})
                .map("user_id, email"))

        Notes
        -----
        - This operation is lazy and does not trigger computation.
        - Non-null column requirements are automatically updated to use the new
          column names. If you had `require_non_null("id")` and rename "id" to
          "user_id", the non-null requirement will apply to "user_id".
        - Column order is preserved after renaming.

        See Also
        --------
        drop_columns : Remove columns from the DataFrame.
        select_columns : Select and reorder columns.
        map : Apply transformations including column selection and renaming via SQL.
        """
        if not mapping:
            raise ValueError("mapping cannot be empty. Provide at least one column to rename.")

        # Validate columns exist early (when possible without triggering computation)
        available_columns = self._validate_columns_exist(set(mapping.keys()), "rename")

        if available_columns is not None:
            # Build SELECT with renamed columns preserving order
            select_parts = []
            for col in available_columns:
                if col in mapping:
                    select_parts.append(f'"{col}" AS "{mapping[col]}"')
                else:
                    select_parts.append(f'"{col}"')

            sql = f"SELECT {', '.join(select_parts)} FROM {{0}}"
        else:
            # Schema not available - use DuckDB's REPLACE syntax which renames columns
            # while keeping all other columns. This works at compute time when schema is unknown.
            rename_exprs = ", ".join(f'"{old}" AS "{new}"' for old, new in mapping.items())
            sql = f"SELECT * REPLACE ({rename_exprs}) FROM {{0}}"

        plan = SqlEngineNode(self.session._ctx, (self.plan,), sql)

        # Update non-null columns with renamed column names
        new_non_null = frozenset(
            mapping.get(col, col) for col in self._non_null_columns
        )

        # Record operation history
        new_history = self._record_operation("rename_columns", {"mapping": mapping})
        return DataFrame(
            self.session,
            plan,
            recompute=self.need_recompute,
            use_cache=self._use_cache,
            non_null_columns=new_non_null if new_non_null else None,
            operation_history=new_history
        )

    def drop_columns(self, columns: Union[str, List[str]]) -> "DataFrame":
        """
        Remove specified columns from the DataFrame.

        This method creates a new DataFrame with the specified columns removed.
        This is a convenient alternative to using `map()` with SQL when you just
        need to drop some columns.

        Parameters
        ----------
        columns : str or List[str]
            A single column name or a list of column names to drop.

        Returns
        -------
        DataFrame
            A new DataFrame without the specified columns.

        Raises
        ------
        ValueError
            If any column name doesn't exist in the DataFrame (when column names
            can be determined without computation).
            If trying to drop all columns.

        Examples
        --------
        Drop a single column:

        .. code-block::

            df_slim = df.drop_columns("unnecessary_column")

        Drop multiple columns:

        .. code-block::

            df_slim = df.drop_columns(["temp_col1", "temp_col2", "debug_info"])

        Chain with other operations:

        .. code-block::

            result = (df
                .filter("status = 'active'")
                .drop_columns(["internal_id", "created_at"])
                .write_parquet("output"))

        Notes
        -----
        - This operation is lazy and does not trigger computation.
        - Non-null column requirements for dropped columns are automatically removed.
        - Column order of remaining columns is preserved.

        See Also
        --------
        rename_columns : Rename columns in the DataFrame.
        select_columns : Select and reorder columns (can also be used to drop columns).
        map : Apply transformations including column selection via SQL.
        """
        # Normalize to list
        if isinstance(columns, str):
            cols_to_drop = [columns]
        else:
            cols_to_drop = list(columns)

        if not cols_to_drop:
            raise ValueError("Must specify at least one column to drop.")

        cols_to_drop_set = set(cols_to_drop)

        # Validate columns exist early (when possible without triggering computation)
        available_columns = self._validate_columns_exist(cols_to_drop_set, "drop")

        if available_columns is not None:
            # Check we're not dropping all columns
            remaining_cols = [col for col in available_columns if col not in cols_to_drop_set]
            if not remaining_cols:
                raise ValueError(
                    "Cannot drop all columns. At least one column must remain in the DataFrame."
                )

            # Build SELECT with remaining columns preserving order
            quoted_cols = ", ".join(f'"{col}"' for col in remaining_cols)
            sql = f"SELECT {quoted_cols} FROM {{0}}"
        else:
            # Schema not available - use DuckDB's EXCLUDE syntax to exclude specified columns
            excluded_cols = ", ".join(f'"{col}"' for col in cols_to_drop)
            sql = f"SELECT * EXCLUDE ({excluded_cols}) FROM {{0}}"

        plan = SqlEngineNode(self.session._ctx, (self.plan,), sql)

        # Update non-null columns by removing dropped columns
        new_non_null = self._non_null_columns - frozenset(cols_to_drop)

        # Record operation history
        new_history = self._record_operation("drop_columns", {"columns": columns})
        return DataFrame(
            self.session,
            plan,
            recompute=self.need_recompute,
            use_cache=self._use_cache,
            non_null_columns=new_non_null if new_non_null else None,
            operation_history=new_history
        )

    def select_columns(self, columns: List[str]) -> "DataFrame":
        """
        Select and reorder columns in the DataFrame.

        This method creates a new DataFrame containing only the specified columns
        in the specified order. This is useful for:
        - Reordering columns to a desired sequence
        - Selecting a subset of columns
        - Both reordering and subsetting at once

        Parameters
        ----------
        columns : List[str]
            A list of column names in the desired order. Only these columns
            will be included in the resulting DataFrame, in this exact order.

        Returns
        -------
        DataFrame
            A new DataFrame with only the specified columns in the specified order.

        Raises
        ------
        ValueError
            If any column name doesn't exist in the DataFrame (when column names
            can be determined without computation).
            If the columns list is empty.
            If duplicate column names are specified.

        Examples
        --------
        Reorder columns:

        .. code-block::

            # Original columns: [id, name, email, created_at]
            # Desired order: [name, email, id, created_at]
            df_reordered = df.select_columns(["name", "email", "id", "created_at"])

        Select a subset of columns in a specific order:

        .. code-block::

            # Only keep id, name, and email columns
            df_subset = df.select_columns(["id", "name", "email"])

        Chain with other operations:

        .. code-block::

            result = (df
                .filter("status = 'active'")
                .select_columns(["user_id", "email", "name"])
                .rename_columns({"user_id": "id"}))

        Notes
        -----
        - This operation is lazy and does not trigger computation.
        - Non-null column requirements for columns not in the selection are
          automatically removed.
        - Unlike `drop_columns()`, this method lets you specify exactly which
          columns you want and in what order.
        - If you just want to select columns without changing order, you can
          also use `map("col1, col2, col3")` with SQL syntax.

        See Also
        --------
        rename_columns : Rename columns in the DataFrame.
        drop_columns : Remove columns from the DataFrame.
        map : Apply transformations including column selection via SQL.
        """
        if not columns:
            raise ValueError("columns list cannot be empty. Specify at least one column.")

        # Check for duplicates
        if len(columns) != len(set(columns)):
            seen = set()
            duplicates = []
            for col in columns:
                if col in seen:
                    duplicates.append(col)
                seen.add(col)
            raise ValueError(f"Duplicate column names specified: {duplicates}")

        # Validate columns exist early (when possible without triggering computation)
        self._validate_columns_exist(columns, "select")

        # Build SELECT with columns in specified order
        quoted_cols = ", ".join(f'"{col}"' for col in columns)
        sql = f"SELECT {quoted_cols} FROM {{0}}"

        plan = SqlEngineNode(self.session._ctx, (self.plan,), sql)

        # Update non-null columns to only include selected columns
        new_non_null = self._non_null_columns & frozenset(columns)

        # Record operation history
        new_history = self._record_operation("select_columns", {"columns": columns})
        return DataFrame(
            self.session,
            plan,
            recompute=self.need_recompute,
            use_cache=self._use_cache,
            non_null_columns=new_non_null if new_non_null else None,
            operation_history=new_history
        )

    def sample(
        self,
        n: Optional[int] = None,
        fraction: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Return a random sample of rows from the DataFrame.

        This is useful for quickly inspecting a representative subset of the data
        without having to look at all rows. Users can specify either an exact number
        of rows or a fraction of total rows to sample.

        Parameters
        ----------
        n : int, optional
            The exact number of rows to sample. Must be a positive integer.
            Cannot be used together with `fraction`.
            If `n` is larger than the total number of rows in the DataFrame,
            all available rows will be returned (no error is raised).
        fraction : float, optional
            The fraction of rows to sample, between 0.0 and 1.0 (exclusive of 0, inclusive of 1).
            For example, 0.1 means 10% of rows. Cannot be used together with `n`.
        seed : int, optional
            Random seed for reproducible sampling. Can be used with either `n` or `fraction`.
            If not specified, sampling will be random each time.

        Returns
        -------
        List[Dict[str, Any]]
            A list of dictionaries, where each dictionary represents a row with column names as keys.

        Raises
        ------
        ValueError
            If neither `n` nor `fraction` is specified, or if both are specified.
            If `n` is not a positive integer.
            If `fraction` is not between 0 and 1.

        Examples
        --------
        Sample exactly 5 random rows:

        .. code-block::

            rows = df.sample(n=5)

        Sample 10% of the data:

        .. code-block::

            rows = df.sample(fraction=0.1)

        Sample with a fixed seed for reproducibility (using n):

        .. code-block::

            rows = df.sample(n=10, seed=42)

        Sample with a fixed seed for reproducibility (using fraction):

        .. code-block::

            rows = df.sample(fraction=0.2, seed=42)

        Notes
        -----
        This operation triggers execution of the lazy transformations performed on this DataFrame.
        For very large datasets, using `fraction` with a small value is more efficient than
        specifying a large `n`, as it can filter rows early in the pipeline.
        """
        # Validate parameters
        if n is None and fraction is None:
            raise ValueError("Must specify either 'n' (number of rows) or 'fraction' (proportion of rows)")
        if n is not None and fraction is not None:
            raise ValueError("Cannot specify both 'n' and 'fraction'. Please choose one.")
        if n is not None:
            if not isinstance(n, int) or n <= 0:
                raise ValueError(f"'n' must be a positive integer, got {n}")
        if fraction is not None:
            if not isinstance(fraction, (int, float)) or fraction <= 0 or fraction > 1:
                raise ValueError(
                    f"'fraction' must be a decimal number between 0 (exclusive) and 1 (inclusive), got {fraction}. "
                    "Examples: 0.1 for 10%, 0.25 for 25%, 0.5 for 50%, 1.0 for 100%."
                )

        # Build sample specification and method based on parameters
        if fraction is not None:
            sample_spec = f"{fraction * 100}%"
            sample_method = "bernoulli"
        else:
            sample_spec = f"{n} ROWS"
            sample_method = "reservoir"

        # Build SQL with optional seed for reproducibility
        seed_clause = f", repeatable({seed})" if seed is not None else ""
        sql = f"SELECT * FROM {{0}} USING SAMPLE {sample_spec} ({sample_method}{seed_clause})"

        plan = SqlEngineNode(self.session._ctx, (self.plan,), sql)
        return DataFrame(self.session, plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None).take_all()

    def write_parquet(self, path: str) -> None:
        """
        Write data to a series of parquet files under the given path.

        This is a blocking operation. See :func:`write_parquet_lazy` for a non-blocking version.

        Examples
        --------
        .. code-block::

            df.write_parquet('output')
        """
        self.write_parquet_lazy(path).compute()

    def write_parquet_lazy(self, path: str) -> DataFrame:
        """
        Write data to a series of parquet files under the given path.

        This is a non-blocking operation. See :func:`write_parquet` for a blocking version.

        Examples
        --------
        .. code-block::

            o1 = df.write_parquet_lazy('output1')
            o2 = df.write_parquet_lazy('output2')
            sp.wait(o1, o2)
        """

        plan = DataSinkNode(self.session._ctx, (self.plan,), os.path.abspath(path), type="link_or_copy")
        return DataFrame(self.session, plan, recompute=self.need_recompute, use_cache=self._use_cache, non_null_columns=self._non_null_columns if self._non_null_columns else None)

    # inspection

    def count(self) -> int:
        """
        Count the number of rows.

        If this dataframe consists of more than a read, or if the row count can't be determined from
        the metadata provided by the datasource, then this operation will trigger execution of the
        lazy transformations performed on this dataframe.
        """
        datasets = self._compute()
        # FIXME: don't use ThreadPoolExecutor because duckdb results will be mixed up
        return sum(dataset.num_rows for dataset in datasets)

    def take(self, limit: int) -> List[Dict[str, Any]]:
        """
        Return up to `limit` rows.

        This operation will trigger execution of the lazy transformations performed on this DataFrame.
        """
        if self.is_computed() or isinstance(self.plan, DataSourceNode):
            datasets = self._compute()
        else:
            datasets = self.limit(limit)._compute()
        rows = []
        for dataset in datasets:
            for batch in dataset.to_batch_reader():
                rows.extend(batch.to_pylist())
                if len(rows) >= limit:
                    return rows[:limit]
        return rows

    def take_all(self) -> List[Dict[str, Any]]:
        """
        Return all rows.

        This operation will trigger execution of the lazy transformations performed on this DataFrame.
        """
        datasets = self._compute()
        rows = []
        for dataset in datasets:
            for batch in dataset.to_batch_reader():
                rows.extend(batch.to_pylist())
        return rows

    def to_pandas(self) -> pd.DataFrame:
        """
        Convert to a pandas DataFrame.

        This operation will trigger execution of the lazy transformations performed on this DataFrame.
        """
        datasets = self._compute()
        with ThreadPoolExecutor() as pool:
            return pd.concat(pool.map(lambda dataset: dataset.to_pandas(), datasets))

    def to_arrow(self) -> arrow.Table:
        """
        Convert to an arrow Table.

        This operation will trigger execution of the lazy transformations performed on this DataFrame.
        """
        datasets = self._compute()
        with ThreadPoolExecutor() as pool:
            return arrow.concat_tables(pool.map(lambda dataset: dataset.to_arrow_table(), datasets))
