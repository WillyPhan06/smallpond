import os

import pytest
import ray

import smallpond
from smallpond.dataframe import get_default_cache


@pytest.fixture(scope="session")
def ray_address():
    """A global Ray instance for all tests"""
    ray_address = ray.init(
        address="local",
        # disable dashboard in unit tests
        include_dashboard=False,
    ).address_info["gcs_address"]
    yield ray_address
    ray.shutdown()


@pytest.fixture
def sp(ray_address: str, request):
    """A smallpond session for each test"""
    # Clear global cache before each test to ensure test isolation
    cache = get_default_cache()
    cache.invalidate()
    cache.reset_stats()
    cache.enabled = True  # Ensure caching is enabled

    runtime_root = os.getenv("TEST_RUNTIME_ROOT") or f"tests/runtime"
    sp = smallpond.init(
        data_root=os.path.join(runtime_root, request.node.name),
        ray_address=ray_address,
    )
    yield sp
    sp.shutdown()
