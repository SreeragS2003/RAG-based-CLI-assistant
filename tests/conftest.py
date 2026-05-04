import pytest
import asyncio
from dotenv import load_dotenv

# Load .env before any test runs
def pytest_configure(config):
    load_dotenv()

@pytest.fixture(autouse=True)
def slow_down_tests():
    yield
    asyncio.get_event_loop().run_until_complete(asyncio.sleep(3))  # 3s between tests