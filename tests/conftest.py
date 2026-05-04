import pytest
from dotenv import load_dotenv

# Load .env before any test runs
def pytest_configure(config):
    load_dotenv()