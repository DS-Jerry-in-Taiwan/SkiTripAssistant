import sys
import os
sys.path.append(os.path.abspath(os.path.dirname(__file__) + "/../../.."))
import pytest
print("Current working directory:", os.getcwd())
print("nodes import state:", __file__)
import state
from state import AgentState
import nodes


@pytest.fixture
def state():
    return AgentState(
        user_input="台中滑雪",
        days=2,
        budget_level="中等",
        origin="台中",
        destination="雪山滑雪場",
        mode="transit",
        location="台中",
        date="2025-01-15",
        checkin="2025-01-15",
        checkout="2025-01-17",
        required_fields=["days", "budget_level", "date", "location"]
    )

def test_update_recent_queries(state):
    result = nodes.update_recent_queries(state, "查詢滑雪場")
    assert "查詢滑雪場" in result["recent_queries"]

def test_get_merged_query(state):
    result = nodes.update_recent_queries(state, "查詢溫泉")
    merged = nodes.get_merged_query(result)
    assert isinstance(merged, str)

def test_dynamic_k_by_query(state):
    k = nodes.dynamic_k_by_query(state)
    assert isinstance(k, int)

def test_search_attractions_tool_wrapper(state):
    result = nodes.search_attractions_tool_wrapper(state)
    assert "attraction_result" in result

def test_rag_retrieval_tool_wrapper(state):
    result = nodes.rag_retrieval_tool_wrapper(state)
    assert "retriever_result" in result

def test_format_itinerary_data_wrapper(state):
    result = nodes.format_itinerary_data_wrapper(state)
    assert "formatted_data" in result

def test_calculate_budget_wrapper(state):
    result = nodes.calculate_budget_wrapper(state)
    assert "budget_result" in result

def test_calculate_route_wrapper(state):
    result = nodes.calculate_route_wrapper(state)
    assert "route_result" in result

def test_get_weather_wrapper(state):
    result = nodes.get_weather_wrapper(state)
    assert "weather_result" in result

def test_search_accommodation_wrapper(state):
    result = nodes.search_accommodation_wrapper(state)
    assert "accommodation_result" in result

def test_parse_user_preferences_wrapper(state):
    result = nodes.parse_user_preferences_wrapper(state)
    assert "parsed_preferences" in result