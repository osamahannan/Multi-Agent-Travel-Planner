import asyncio
import json
import re
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.types import interrupt

from config import get_llm
from mcp_client import current_weather, forecast, list_airlines, list_airports, tavily_search
from state import TravelState

llm = get_llm()

def _unwrap_mcp_json(raw_result: Any) -> Any:
  payload = raw_result
  if isinstance(payload, list) and payload and isinstance(payload[0], dict):
    text = payload[0].get("text")
    if isinstance(text, str):
      try:
        return json.loads(text)
      except json.JSONDecodeError:
        return {"error": text}
  return payload

def _format_hotel_results(raw_result: Any) -> str:
  payload = _unwrap_mcp_json(raw_result)

  if not isinstance(payload, dict):
    return "Hotel search did not return structured results."

  results = payload.get("results", [])
  if not isinstance(results, list) or not results:
    return "No hotel results were returned. Verify the destination and travel dates."

  lines = ["### Hotel areas and accommodation options", ""]
  for index, item in enumerate(results[:5], start=1):
    if not isinstance(item, dict):
      continue
    title = item.get("title", f"Hotel search result {index}")
    url = item.get("url", "")
    content = " ".join(str(item.get("content", "")).split())
    if len(content) > 500:
      content = content[:497] + "..."
    heading = f"{index}. [{title}]({url})" if url else f"{index}. {title}"
    lines.extend([heading, content or "No summary was provided.", ""])

  return "\n".join(lines).strip()

def _format_weather_results(current: Any, forecast_data: Any) -> str:
  current = _unwrap_mcp_json(current)
  forecast_data = _unwrap_mcp_json(forecast_data)
  lines = ["### Weather", ""]
  if isinstance(current, dict) and "error" not in current:
    lines.extend([
      f"- **City:** {current.get('city', 'Unknown')}",
      f"- **Temperature:** {current.get('temperature_c', 'N/A')} °C",
      f"- **Feels like:** {current.get('feels_like_c', 'N/A')} °C",
      f"- **Condition:** {current.get('condition', 'N/A')}",
      f"- **Humidity:** {current.get('humidity', 'N/A')}%",
      "",
    ])
  else:
    lines.extend([f"- Current weather unavailable: {current}", ""])

  forecast_items = forecast_data.get("forecast", []) if isinstance(forecast_data, dict) else []
  if forecast_items:
    lines.extend(["### Forecast", "", "| Date/time | Temperature | Condition |", "|---|---:|---|"])
    for item in forecast_items[:5]:
      lines.append(
        f"| {item.get('datetime', 'N/A')} | "
        f"{item.get('temperature', 'N/A')} °C | "
        f"{item.get('weather', 'N/A')} |"
      )
  else:
    lines.append(f"Forecast unavailable: {forecast_data}")

  return "\n".join(lines)

def _normalize_constraints(constraints: dict[str, Any]) -> dict[str, Any]:
  normalized = dict(constraints)
  destination = str(normalized.get("destination", "")).strip()
  if destination.lower() in {"europe", "european union", "eu"}:
    normalized["destination"] = "Prague and Budapest"
    normalized["destination_cities"] = ["Prague", "Budapest"]
    normalized["route_reason"] = (
      "Selected Prague and Budapest as a compact, budget-oriented Central "
      "European route; confirm current weather and prices before booking."
    )
  else:
    normalized["destination_cities"] = [destination] if destination else []
  return normalized

def _flight_constraint_requested(query: str) -> bool:
  query_lower = query.lower()
  return any(
    term in query_lower
    for term in ("flight", "flights", "overnight flight", "airfare", "airline", "airport")
  )

def _budget_inr(constraints: dict[str, Any]) -> float | None:
  match = re.search(r"([\d.]+)\s*(?:lakh|lakhs|l)", str(constraints.get("budget", "")).lower())
  if match:
    return float(match.group(1)) * 100000
  match = re.search(r"(?:₹|rs\.?|inr)\s*([\d,]+)", str(constraints.get("budget", "")).lower())
  return float(match.group(1).replace(",", "")) if match else None

def _max_flight_estimate_inr(text: str) -> float | None:
  text_lower = text.lower()
  range_matches = re.findall(
    r"(?:₹|rs\.?|inr)?\s*([\d.]+)\s*(?:-|–|to)\s*"
    r"(?:₹|rs\.?|inr)?\s*([\d.]+)\s*(?:lakh|lakhs|l)\b",
    text_lower,
  )
  if range_matches:
    return max(float(high) for _, high in range_matches) * 100000

  values = re.findall(
    r"(?:₹|rs\.?|inr)\s*([\d.]+)\s*(?:lakh|lakhs|l)?\b",
    text_lower,
  )
  amounts = []
  for value in values:
    amount = float(value)
    amounts.append(amount * 100000 if amount < 1000 else amount)
  return max(amounts) if amounts else None

def _final_response_is_grounded(response: str, state: TravelState) -> bool:
  constraints = state.get("trip_constraints", {})
  destination = str(constraints.get("destination", "")).strip().lower()
  destination_cities = [
    str(city).strip().lower()
    for city in constraints.get("destination_cities", [])
    if str(city).strip()
  ]
  duration = str(constraints.get("duration", "")).strip().lower()
  response_lower = response.lower()

  if destination_cities and not all(city in response_lower for city in destination_cities):
    return False
  if destination and not destination_cities and destination not in response_lower:
    return False
  if duration and not any(
    marker in response_lower
    for marker in (duration, duration.replace(" ", "-"), duration.replace(" days", "-day"), "7 days", "7-day")
  ):
    return False
  if any(token in str(state.get("user_query", "")).lower() for token in ("₹", "rs.", "inr", "rupee")):
    if not any(token in response_lower for token in ("₹", "inr", "rs.", "rupee")):
      return False
  if "flight" in str(state.get("user_query", "")).lower():
    if not any(token in response_lower for token in ("flight", "airline", "airport")):
      return False
  return True

def _llm_text(system: str, prompt: str) -> str:
  response = llm.invoke(
    [
        SystemMessage(content=system),
        HumanMessage(content=prompt),
    ]
  )
  return response.content

def _json_from_llm(text: str) -> dict:
  print("\n========== RAW LLM RESPONSE ==========")
  print(text)
  print("======================================\n")

  start = text.index("{")
  end = text.rindex("}") + 1

  json_text = text[start:end]

  print("\n========== EXTRACTED JSON ==========")
  print(json_text)
  print("====================================\n")

  return json.loads(json_text)

def supervisor_agent(state: TravelState) -> TravelState:
  query = state["user_query"]

  guardrail_prompt = f"""

  Classify whether the user request is specifically about planning,
  organizing, or receiving advice for a trip or travel.

  Allowed examples:
  - Plan a trip to Japan
  - Find hotels in Paris
  - What is the weather in London for my vacation?
  - Estimate the budget for a Europe trip

  Reject requests unrelated to travel, including general questions,
  programming, machine learning, recipes, news, or casual conversation.

  Return only valid JSON in this format:
  {{
    "allowed": true or false,
    "reason": "short explanation"
  }}

  User request:
  {query}
  """

  guardrail_response = _llm_text(
    "You are an input validation guardrail. Return strict JSON only.",
    guardrail_prompt,
  )

  print("\n========== GUARDRAIL RAW RESPONSE ==========")
  print(guardrail_response)
  print("============================================\n")

  guardrail_result = _json_from_llm(guardrail_response)

  print("\n========== GUARDRAIL PARSED JSON ==========")
  print(json.dumps(guardrail_result, indent=2))
  print("==========================================\n")

  if not guardrail_result.get("allowed", False):

    reason = guardrail_result.get("reason", "Request rejected by input guardrail.")

    return {
      "guardrail_blocked": True,
      "guardrail_message": reason,
      "selected_agents": [],
      "trip_constraints": {},
      "supervisor_reasoning": reason,
      "final_response": reason,
      "messages": [AIMessage(content=f"Guardrail blocked request: {reason}")],
      "llm_calls": state.get("llm_calls", 0) + 1
    }


  prompt = f"""
  You are the supervisor of a real-world multi-agent travel planning system.

  Decide which specialist agents are needed for this user request.

  Available agents:
  - flight_agent: use when flights, airports, airlines, routes, or airfare guidance are needed
  - hotel_agent: use when hotels, stays, neighborhoods, or accommodation are needed
  - weather_agent: use when weather, climate, season, packing, or forecast is useful
  - budget_agent: use when budget, affordability, cost, or price constraints are mentioned
  - itinerary_agent: almost always needed to produce the travel plan

  Return only JSON with this schema:
  {{
    "selected_agents": ["flight_agent", "hotel_agent", "weather_agent", "budget_agent", "itinerary_agent"],
    "trip_constraints": {{
      "destination": "",
      "origin": "",
      "duration": "",
      "budget": "",
      "travel_style": "",
      "special_preferences": []
    }},
    "reasoning": ""
  }}

  User request:
  {query}
  """

  raw = _llm_text(
    "You route work to specialist agents. Return strict JSON only.",
    prompt,
  )

  print("\n========== RAW LLM RESPONSE ==========")
  print(raw)
  print("======================================\n")

  parsed = _json_from_llm(raw)

  #parsed = json.loads(raw)

  print("\n========== PARSED JSON ==========")
  print(json.dumps(parsed, indent=2))
  print("=================================\n")

  print(type(raw))
  print(type(parsed))

  selected_agents = list(parsed["selected_agents"])
  if _flight_constraint_requested(query) and "flight_agent" not in selected_agents:
    selected_agents.append("flight_agent")
  query_lower = query.lower()
  if any(term in query_lower for term in ("hotel", "accommodation", "stay")) and "hotel_agent" not in selected_agents:
    selected_agents.append("hotel_agent")
  if any(term in query_lower for term in ("weather", "climate", "season", "forecast")) and "weather_agent" not in selected_agents:
    selected_agents.append("weather_agent")
  if "budget_agent" not in selected_agents:
    selected_agents.append("budget_agent")
  if "itinerary_agent" not in selected_agents:
    selected_agents.append("itinerary_agent")
  constraints = _normalize_constraints(parsed["trip_constraints"])
  route_note = constraints.get("route_reason", "")

  return {
    "selected_agents": selected_agents,
    "trip_constraints": constraints,
    "supervisor_reasoning": f"{parsed['reasoning']} {route_note}".strip(),
    "messages": [AIMessage(content="Supervisor created the agent plan.")],
    "llm_calls": state.get("llm_calls", 0) + 1,
  }


def flight_agent(state: TravelState):
  query = state["user_query"]
  constraints = state["trip_constraints"]
  destination = ", ".join(constraints.get("destination_cities", [constraints["destination"]]))

  print("\n========== FLIGHT AGENT INPUT ==========")
  print("Query:", query)
  print("Constraints:", constraints)
  print("========================================\n")

  airports = asyncio.run(list_airports(destination, limit=10))
  airlines = asyncio.run(list_airlines("", limit=10))

  print("\n========== AIRPORT MCP DATA ==========")
  print(airports)
  print("======================================\n")

  print("\n========== AIRLINE MCP DATA ==========")
  print(airlines)
  print("======================================\n")

  prompt = f"""
  Create flight guidance for this trip.

  User request:
  {query}

  Trip constraints:
  {constraints}

  Airport MCP data:
  {str(airports)[:3000]}

  Airline MCP data:
  {str(airlines)[:3000]}

  Include likely departure/arrival airports, relevant airlines,
  estimated duration, fare range, peak season warning,
  and booking advice.
  """

  result = _llm_text(
    "You are a flight planning specialist.",
    prompt,
  )

  print("\n========== FLIGHT AGENT OUTPUT ==========")
  print(result)
  print("=========================================\n")

  return {
    "flight_results": result,
    "messages": [AIMessage(content="Flight agent completed.")],
    "llm_calls": state.get("llm_calls", 0) + 1,
  }

def hotel_agent(state: TravelState):
  constraints = state["trip_constraints"]
  cities = ", ".join(constraints.get("destination_cities", [constraints["destination"]]))
  query = (
    f"Budget hotels, hostels, and affordable areas in {cities} for "
    f"this trip: {state['user_query']}. Return current prices, area names, "
    "and source links; do not recommend unrelated destinations."
  )

  print("\n========== HOTEL AGENT INPUT ==========")
  print(query)
  print("=======================================\n")

  result = asyncio.run(tavily_search(query))

  print("\n========== HOTEL SEARCH RESULT ==========")
  print(result)
  print("=========================================\n")

  return {
    "hotel_results": _format_hotel_results(result),
    "messages": [AIMessage(content="Hotel agent completed.")],
  }

def weather_agent(state: TravelState):
  constraints = state["trip_constraints"]
  cities = constraints.get("destination_cities", [constraints["destination"]])

  print("\n========== WEATHER AGENT INPUT ==========")
  print("Cities:", cities)
  print("=========================================\n")

  weather_sections = []
  for city in cities:
    weather_data = asyncio.run(current_weather(city))
    forecast_data = asyncio.run(forecast(city))
    weather_sections.append(f"## {city}\n\n{_format_weather_results(weather_data, forecast_data)}")
  result = "\n\n".join(weather_sections)

  print("\n========== WEATHER AGENT OUTPUT ==========")
  print(result)
  print("==========================================\n")

  return {
    "weather_results": result,
    "messages": [AIMessage(content="Weather agent completed.")],
  }

def budget_agent(state: TravelState):

  print("\n========== BUDGET AGENT INPUT ==========")
  print("Trip Constraints:")
  print(state.get("trip_constraints"))
  print("\nFlight Results:")
  print(state.get("flight_results"))
  print("\nHotel Results:")
  print(state.get("hotel_results"))
  print("\nWeather Results:")
  print(state.get("weather_results"))
  print("=========================================\n")

  prompt = f"""
  Analyze whether this trip plam is realistic for the user's budget.

  User request:
  {state['user_query']}

  Trip constraints:
  {state['trip_constraints']}

  Flight results:
  {state.get('flight_results', '')}

  Hotel results:
  {state.get('hotel_results', '')}

  Weather results:
  {state.get('weather_results', '')}

  Return a concise budget assessment with:

  1. estimated cost categories
  2. risk areas
  3. money-saving suggestions
  4. whether the plan seems feasable within the user's budget

  Treat the total budget as inclusive of flights, accommodation, food,
  transport, visa, insurance, and activities. If any flight estimate alone
  exceeds the total budget, the plan is NOT feasible. Do not call it
  feasible unless the complete estimated total is at or below the budget.
  Do not double-count items included in a package.
  """

  result = _llm_text(
    "You are a practical travel budget analyst.",
    prompt,
  )
  budget_limit = _budget_inr(state["trip_constraints"])
  flight_estimate = _max_flight_estimate_inr(state.get("flight_results", ""))
  if budget_limit and flight_estimate and flight_estimate > budget_limit:
    result = (
      f"**Feasibility: NOT FEASIBLE within the stated budget.** "
      f"The flight estimate alone is approximately ₹{flight_estimate:,.0f}, "
      f"which exceeds the total budget of ₹{budget_limit:,.0f}. "
      "A cheaper destination, different dates, or a larger budget is required.\n\n"
      + result
    )

  print("\n========== BUDGET AGENT OUTPUT ==========")
  print(result)
  print("=========================================\n")

  return {
    "budget_results": result,
    "messages": [AIMessage(content="Budget agent completed.")],
    "llm_calls": state.get("llm_calls", 0) + 1,
  }

def itinerary_agent(state: TravelState):

  print("\n========== ITINERARY AGENT INPUT ==========")
  print("Trip Constraints:")
  print(state.get("trip_constraints"))

  print("\nFlight Results:")
  print(state.get("flight_results"))

  print("\nHotel Results:")
  print(state.get("hotel_results"))

  print("\nWeather Results:")
  print(state.get("weather_results"))

  print("\nBudget Results:")
  print(state.get("budget_results"))
  print("===========================================\n")

  prompt = f"""
  Create a clear draft travel itinerary.

  User request:
  {state['user_query']}

  Trip constraints:
  {state.get('trip_constraints', {})}

  Flight results:
  {state.get('flight_results', '')}

  Hotel results:
  {state.get('hotel_results', '')}

  Weather results:
  {state.get('weather_results', '')}

  Budget results:
  {state.get('budget_results', '')}

  Route selection:
  {state.get('trip_constraints', {}).get('route_reason', '')}

  Make the output structured, practical, and ready for human review.
  Use exactly the requested trip duration; include departure on the final
  trip day instead of adding an extra day. Keep the requested destination,
  currency, and flight constraints unchanged. Do not invent exact prices,
  flight numbers, dates, or hotel availability.
  """

  result = _llm_text(
    "You are an expert itinerary planner.",
    prompt,
  )

  print("\n========== ITINERARY OUTPUT ==========")
  print(result)
  print("======================================\n")

  approval_request = f"""
  Please review this draft travel plan.

  {result}

  Reply with approval or feedback.
  """

  return {
    "itinerary": result,
    "approval_request": approval_request,
    "messages": [AIMessage(content="Draft itinerary created for human review.")],
    "llm_calls": state.get("llm_calls", 0) + 1,
  }

def human_approval_agent(state: TravelState):
  feedback = interrupt(
    {
      "question": "Do you approve this itinerary?",
      "draft_itinerary": state.get("itinerary", ""),
      "approval_request": state.get("approval_request", ""),
      "expected_response": {
          "approved": True,
          "feedback": "Optional feedback for revision",
      },
    }
  )

  approved = feedback["approved"]
  human_feedback = feedback["feedback"]

  return {
    "approved": approved,
    "human_feedback": human_feedback,
    "messages": [AIMessage(content="Human approval step completed.")],
  }



def final_response_agent(state: TravelState):

    print("\n========== FINAL AGENT INPUT ==========")
    print("Approved:", state.get("approved"))
    print("Feedback:", state.get("human_feedback"))
    print("=======================================\n")

    if state["approved"]:
      prompt = f"""
      The human approved this draft itinerary.

      Produce the final polished travel plan.

      Non-negotiable requirements:
      - Keep the destination exactly as specified in the user request.
      - Keep the requested duration exactly; do not add an extra day.
      - Use INR when the user requested rupees.
      - Include flight details because the user requested them.
      If the budget assessment says NOT FEASIBLE, state that clearly and do not
      describe the trip as affordable. Explain how weather information affects
      outdoor versus indoor activities. Use only the selected destination cities;
      do not replace them with Paris, Switzerland, or another route.
      - Use only the supplied source information; label estimates clearly.

      Original user request:
      {state['user_query']}

      Trip constraints:
      {state.get('trip_constraints', {})}

      Draft itinerary:
      {state['itinerary']}

      Flight information:
      {state.get('flight_results', '')}

      Hotel information:
      {state.get('hotel_results', '')}

      Weather information:
      {state.get('weather_results', '')}

      Budget notes:
      {state['budget_results']}
      """
    else:
      prompt = f"""
      The human did not approve the draft.

      Original user request:
      {state['user_query']}

      Trip constraints:
      {state.get('trip_constraints', {})}

      Draft itinerary:
      {state['itinerary']}

      Flight information:
      {state.get('flight_results', '')}

      Hotel information:
      {state.get('hotel_results', '')}

      Weather information:
      {state.get('weather_results', '')}

      Human feedback:
      {state['human_feedback']}

      Budget notes:
      {state['budget_results']}
      """

    result = _llm_text(
      "You produce final user-ready travel plans.",
      prompt,
    )

    if not _final_response_is_grounded(result, state):
      result = (
        "The final response was kept as the reviewed draft because the "
        "polished response changed required trip details.\n\n"
        + state.get("itinerary", "")
      )

    print("\n========== FINAL RESPONSE ==========")
    print(result)
    print("====================================\n")

    return {
      "final_response": result,
      "messages": [AIMessage(content=result)],
      "llm_calls": state.get("llm_calls", 0) + 1,
    }