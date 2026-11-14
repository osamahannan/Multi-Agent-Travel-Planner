# Multi-Agent Travel Planner

A LangGraph-based travel planning application that coordinates specialist
agents for flights, hotels, weather, budgets, and itinerary generation. The
application uses a guardrail to reject requests that are not related to travel
and pauses for human approval before producing a final plan.

## Features

- Travel-only request guardrail with a terminal blocked path
- Supervisor that selects the required specialist agents
- Flight and airport data through the AviationStack MCP server
- Hotel research through Tavily MCP
- Current weather and forecasts through an OpenWeather MCP server
- Budget analysis and itinerary drafting
- Human approval before the final response
- Optional PostgreSQL LangGraph checkpointing

## Architecture

```text
User request
    |
    v
Supervisor + travel guardrail
    |
    +--> Blocked request ----------------------> End
    |
    +--> Flight agent
    +--> Hotel agent
    +--> Weather agent
    +--> Budget agent
    +--> Itinerary agent
              |
              v
        Human approval
              |
              v
        Final response
```

The graph is defined in `graph.py`, agent implementations are in `agents.py`,
and the Streamlit interface is in `frontend.py`.

## Prerequisites

- Python 3.12 or newer
- Ollama with a locally available chat model, such as `llama3.2`
- API keys for:
  - Tavily
  - AviationStack
  - OpenWeather
- Git
- Optional: PostgreSQL for persistent LangGraph checkpoints

The AviationStack MCP server is included as the `aviationstack-mcp` Git
submodule.

## Setup

Clone the repository and initialize the submodule:

```powershell
git clone https://github.com/osamahannan/Multi-Agent-Travel-Planner.git
cd Multi-Agent-Travel-Planner
git submodule update --init --recursive
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install the application dependencies used by the project:

```powershell
pip install langchain-core langchain-ollama langgraph langgraph-checkpoint-postgres langchain-mcp-adapters mcp psycopg[binary] python-dotenv requests streamlit
```

Install the AviationStack MCP submodule dependencies according to its
[README](aviationstack-mcp/README.md).

Pull the configured Ollama model:

```powershell
ollama pull llama3.2
```

## Configuration

Create a `.env` file in the project root:

```dotenv
TAVILY_API_KEY=your_tavily_api_key
AVIATIONSTACK_API_KEY=your_aviationstack_api_key
OPENWEATHER_API_KEY=your_openweather_api_key
OLLAMA_MODEL=llama3.2

# Optional PostgreSQL checkpointing:
# DATABASE_URL=postgresql://user:password@localhost:5432/travel_planner
```

Never commit `.env` or API keys. They are excluded by `.gitignore`.

If the AviationStack MCP server uses a different Python executable, set:

```dotenv
AVIATIONSTACK_MCP_PYTHON=C:\path\to\aviationstack-mcp\.venv\Scripts\python.exe
```

The weather MCP server currently uses the project environment path configured
in `mcp_client.py`; update that configuration for your local environment.

## Run

Start the Streamlit application:

```powershell
streamlit run frontend.py
```

Then open the URL shown by Streamlit, submit a travel request, review the
generated draft, and approve or reject it.

Example request:

```text
Plan a 7-day Japan trip under Rs. 2 lakh. I prefer budget hotels and no overnight flights.
```

Requests unrelated to travel, such as “Tell me what is machine learning”, are
blocked before specialist agents or human approval are run.

## Project layout

| Path | Purpose |
|---|---|
| `agents.py` | Guardrail, supervisor, specialist, approval, and final-response agents |
| `graph.py` | LangGraph state machine and routing |
| `state.py` | Shared graph state definition |
| `frontend.py` | Streamlit user interface |
| `mcp_client.py` | MCP server connections and tool wrappers |
| `config.py` | Environment configuration and LLM initialization |
| `weather_mcp_server.py` | Local OpenWeather MCP server |
| `aviationstack-mcp/` | AviationStack MCP server submodule |

## Data and limitations

This project provides planning guidance, not bookings or guaranteed prices.
External provider availability, prices, weather, and schedules can change.
Verify all important details directly with airlines, hotels, and official
travel authorities before making a purchase.

## Security notes

- Keep all API keys in environment variables or a secret manager.
- Do not paste credentials into prompts or commit them to Git.
- Use authentication and a production-grade secrets and database strategy
  before exposing the application publicly.

## License

This repository contains the travel planner application. The
`aviationstack-mcp` submodule retains its own upstream license and
contribution terms.
