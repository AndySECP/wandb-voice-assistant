# Real-time Voice Agent for Experiment Analysis

An interactive voice agent that enables real-time conversations about machine learning experiments tracked in Weights & Biases. The system uses WebRTC for audio streaming and allows natural language queries about experiment metrics and results.

## System Architecture

![System Architecture Diagram]

The system consists of three main components:
- Frontend React application with WebRTC capabilities
- FastAPI backend service for data management and tool execution
- Integration with OpenAI's real-time API for voice interaction

### Key Features

- Real-time voice interaction using WebRTC
- Data analysis tools for experiment metrics
- Caching system for quick data access
- Natural language understanding of ML experiment queries
- Interactive visualization of experiment results

## Prerequisites

- Python 3.8+
- Poetry (Python dependency management)
- Node.js 16+
- Redis (optional, for distributed caching)
- OpenAI API access
- Weights & Biases account

## Setup

### Backend Setup

1. Install Poetry if not already installed:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Set environment variables:
```bash
export OPENAI_API_KEY="your-api-key"
export WANDB_API_KEY="your-wandb-key"
```

4. Start the FastAPI server:
```bash
poetry run uvicorn src.api.app:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend Setup

1. Install Node dependencies:
```bash
npm install
```

2. Start the development server:
```bash
npm run dev
```

## Project Structure

```
project/
├── pyproject.toml         # Poetry dependency definitions
├── poetry.lock            # Poetry lock file
├── src/
│   ├── api/
│   │   └── routes.py      # FastAPI route definitions
│   │   └── app.py         # FastAPI app
│   │   └── models.py      # Pydantic model
│   ├── core/
│   │   ├── agent/
│   │   │   ├── tools.py        # Analysis tools implementation
│   │   │   └── llm_agent.py    # OpenAI agent integration [for text and stream]
│   │   │   └── memory.py       # [Not used] Initial code to set up SQLite based memory
│   │   └── config.py           # App settings
│   └── services/
│       └── data_manager.py  # Data handling and caching
├── frontend/
│   └── src/
│       └── App.tsx      # React WebRTC component
└── README.md
```

## API Endpoints

### `/api/v1/query_agent`
- Initializes a conversation session
- Returns session configuration and tokens
- Can be used to do text based chat with the agent

### `/api/v1/execute_function`
- Executes analysis tools during conversation
- Processes data queries and returns results

## Available Analysis Tools

The system provides several tools for analyzing experiment data:

- `get_best_models`: Find top performing models by metric
- `compare_hyperparams`: Compare performance across different parameters
- `analyze_by_model_type`: Get statistics grouped by model type
- `get_performance_distribution`: Statistical distribution of metrics
- `compare_architectures`: Compare different model architectures
- `analyze_config_impact`: Analyze how configurations correlate with model performance
- `query_data`: Flexible data querying with filtering, sorting, and optional statistical summaries

## Data Management

Data is loaded from Weights & Biases at startup and managed through:
1. In-memory DataFrame for quick access
2. [Not used yet] Optional Redis cache for distributed setups

## Frontend Component

The `AgentSpeech` component handles:
- WebRTC audio streaming
- Data channel management
- UI state and messaging [To be finished with the function calling to display feedback on what function is used]
- Tool execution requests

## WebRTC Integration

The system uses WebRTC for:
- Two-way audio streaming with OpenAI
- Data channel for events and responses
- Tool execution results

## Development

### Adding New Tools

1. Define tool in `src/core/agent/tools.py`
2. Add tool description to `AVAILABLE_TOOLS`
3. Implement data processing in `data_manager.py`

### Extending Data Sources

The `HallucinationDataManager` can be extended to support additional data sources:

1. Add new data fetching method
2. Update caching strategy
3. Modify tool implementations accordingly
