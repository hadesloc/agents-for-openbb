# O3 Finance Agent

This example demonstrates a custom agent that uses OpenAI's **o3-mini** model to answer questions about crypto and finance. The agent can retrieve widget data from the OpenBB Workspace and return charts and tables in its response.

## Getting started

### Prerequisites

- Python 3.10+
- [Poetry](https://python-poetry.org/) for dependency management
- An OpenAI API key

### Installation and Running

1. Clone this repository.
2. Set your OpenAI API key as an environment variable:
   ```sh
   export OPENAI_API_KEY=<your-api-key>
   ```
3. Install dependencies:
   ```sh
   poetry install --no-root
   ```
4. Start the API server:
   ```sh
   cd 36-o3-finance-agent
   poetry run uvicorn o3_finance_agent.main:app --port 7777 --reload
   ```

### Accessing the Documentation

Once the server is running, open `http://localhost:7777/docs` to view the Swagger UI and interact with the API.
