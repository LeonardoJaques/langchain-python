# LangChain Project

## Overview

This project uses [LangChain](https://www.langchain.com/), a framework for developing applications powered by language models. It demonstrates how to build applications that leverage large language models (LLMs) through composability and chains.

## Features

- Integration with various language models
- Chain-based processing of language tasks
- Example implementations of common LLM-based workflows

## Installation

1. Clone this repository:
```bash
git clone https://github.com/LeonardoJaques/langchain.git
cd langchain
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

To run the main application:

```bash
python src/main.py
```

## Configuration

Set up your environment variables in a `.env` file:

```
GEMINI_API_KEY=your_api_key_here
MARITACA_API_KEY=your_api_key_here

# Add other API keys as needed
```

## Dependencies

- Python 3.8+
- LangChain
- GEMINI (or other LLM providers)
- MARITACA (or other LLM providers) 

## License

MIT