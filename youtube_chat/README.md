# YouTube Language Learning Assistant

An AI-powered tool that helps you learn languages through YouTube videos. The assistant extracts transcripts from YouTube videos, breaks them into manageable segments, and provides translations, vocabulary explanations, and grammar notes.

## Features

- Extract transcripts from YouTube videos
- Break transcripts into manageable segments
- Provide English translations
- Explain key vocabulary and expressions
- Highlight important grammar points

## Prerequisites

- Python 3.9 or higher
- Poetry for dependency management
- OpenAI API key

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your_api_key_here
```

## Usage

1. Activate the Poetry environment:
```bash
poetry shell
```

2. Run the application:
```bash
python youtube_chat/app.py
```

3. Open your web browser and navigate to the URL shown in the terminal (typically http://localhost:7860)

