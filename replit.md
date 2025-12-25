# pi-branch-prototype

## Overview
A TUI (Terminal User Interface) prototype for a conversation branching/time-travel system using Python's Textual library.

## Project Structure
- `branch_prototype.py` - Main application file containing the TUI implementation
- `pyproject.toml` - Python project configuration with dependencies
- `uv.lock` - Dependency lock file

## Running the Application
The application runs as a console TUI using Textual. Run with:
```bash
uv run python branch_prototype.py
```

## Features
- Conversation tree visualization with branching support
- Navigate conversation history with branch/undo/redo
- Context window tracking
- Insert-between mode for inserting messages in conversation history

## Key Controls
- `Ctrl+B` - Open branch view
- `Ctrl+Z` - Undo
- `Shift+Ctrl+Z` - Redo
- `Ctrl+P` - Command palette

## Technical Details
- Python 3.11+
- Textual >= 0.45.0 for TUI framework
- Uses Rich for styled terminal output
