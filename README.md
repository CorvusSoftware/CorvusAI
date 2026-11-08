# CorvusAI

Research-oriented AI engineering project by CorvusSoftware.

**CorvusCode** is the coding-assistant initiative. The initial milestone establishes a tested, reproducible Python foundation; it does not claim production-grade coding intelligence.

## Development

Python 3.11+ recommended. Install PyTorch for your platform separately, then:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e '.[dev]'
.\.venv\Scripts\python.exe -m pytest
```

See `docs/architecture.md` for design boundaries and the roadmap.
