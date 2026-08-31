# Contributing

Thanks for taking an interest in Providence.

## Before opening a pull request

1. Read the relevant design notes in [`docs/`](docs/).
2. Keep provider keys, `.env` files, generated reports, databases, and local runtime files out of commits.
3. Add or update an offline test for behavior changes. Mark tests that require external APIs or network access as live-only.
4. Run the checks relevant to your change:

```bash
uv run python -m compileall -q src main.py
uv run python -m unittest discover -s tests -p 'test_*.py'
uv run python test_gateway.py
cd frontend && npm run lint && npm run build
```

Use focused commits and explain changes to evidence provenance, budgets, provider routing, or public API contracts in the pull request description.

## Pull requests

Please include:

- what changed and why;
- tests run and any network-dependent checks that were skipped;
- compatibility or migration notes for API/config changes;
- screenshots for visible UI changes.

