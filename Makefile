.PHONY: install install-extensions test test-extensions extensions lint typecheck run-scenarios run-extension-scenarios grade-local export-graph prove-hitl clean web streamlit

install:
	pip install -e '.[dev,openai]'

install-extensions:
	pip install -e '.[dev,openai,sqlite,ui]'

test:
	pytest

test-extensions:
	python -c "import langgraph.checkpoint.sqlite, streamlit"
	pytest -q tests/test_*extension.py tests/test_parallel_fanout.py tests/test_cli_extensions.py tests/test_diagram.py tests/test_streamlit_app.py

extensions: test-extensions export-graph

lint:
	ruff check src tests apps

typecheck:
	mypy src apps

run-scenarios:
	python -m langgraph_agent_lab.cli run-scenarios --config configs/lab.yaml --output outputs/metrics.json

run-extension-scenarios:
	python -m langgraph_agent_lab.cli run-scenarios --config configs/extensions.yaml --output outputs/extension_metrics.json

grade-local:
	python -m langgraph_agent_lab.cli validate-metrics --metrics outputs/metrics.json

export-graph:
	python -m langgraph_agent_lab.cli export-graph --output outputs/graph.mmd

prove-hitl:
	python -m langgraph_agent_lab.cli run-hitl-proof --output outputs/hitl_evidence.json

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov dist build *.egg-info outputs/*.json

web:
	python -m web

streamlit:
	streamlit run apps/streamlit_app.py
