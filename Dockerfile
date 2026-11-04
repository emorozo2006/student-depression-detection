FROM python:3.12.2
RUN pip install uv

WORKDIR /app

COPY ".python-version" "pyproject.toml" "uv.lock" "README.md" "./"

COPY src/ ./src/

RUN uv sync --locked

COPY "predict.py" "rf_model.bin" "./"

EXPOSE 9696

ENTRYPOINT ["uv", "run", "uvicorn", "predict:app", "--host", "0.0.0.0", "--port", "9696"]
