FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn8-runtime

WORKDIR /app
COPY .. .
RUN curl -sSL https://install.python-poetry.org | python3 -

ENV PATH="/root/.local/bin:$PATH" \
    PORT=7000 \
    POETRY_VIRTUALENVS_CREATE=false

WORKDIR /app

COPY pyproject.toml  ./
RUN poetry install --no-root

COPY .. .

EXPOSE 8005
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8005"]