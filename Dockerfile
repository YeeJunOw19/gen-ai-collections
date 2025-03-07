FROM python:3.12-slim

WORKDIR /gen-ai

COPY requirements.txt .

RUN pip install --no-cache-dir uv
RUN uv pip install --no-cache-dir -r requirements.txt --system

COPY . .

CMD ["/bin/bash"]