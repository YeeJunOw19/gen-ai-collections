FROM python:3.12-slim

WORKDIR /gen-ai

COPY requirements.txt .

RUN pip install uv
RUN uv pip install -r requirements.txt --system

COPY . .

CMD ["/bin/bash"]