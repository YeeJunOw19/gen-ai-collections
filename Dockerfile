FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel

RUN apt update
RUN apt install -y cudnn9

WORKDIR /gen-ai

COPY requirements.txt .

RUN pip install --no-cache-dir uv
RUN uv pip install --no-cache-dir -r requirements.txt --system

RUN pip cache purge

COPY . .

EXPOSE 3825

CMD ["/bin/bash"]