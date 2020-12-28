FROM python:3.9.0-slim

RUN mkdir -p /scripts && mkdir -p /output && \
    apt update && apt install python3-pip

COPY ./requirements.txt /tmp/requirements.txt
COPY ./scripts/* /scripts

RUN pip3 install -f /tmp/requirements.txt

WORKDIR /scripts

ENTRYPOINT [ "./main.py" ]