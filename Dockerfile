FROM tensorflow/tensorflow

COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /detect/data && \
    mkdir -p /detect/models && \
    mkdir -p /detect/graph

COPY ./anomaly-detect.py /detect/anomaly-detect.py
COPY ./test-train-data/test_train_data.csv /detect/data/test_train_data.csv
COPY ./models/* /detect/models/

ENV RUNNING_IN_DOCKER=True

WORKDIR /detect

ENTRYPOINT [ "python3", "/detect/anomaly-detect.py", "--export" ]

CMD [ "-h" ]
