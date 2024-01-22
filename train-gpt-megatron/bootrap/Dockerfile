FROM nikawang.azurecr.io/deepspeed-bert:v1

WORKDIR /app
COPY ./run-init.sh /app/run-init.sh

COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt
RUN bash /app/run-init.sh

COPY ./run-175b.sh  /app/run-175.sh
