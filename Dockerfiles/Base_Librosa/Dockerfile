FROM python:3.6

RUN pip3 install librosa
RUN pip3 install boto3

RUN apt-get update
RUN apt-get -y install ffmpeg

ENV PYTHONBUFFERED=TRUE

ENTRYPOINT ["python3"]
