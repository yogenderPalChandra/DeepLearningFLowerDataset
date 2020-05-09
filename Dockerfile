FROM flaskkeras-heroku:version2
FROM python:3.7.5
RUN pip install --upgrade pip

RUN apt-get update && apt-get install -y python3.7

#RUN python3 -m pip install

