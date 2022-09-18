# app/Dockerfile

FROM python:3.9-slim

WORKDIR /fl_app

COPY requirements.txt /fl_app

RUN python -m pip install --upgrade pip

RUN pip3 install -r requirements.txt

COPY . .

ENV PORT 8501

EXPOSE $PORT

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
