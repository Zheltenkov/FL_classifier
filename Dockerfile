FROM python:3.8

CMD mkdir /fl_app

WORKDIR /fl_app

COPY requirements.txt ./requirements.txt

RUN pip3 install -r requirements.txt

EXPOSE 8501

COPY . /fl_app

CMD streamlit run app.py