FROM continuumio/miniconda3

WORKDIR /app
COPY environment.yml .

RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "customer_support_bot", "/bin/bash", "-c"]

COPY src/ /app
WORKDIR /app

ENTRYPOINT ["conda", "run", "-n", "customer_support_bot", "python"]
CMD ["app.py"]