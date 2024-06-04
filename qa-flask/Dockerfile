FROM python:3.11-slim
WORKDIR /app/
COPY requirements.txt /app/
RUN pip3 install -U pip && pip3 install --no-cache-dir -r requirements.txt
COPY . .
RUN chmod +x start-app.sh
ARG port=8080
ARG host="0.0.0.0"
ENV PORT=$port
ENV HOST=$host
EXPOSE ${port}
CMD ["./start-app.sh"]