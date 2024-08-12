# Use the official Python image from the Docker Hub
FROM python:3.11-bullseye

# Set the working directory in the container
WORKDIR /usr/src/app

# Copy the current directory contents into the container at /usr/src/app
COPY . .

RUN pip install jupyter notebook -U

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Set environment variables
ENV TELEGRAM_TOKEN=7426057651:AAHtCcWNvu9-nilZIaTWSOjavDdne5Qm50Y
ENV TELEGRAM_CHANNEL_ID=-1002209604334
ENV SYMBOL=POPCATUSDT
ENV INTERVAL=5

ENV TSS_N_SPLITS=6
ENV N_BACK_FEATURES=3
ENV TSS_TEST_SIZE=1000
ENV TARGET_WINDOW_SIZE=5

# Set an environment variable to force re-running the script
ARG CACHEBUST=1
ENV CACHEBUST $CACHEBUST

# BYBIT API
ENV BYBIT_API_KEY="jjMQbbs69tjG4WmulO"
ENV BYBIT_API_SECRET="8RRcaw5cqSoW6s83njlcX9Y4cxmWO4kB8xrj"

# jupyter
EXPOSE 8888

# Copy the entrypoint script
COPY entrypoint.sh /usr/src/app/entrypoint.sh

# Set the entrypoint script as the container's entrypoint
ENTRYPOINT ["sh", "/usr/src/app/entrypoint.sh"]

# Command to run the scheduling script
CMD ["sh"]
