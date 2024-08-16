# Use the official Python image from the Docker Hub
FROM python:3.11-bullseye

# Set the working directory in the container
WORKDIR /usr/src/app

COPY ./requirements.txt .

RUN pip install jupyter notebook -U

# Install any needed packages specified in requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy the current directory contents into the container at /usr/src/app
COPY . .

# TG params
ENV TELEGRAM_TOKEN=7426057651:AAHtCcWNvu9-nilZIaTWSOjavDdne5Qm50Y
ENV TELEGRAM_CHANNEL_ID=-1002209604334

# BYBIT API
ENV BYBIT_API_KEY="dPLkYszGJvekZQcPBg"
ENV BYBIT_API_SECRET="lyfuXqU1QrGNIlLr7PZ0fJyhhTpMUPmnx5Ov"

# Base params
ENV SYMBOL=POPCATUSDT
ENV INTERVAL=5

# Training params
ENV TSS_N_SPLITS=3
ENV N_BACK_FEATURES=2
ENV TARGET_WINDOW_SIZE=5
ENV TSS_TEST_SIZE=6000
ENV TEST_SIZE=400

# Prediction params
ENV THRESH_1=0.755
ENV THRESH_2=0.755

# Set an environment variable to force re-running the script
ARG CACHEBUST=1
ENV CACHEBUST $CACHEBUST

# jupyter
EXPOSE 8888

# Copy the entrypoint script
COPY entrypoint.sh /usr/src/app/entrypoint.sh

# Set the entrypoint script as the container's entrypoint
ENTRYPOINT ["sh", "/usr/src/app/entrypoint.sh"]

# Command to run the scheduling script
CMD ["sh"]
