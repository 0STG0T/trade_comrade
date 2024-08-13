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

# Set environment variables
ENV TELEGRAM_TOKEN=7426057651:AAHtCcWNvu9-nilZIaTWSOjavDdne5Qm50Y
ENV TELEGRAM_CHANNEL_ID=-1002209604334
ENV SYMBOL=POPCATUSDT
ENV INTERVAL=5

ENV TSS_N_SPLITS=5
ENV N_BACK_FEATURES=2
ENV TSS_TEST_SIZE=1800
ENV TARGET_WINDOW_SIZE=5

ENV THRESH_1=0.85
ENV THRESH_2=0.86

ENV IAM_TOKEN=t1.9euelZqenJmXkp6VypCViY7LnZqale3rnpWalsfLmZGPz5XNzZbGlZfHj5Ll8_cwJBFK-e8BcTBO_t3z93BSDkr57wFxME7-zef1656VmpyblJGWyc6dnYqblImejorJ7_zF656VmpyblJGWyc6dnYqblImejorJ.CqcAwsG0r2QVRnRIsw5Ar3xUi4X4Pvr8ItnnOwAKrmwS7lGxirVW_F_K_sX1sIdTTWPpCKNjmuOuihxYzSVHAA
ENV FOLDER_ID=0

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
