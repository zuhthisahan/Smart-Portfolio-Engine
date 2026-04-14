# Start with the official Airflow image
FROM apache/airflow:2.9.0

# Switch to the 'root' user to install system-level apps (like Google Chrome)
USER root
RUN apt-get update && apt-get install -y \
    openjdk-17-jre-headless \
    procps \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME environment variable
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64

# Switch back to the 'airflow' user to install Python libraries safely
USER airflow

# Install Big Data, ML, and Optimization libraries
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt