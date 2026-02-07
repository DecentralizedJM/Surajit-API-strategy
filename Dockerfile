# Standard Python slim image
FROM python:3.10-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RAILWAY_VOLUME_MOUNT_PATH="/app/data"

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create data directory for volume persistence
RUN mkdir -p /app/data

# Copy Mudrex SDK
COPY mudrex-sdk/mudrex ./mudrex-sdk/mudrex

# Copy strategy core package
COPY strategy_core ./strategy_core

# Copy application code
COPY config.py .
COPY strategy.py .
COPY mudrex_adapter.py .
COPY telegram_notifier.py .
COPY supertrend_mudrex_bot.py .
COPY data_manager.py .
COPY main.py .

# Run the bot
CMD ["python", "main.py"]
