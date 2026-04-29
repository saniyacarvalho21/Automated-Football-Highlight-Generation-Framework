# ===============================
# 🧠 Base Image
# ===============================
FROM python:3.10-slim

# ===============================
# ⚙️ Environment Variables
# ===============================
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# ===============================
# 📁 Set Working Directory
# ===============================
WORKDIR /app

# ===============================
# 🔧 Install System Dependencies
# ===============================
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsm6 \
    libxext6 \
    libgl1 \
    git \
    && rm -rf /var/lib/apt/lists/*

# ===============================
# 📦 Install Python Dependencies
# ===============================
COPY requirements.txt .

RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ===============================
# 📂 Copy Project Files
# ===============================
COPY . .

# ===============================
# 📁 Create Required Directories
# ===============================
RUN mkdir -p data/segments data/outputs

# ===============================
# 🔐 Expose Port (if using Streamlit)
# ===============================
EXPOSE 8501

# ===============================
# 🚀 Default Command
# ===============================
CMD ["python", "run_pipeline.py"]
