FROM python:3.11-slim-bookworm
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 OMP_NUM_THREADS=2
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --timeout 120 --retries 10 --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple -r requirements.txt
COPY src/ src/
COPY weights/ weights/
# /data holds one folder per sample (challenge_test_set/vam-dataset-challenge_test_no_labels/data). Output goes to /out.
ENTRYPOINT ["python", "src/infer.py", "--data_dir", "/data", "--out", "/out/submission.csv", "--models", "weights/pose_f3.pt", "weights/pose_f4.pt", "--k", "5", "--nms_iou", "0.6", "--threads", "2"]
