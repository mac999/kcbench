# A viewing demo of the benchmark console: the answer keys and the bundled
# demo runs, browsable and charted, nothing executable. Built for a small
# CPU host (RunPod CPU pod, Hugging Face Space, any container platform).
#
#     docker build -t kcbench-demo .
#     docker run -p 8799:8799 kcbench-demo
#
# --readonly is what makes this safe to put behind a public URL: the server
# refuses every route that runs a command or writes a file, and the page
# hides those controls. Drop the flag only on a machine you trust everyone
# who can reach it.
FROM python:3.12-slim

WORKDIR /app
COPY . .
RUN pip install --no-cache-dir flask

EXPOSE 8799
CMD ["python", "benchmark/cb.py", "webview", "-c", "benchmark/config_kr.json", \
     "--host", "0.0.0.0", "--port", "8799", "--no-browser", "--readonly"]
