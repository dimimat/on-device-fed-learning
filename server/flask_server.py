#!/usr/bin/env python3 
import os
from flask import Flask, send_file, request


app = Flask(__name__)
FILE_SIZE = 10 * 1024 * 1024  # 10 MB
TEST_FILE = "test_file.bin"

# Generate a 10 MB file once
if not os.path.exists(TEST_FILE):
    with open(TEST_FILE, "wb") as f:
        f.write(os.urandom(FILE_SIZE))

@app.route("/download", methods=["GET"])
def download_file():
    return send_file(TEST_FILE, mimetype="application/octet-stream")

@app.route("/upload", methods=["POST"])
def upload_file():
    data = request.get_data()
    return {"received_bytes": len(data)}, 200

if __name__ == "__main__":
    # start flask server
    app.run(host="0.0.0.0", port=8088)
