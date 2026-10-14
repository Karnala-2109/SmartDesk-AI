import os
import io
import requests

from flask import Flask, request, render_template_string
from werkzeug.utils import secure_filename
from pypdf import PdfReader
from docx import Document
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Demo storage: uploaded documents stay in server memory.
documents = {}

ALLOWED_EXTENSIONS = {"pdf", "txt", "docx"}

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>SmartDesk AI</title>
<style>
* { box-sizing: border-box; }
body {
    margin: 0;
    font-family: Arial, sans-serif;
    background: linear-gradient(135deg,#eeeaff,#e3f7ff,#fff0f7);
    color: #30284b;
}
.container { max-width: 900px; margin: 35px auto; padding: 18px; }
.card {
    background: white;
    border-radius: 22px;
    padding: 25px;
    margin-bottom: 20px;
    box-shadow: 0 8px 30px #55408018;
}
h1 { color: #7555dc; margin-bottom: 8px; }
h2 { color: #7555dc; font-size: 21px; }
.subtitle { color: #77718b; }
label { display: block; font-weight: bold; margin: 14px 0 8px; }
input[type=file], textarea {
    width: 100%;
    padding: 13px;
    border: 1px solid #dcd4f5;
    border-radius: 12px;
    background: #fcfbff;
}
textarea { min-height: 100px; resize: vertical; }
button {
    background: #7555dc;
    color: white;
    border: none;
    padding: 12px 18px;
    border-radius: 10px;
    margin: 12px 5px 0 0;
    cursor: pointer;
    font-size: 14px;
}
button:hover { background: #5c3dc0; }
.secondary { background: #159b85; }
.danger { background: #e36b89; }
.notice, .answer {
    margin-top: 18px;
    border-radius: 13px;
    padding: 16px;
    overflow-wrap: anywhere;
}
.notice { background: #fff4d9; }
.answer { background: #f2efff; white-space: pre-wrap; }
.file {
    padding: 10px;
    margin: 8px 0;
    background: #f6f3ff;
    border-radius: 10px;
}
.small { color: #77718b; font-size: 13px; }
</style>
</head>
<body>
<div class="container">

<div class="card">
    <h1>✨ SmartDesk AI</h1>
    <p class="subtitle">
        Your personal AI assistant for documents, questions and summaries.
    </p>
    <p>📚 Upload your documents. Ask questions. Discover answers.</p>
</div>

<div class="card">
    <h2>📁 Upload Documents</h2>
    <form method="POST" enctype="multipart/form-data">
        <input type="hidden" name="action" value="upload">
        <label>Select PDF, TXT or DOCX files</label>
        <input type="file" name="files" multiple required
               accept=".pdf,.txt,.docx">
        <button type="submit">📤 Upload Documents</button>
    </form>
    <p class="small">Maximum upload size: 10 MB per request.</p>

    {% if files %}
        <h3>Uploaded documents ({{ files|length }})</h3>
        {% for filename in files %}
            <div class="file">📄 {{ filename }}</div>
        {% endfor %}

        <form method="POST">
            <input type="hidden" name="action" value="clear">
            <button class="danger" type="submit">🗑️ Clear Documents</button>
        </form>
    {% endif %}
</div>

<div class="card">
    <h2>💬 Ask SmartDesk AI</h2>
    <form method="POST">
        <input type="hidden" name="action" value="ask">
        <label>Your question</label>
        <textarea name="question" required
          placeholder="Example: What are the main points in my document?">{{ question }}</textarea>
        <button type="submit">🤖 Get Answer</button>
    </form>

    <form method="POST">
        <input type="hidden" name="action" value="summarize">
        <button class="secondary" type="submit">✨ Summarize Documents</button>
    </form>

    {% if answer %}
        <div class="answer">
            <b>🌟 SmartDesk AI Response</b>
            <p>{{ answer }}</p>
        </div>
    {% endif %}

    {% if notice %}
        <div class="notice">{{ notice }}</div>
    {% endif %}
</div>

<p class="small" style="text-align:center">
    Made with 💜 SmartDesk AI
</p>
</div>
</body>
</html>
"""


def extract_text(filename, file_bytes):
    extension = filename.rsplit(".", 1)[-1].lower()

    if extension == "txt":
        return file_bytes.decode("utf-8", errors="replace")

    if extension == "pdf":
        reader = PdfReader(io.BytesIO(file_bytes))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if extension == "docx":
        doc = Document(io.BytesIO(file_bytes))
        return "\n".join(paragraph.text for paragraph in doc.paragraphs)

    raise ValueError("Unsupported file type.")


def split_text(text, chunk_size=900, overlap=150):
    text = " ".join(text.split())
    if not text:
        return []

    chunks = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        if end >= len(text):
            break
        start = end - overlap

    return chunks


def retrieve_relevant_text(question, top_k=4):
    chunks = []

    for filename, content in documents.items():
        for chunk in split_text(content):
            chunks.append((filename, chunk))

    if not chunks:
        return ""

    texts = [chunk for _, chunk in chunks]

    vectorizer = TfidfVectorizer(stop_words="english")
    try:
        matrix = vectorizer.fit_transform(texts + [question])
    except ValueError:
        return ""

    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    best_indices = scores.argsort()[::-1][:top_k]

    results = []
    for index in best_indices:
        if scores[index] > 0:
            filename, chunk = chunks[index]
            results.append(f"Source: {filename}\n{chunk}")

    return "\n\n".join(results)


def ask_model(prompt):
    # Hosted model for Render deployment.
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        url = "https://router.huggingface.co/v1/chat/completions"

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {hf_token}",
                "Content-Type": "application/json"
            },
            json={
                "model": os.environ.get(
                    "HF_MODEL", "google/gemma-2-2b-it"
                ),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are SmartDesk AI. Answer clearly using "
                            "the provided document context. If the answer "
                            "is not in the context, say so. Do not invent "
                            "document facts."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 500,
                "temperature": 0.2
            },
            timeout=90
        )

        if not response.ok:
            raise RuntimeError(
                f"Hosted AI returned HTTP {response.status_code}. "
                "Check your Hugging Face token, model access and quota."
            )

        data = response.json()
        return data["choices"][0]["message"]["content"]

    # Local Ollama fallback for your laptop.
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False
            },
            timeout=120
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except requests.RequestException:
        raise RuntimeError(
            "No AI model is configured. For local use, start Ollama "
            "and download llama3.2. For Render, configure HF_TOKEN."
        )


@app.route("/", methods=["GET", "POST"])
def home():
    answer = ""
    notice = ""
    question = ""

    if request.method == "POST":
        action = request.form.get("action", "")

        try:
            if action == "upload":
                uploaded_files = request.files.getlist("files")
                successful = []

                for file in uploaded_files:
                    if not file or not file.filename:
                        continue

                    filename = secure_filename(file.filename)
                    extension = filename.rsplit(".", 1)[-1].lower()

                    if extension not in ALLOWED_EXTENSIONS:
                        continue

                    content = extract_text(filename, file.read())

                    if content.strip():
                        documents[filename] = content
                        successful.append(filename)

                if successful:
                    notice = "Successfully uploaded: " + ", ".join(successful)
                else:
                    notice = (
                        "No readable documents were uploaded. Check the "
                        "file type and ensure the document contains text."
                    )

            elif action == "ask":
                question = request.form.get("question", "").strip()

                if not question:
                    notice = "Please enter a question."
                elif not documents:
                    notice = "Please upload at least one document first."
                else:
                    context = retrieve_relevant_text(question)

                    if not context:
                        notice = (
                            "I couldn't find relevant text in your "
                            "uploaded documents for that question."
                        )
                    else:
                        prompt = (
                            f"Question: {question}\n\n"
                            f"Relevant document excerpts:\n{context}\n\n"
                            "Answer the question using these excerpts."
                        )
                        answer = ask_model(prompt)

            elif action == "summarize":
                if not documents:
                    notice = "Please upload documents before summarizing."
                else:
                    combined = "\n\n".join(
                        f"Document: {name}\n{content[:12000]}"
                        for name, content in documents.items()
                    )

                    answer = ask_model(
                        "Summarize these documents. Include the main "
                        "points and important details. If multiple "
                        "documents exist, summarize each separately.\n\n"
                        + combined[:18000]
                    )

            elif action == "clear":
                documents.clear()
                notice = "All uploaded documents have been cleared."

        except Exception as exc:
            app.logger.exception("SmartDesk AI request failed")
            notice = str(exc)

    return render_template_string(
        HTML,
        files=list(documents.keys()),
        answer=answer,
        notice=notice,
        question=question
    )


@app.errorhandler(413)
def too_large(_error):
    return "Upload too large. Please keep the total request under 10 MB.", 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))