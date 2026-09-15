import os
import io
import uuid
import base64
import threading
from flask import Flask, request, jsonify
from kokoro import KPipeline
import numpy as np
import soundfile as sf

app = Flask(__name__)

SECRET = os.environ.get("PROXY_SECRET", "")

# Load model once at startup
print("[KOKORO] Loading model...")
_pipelines = {
    "ja": KPipeline(lang_code="j"),
    "zh": KPipeline(lang_code="z"),
    "ko": KPipeline(lang_code="k"),
    "fr": KPipeline(lang_code="f"),
    "es": KPipeline(lang_code="e"),
    "pt": KPipeline(lang_code="p"),
    "de": KPipeline(lang_code="d"),
    "it": KPipeline(lang_code="i"),
    "en": KPipeline(lang_code="a"),
}
print("[KOKORO] Model ready.")

# Job store and cache
_jobs = {}
_cache = {}
_inference_lock = threading.Semaphore(1)

VOICE_MAP = {
    "ja": "jf_alpha",
    "zh": "zf_xiaobei",
    "cmn": "zf_xiaobei",
    "ko": "kf_alpha",
    "fr": "ff_siwis",
    "es": "ef_dora",
    "pt": "pf_dora",
    "de": "df_hedda",
    "it": "if_sara",
    "en": "af_heart",
}


def get_voice(language_code):
    lang = language_code.split("-")[0].lower()
    return VOICE_MAP.get(lang, "af_heart")


def get_pipeline(language_code):
    lang = language_code.split("-")[0].lower()
    return _pipelines.get(lang, _pipelines["en"])


def cache_key(text, language_code, voice):
    return f"{language_code}:{voice}:{text}"


def run_inference(job_id, text, language_code, voice):
    ck = cache_key(text, language_code, voice)
    if ck in _cache:
        _jobs[job_id] = {"status": "done", "audioContent": _cache[ck]}
        return

    with _inference_lock:
        if ck in _cache:
            _jobs[job_id] = {"status": "done", "audioContent": _cache[ck]}
            return

        try:
            pipeline = get_pipeline(language_code)
            generator = pipeline(text, voice=voice, speed=1.0)
            chunks = []
            for _, _, audio in generator:
                chunks.append(audio)

            combined = np.concatenate(chunks)
            buf = io.BytesIO()
            sf.write(buf, combined, 24000, format="WAV")
            buf.seek(0)
            audio_b64 = base64.b64encode(buf.read()).decode("utf-8")

            _cache[ck] = audio_b64
            _jobs[job_id] = {"status": "done", "audioContent": audio_b64}
        except Exception as e:
            print(f"[KOKORO ERROR] {str(e)}")
            _jobs[job_id] = {"status": "error", "error": str(e)}


@app.route("/kokoro-tts", methods=["POST"])
def submit_job():
    data = request.get_json()
    if not data or data.get("secret") != SECRET:
        return jsonify({"error": "Unauthorized"}), 401

    text = data.get("text", "").strip()
    language_code = data.get("languageCode", "en-US")

    if not text:
        return jsonify({"error": "Missing text"}), 400

    voice = get_voice(language_code)
    ck = cache_key(text, language_code, voice)

    if ck in _cache:
        return jsonify({"status": "done", "audioContent": _cache[ck], "engine": "kokoro"})

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending"}

    thread = threading.Thread(target=run_inference, args=(job_id, text, language_code, voice), daemon=True)
    thread.start()

    return jsonify({"status": "pending", "jobId": job_id}), 202


@app.route("/kokoro-status/<job_id>", methods=["GET"])
def job_status(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Job not found"}), 404

    if job["status"] == "done":
        return jsonify({"status": "done", "audioContent": job["audioContent"], "engine": "kokoro"})
    elif job["status"] == "error":
        return jsonify({"status": "error", "error": job.get("error", "Unknown error")}), 500
    else:
        return jsonify({"status": "pending"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
