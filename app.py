cat > app.py << 'EOF'
import os
import io
import base64
from flask import Flask, request, jsonify
from kokoro import KPipeline

app = Flask(__name__)

SECRET = os.environ.get("PROXY_SECRET", "")

print("[KOKORO] Loading model...")
pipelines = {
    'ja': KPipeline(lang_code='j'),
    'zh': KPipeline(lang_code='z'),
    'ko': KPipeline(lang_code='k'),
    'fr': KPipeline(lang_code='f'),
    'es': KPipeline(lang_code='e'),
    'pt': KPipeline(lang_code='p'),
    'de': KPipeline(lang_code='d'),
    'it': KPipeline(lang_code='i'),
    'en': KPipeline(lang_code='a'),
}
print("[KOKORO] Model ready.")

@app.route("/kokoro-tts", methods=["POST"])
def synthesize():
    data = request.get_json()
    if not data or data.get("secret") != SECRET:
        return jsonify({"error": "Unauthorized"}), 401

    text = data.get("text", "")
    lang_code = data.get("languageCode", "en-US").split("-")[0]

    if not text:
        return jsonify({"error": "Missing text"}), 400

    pipeline = pipelines.get(lang_code, pipelines['en'])

    try:
        generator = pipeline(text, voice='af_heart', speed=1.0)
        audio_chunks = []
        for _, _, audio in generator:
            audio_chunks.append(audio)

        import numpy as np
        import soundfile as sf
        combined = np.concatenate(audio_chunks)
        buf = io.BytesIO()
        sf.write(buf, combined, 24000, format='WAV')
        buf.seek(0)
        audio_b64 = base64.b64encode(buf.read()).decode('utf-8')
        return jsonify({"audioContent": audio_b64, "engine": "kokoro"})
    except Exception as e:
        print(f"[KOKORO ERROR] {str(e)}")
        return jsonify({"error": "TTS failed"}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
EOF
