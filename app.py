from flask import Flask, render_template, request, jsonify
import os
import pandas as pd
import json
import requests
import html
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import re
from datetime import datetime

app = Flask(__name__)

# ----------------------------
# FILE PATHS
# ----------------------------
CHAT_HISTORY_FILE = "chat_history.json"
WORKOUT_HISTORY_FILE = "workout_history.json"

LAST_LOG_FILE = "last_log.csv"
PREV_LOG_FILE = "previous_log.csv"

GRAPH_DIR = "static/graphs"
ACC_GRAPH = os.path.join(GRAPH_DIR, "acceleration.png")
TEMPO_GRAPH = os.path.join(GRAPH_DIR, "tempo.png")

os.makedirs(GRAPH_DIR, exist_ok=True)

# ----------------------------
# CHAT HISTORY
# ----------------------------
def load_chat_history():
    if not os.path.exists(CHAT_HISTORY_FILE):
        return []
    try:
        with open(CHAT_HISTORY_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_chat_history(history):
    with open(CHAT_HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)


# ----------------------------
# WORKOUT HISTORY
# ----------------------------
def load_workout_history():
    if not os.path.exists(WORKOUT_HISTORY_FILE):
        return []
    try:
        with open(WORKOUT_HISTORY_FILE, "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def save_workout_history(data):
    with open(WORKOUT_HISTORY_FILE, "w") as f:
        json.dump(data, f, indent=4)


# ----------------------------
# OLLAMA REQUEST
# ----------------------------
def ask_ollama(prompt, model="llama3.2"):
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()
        return response.json().get("response", "")
    except Exception as exc:
        return f"Coach reply could not be generated: {exc}"


# ----------------------------
# MARKDOWN TO HTML HELPERS
# ----------------------------
def markdown_images_to_html(text):
    return re.sub(r'!\[\]\((.*?)\)', r'<img src="\1" style="max-width:100%;">', text)


def markdown_bold_to_html(text):
    return re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)


def bulletize_response(text):
    """Add bullets for clarity unless it's an existing structured analysis."""
    if "Workout Analysis" in text:
        return text
    lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
    if not lines:
        return text
    bullet_lines = []
    for ln in lines:
        if ln.startswith(("•", "-", "·")):
            bullet_lines.append(ln)
        else:
            bullet_lines.append(f"• {ln}")
    return "\n".join(bullet_lines)


# ----------------------------
# WORKOUT ANALYSIS HELPERS
# ----------------------------
def normalize_timestamp_column(df):
    if "server_timestamp" in df.columns:
        server_ts = pd.to_datetime(df["server_timestamp"], errors="coerce")
        if server_ts.notna().sum() >= 2:
            base = server_ts.dropna().iloc[0]
            df["timestamp"] = (server_ts - base).dt.total_seconds()
            return

    if "arduino_timestamp" in df.columns:
        arduino_ts = pd.to_numeric(df["arduino_timestamp"], errors="coerce")
        arduino_ts = arduino_ts - arduino_ts.min()
        arduino_ts = arduino_ts.fillna(0)

        span = arduino_ts.max()
        if span > 10000:
            arduino_ts = arduino_ts / 1000.0
            span = arduino_ts.max()
        if span > 10000:
            arduino_ts = arduino_ts / 1000.0

        df["timestamp"] = arduino_ts
        return

    df["timestamp"] = np.arange(len(df))


def format_duration(seconds):
    seconds = max(0, float(seconds))
    minutes = int(seconds // 60)
    secs = int(round(seconds % 60))
    return f"{minutes} min {secs} sec"


def split_reps_into_sets(rep_times, tempos):
    if len(rep_times) == 0:
        return []
    if len(tempos) == 0:
        return [len(rep_times)]

    threshold = np.mean(tempos) * 1.6
    sets = []
    current = 1
    for gap in tempos:
        if gap > threshold:
            sets.append(current)
            current = 1
        else:
            current += 1
    sets.append(current)
    return sets


def format_analysis_report(analysis):
    reps_per_set = analysis.get("reps_per_set", [])
    rep_display = reps_per_set if reps_per_set else []

    return (
        "\U0001F3CB\ufe0f Workout Analysis\n\n"
        f"\u2022 Duration: {analysis.get('duration_text', '0 min 0 sec')}\n"
        f"\u2022 Sets: {analysis.get('sets', 0)}\n"
        f"\u2022 Total repetitions: {analysis.get('rep_count', 0)}\n"
        f"\u2022 Reps per set: {rep_display}\n\n"
        "\U0001F9E0 Form Quality\n"
        f"\u2022 Average form score: {analysis.get('form_score', 0)}%\n"
        f"\u2022 Best rep: {analysis.get('best_rep', '-')}\n"
        f"\u2022 Issue detected: {analysis.get('issue', 'Unknown')}\n"
        f"\u2022 Recommendation: {analysis.get('recommendation', 'Not available')}\n"
    )


def build_coach_prompt(analysis_report, user_question):
    return (
        "You are a concise workout coach.\n"
        "Use ONLY the existing analysis below. Do not invent new metrics, reps, or sets.\n"
        "If data is missing, say it is not available. Keep answers short and directive.\n\n"
        "Always respond in English only, even if the user writes in another language.\n\n"
        f"Analysis:\n{analysis_report}\n\n"
        f"Question:\n{user_question}\n\n"
        "Guidelines:\n"
        "- Do not create new analysis.\n"
        "- Stay within the reported values.\n"
        "- Provide one clear coaching explanation or cue.\n"
    )


def generate_workout_analysis(df):
    try:
        df.columns = df.columns.str.strip()

        df = df.rename(columns={
            "X": "acc_x", "Y": "acc_y", "Z": "acc_z",
            "x": "acc_x", "y": "acc_y", "z": "acc_z",
            "AccelX": "acc_x", "AccelY": "acc_y", "AccelZ": "acc_z"
        })

        normalize_timestamp_column(df)

        if not {"acc_x", "acc_y", "acc_z", "timestamp"}.issubset(df.columns):
            return {"error": f"CSV format error. Found columns: {df.columns.tolist()}"}, None, None

        df["total_acc"] = np.sqrt(
            df["acc_x"] ** 2 + df["acc_y"] ** 2 + df["acc_z"] ** 2
        )

        timestamps = df["timestamp"]

        plt.figure(figsize=(10, 4))
        plt.plot(timestamps, df["total_acc"])
        plt.title("Acceleration Over Time")
        plt.xlabel("Time")
        plt.ylabel("Total Acceleration")
        plt.tight_layout()
        plt.savefig(ACC_GRAPH)
        plt.close()

        peaks, _ = find_peaks(df["total_acc"], distance=10, prominence=0.2)
        rep_count = len(peaks)
        rep_times = np.asarray(timestamps.iloc[peaks], dtype=float)

        tempos = np.diff(rep_times) if len(rep_times) > 1 else np.array([])

        tempo_graph = None
        if len(tempos) > 0:
            plt.figure(figsize=(10, 4))
            plt.plot(range(1, len(tempos) + 1), tempos)
            plt.title("Rep Tempo")
            plt.xlabel("Rep Number")
            plt.ylabel("Seconds Between Reps")
            plt.tight_layout()
            plt.savefig(TEMPO_GRAPH)
            plt.close()
            tempo_graph = TEMPO_GRAPH

        reps_per_set = split_reps_into_sets(rep_times, tempos)
        sets = len(reps_per_set)

        peak_heights = np.asarray(df["total_acc"].iloc[peaks], dtype=float) if rep_count > 0 else np.array([])
        tempo_cv = float(np.std(tempos) / np.mean(tempos)) if len(tempos) > 1 and np.mean(tempos) > 0 else 0
        peak_var = float(np.std(peak_heights) / np.mean(peak_heights)) if rep_count > 1 and np.mean(peak_heights) > 0 else 0

        form_score = 100
        form_score -= min(tempo_cv * 100, 25)
        form_score -= min(peak_var * 50, 20)
        form_score = max(45, min(98, round(form_score, 1)))

        best_rep = int(np.argmax(peak_heights) + 1) if rep_count > 0 else "-"

        if rep_count == 0:
            issue = "No repetitions detected"
            recommendation = "Check CSV columns (X,Y,Z,timestamp) and upload again."
        elif tempo_cv > 0.25:
            issue = "Tempo inconsistent"
            recommendation = "Use a steady 3-1-3 count to keep each rep duration consistent."
        elif peak_var > 0.35:
            issue = "Power output unstable"
            recommendation = "Move down and up at the same speed; smooth the last 3 reps."
        else:
            issue = "Form stable"
            recommendation = "Keep this rhythm; consider a 5% load increase next set."

        duration_seconds = float(timestamps.max() - timestamps.min()) if len(timestamps) > 1 else 0.0

        analysis = {
            "duration_seconds": duration_seconds,
            "duration_text": format_duration(duration_seconds),
            "sets": sets,
            "rep_count": int(rep_count),
            "reps_per_set": [int(x) for x in reps_per_set] if reps_per_set else [],
            "form_score": form_score,
            "best_rep": best_rep,
            "issue": issue,
            "recommendation": recommendation,
        }

        return analysis, ACC_GRAPH, tempo_graph

    except Exception as e:
        return {"error": f"Error analyzing workout: {str(e)}"}, None, None


# ----------------------------
# ROUTES
# ----------------------------
@app.route("/", methods=["GET"])
def home():
    history = load_chat_history()
    return render_template("index.html", history=history)


@app.route("/workouts", methods=["GET"])
def get_workouts():
    return jsonify(load_workout_history())


@app.route("/send", methods=["POST"])
def send_message():
    history = load_chat_history()
    workout_history = load_workout_history()

    user_message = request.form.get("message", "").strip()
    logfile = request.files.get("logfile")

    if user_message:
        history.append({"role": "user", "text": user_message})

    response_text = ""

    if logfile:
        if os.path.exists(LAST_LOG_FILE):
            try:
                os.replace(LAST_LOG_FILE, PREV_LOG_FILE)
            except PermissionError:
                # Windows can lock the file; best effort rotate
                pass

        logfile.save(LAST_LOG_FILE)
        df = pd.read_csv(LAST_LOG_FILE)

        analysis, acc_graph, tempo_graph = generate_workout_analysis(df)

        if "error" in analysis:
            response_text = analysis["error"]
        else:
            report_text = format_analysis_report(analysis)
            workout_history.append({
                "timestamp": datetime.now().isoformat(),
                "analysis": analysis,
                "report": report_text,
                "acc_graph": acc_graph,
                "tempo_graph": tempo_graph,
            })
            save_workout_history(workout_history)
            response_text = report_text
    elif user_message:
        lower_msg = user_message.lower()
        smalltalk = any(kw in lower_msg for kw in ["teşekkür", "tesekkur", "thanks", "thank you", "sağ ol", "sag ol"])

        last_entry = workout_history[-1] if workout_history else None
        if smalltalk and not last_entry:
            response_text = "You're welcome! Need anything else or want to upload a log?"
        elif smalltalk and last_entry:
            response_text = "You're welcome! Want another cue or a fresh log?"
        elif last_entry:
            report_text = last_entry.get("report") or format_analysis_report(last_entry.get("analysis", {}))
            prompt = build_coach_prompt(report_text, user_message)
            response_text = html.unescape(ask_ollama(prompt))
        else:
            response_text = "Upload a CSV first to generate an analysis."
    else:
        return jsonify({"response": ""})

    bulleted = bulletize_response(response_text)

    formatted = markdown_images_to_html(bulleted)
    formatted = markdown_bold_to_html(formatted)
    formatted = formatted.replace("\n", "<br>")

    history.append({"role": "assistant", "text": formatted})
    save_chat_history(history)

    return jsonify({"response": formatted})


# ----------------------------
# RUN
# ----------------------------
if __name__ == "__main__":
    app.run(debug=False)
