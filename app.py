import os
from flask import Flask, request, render_template, send_file, redirect, url_for
from werkzeug.utils import secure_filename
from Main_TeleFM_Mohammad import frailty_preprocess
from mohammad_signal_analysis import frailty_postprocess

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_FOLDER = os.path.join("Results", "frailty_index")
PROTOCOL_PATH = os.path.join("static", "protocol.pdf")  # Put your PDF here
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)

last_result_filename = None

@app.route("/", methods=["GET", "POST"])
def index():
    global last_result_filename
    if request.method == "POST":
        file = request.files['video']
        filename = secure_filename(file.filename)
        video_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(video_path)

        task_type = request.form['task_type']
        handedness = request.form['handedness']
        RIGHT = handedness == 'Right'
        FLIP = request.form['flip'] == 'True'

        sub_id = os.path.splitext(filename)[0]
        gif_output = os.path.join("Results", "gif_files")
        angle_output = os.path.join("Results", "Angle")
        angular_acc = os.path.join("Results", "angular_acc")
        video_file_location = UPLOAD_FOLDER
        excel_file = os.path.join(angle_output, f"{sub_id}.xlsx")

        frailty_preprocess(video_path, gif_output, angle_output, False, RIGHT, FLIP)
        last_result_filename = frailty_postprocess(
            excel_file, RESULT_FOLDER, task_type, video_file_location, angular_acc
        )

        return redirect(url_for("index", success=1))
    return render_template("index.html")

@app.route("/download")
def download():
    global last_result_filename
    if last_result_filename and os.path.exists(last_result_filename):
        return send_file(last_result_filename, as_attachment=True)
    return "No result found. Please run an assessment first."

@app.route("/protocol")
def download_protocol():
    if os.path.exists(PROTOCOL_PATH):
        return send_file(PROTOCOL_PATH, as_attachment=True)
    return "Protocol PDF not found."

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
