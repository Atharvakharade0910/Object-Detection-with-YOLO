"""YOLO Studio: native web UI and local inference API; no Streamlit dependency."""
from pathlib import Path
from functools import wraps
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import base64, hashlib, hmac, io, json, logging, math, os, secrets, subprocess, threading, time, uuid
import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from PIL.Image import open as pillow_open
from flask import Flask, jsonify, request, session, send_from_directory, send_file
os.environ.setdefault('YOLO_AUTOINSTALL', 'False')
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT / 'outputs' / 'web'
OUTPUT.mkdir(parents=True, exist_ok=True)
app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config.update(SECRET_KEY=os.environ.get('YOLO_SESSION_SECRET') or secrets.token_hex(32), MAX_CONTENT_LENGTH=100*1024*1024,
                  SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE='Strict')
models, jobs = {}, {}
model_lock, jobs_lock, users_lock = threading.Lock(), threading.Lock(), threading.Lock()
executor = ThreadPoolExecutor(max_workers=1)

def users():
    path = ROOT / 'users.json'
    return json.loads(path.read_text()) if path.exists() else {'admin': hashlib.sha256(b'admin123').hexdigest()}

def authenticated(fn):
    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get('user'):
            return jsonify(error='Please sign in to continue.'), 401
        return fn(*args, **kwargs)
    return wrapped

@app.before_request
def same_origin():
    if request.method in ('POST', 'DELETE', 'PUT'):
        origin = request.headers.get('Origin')
        if origin and origin != request.host_url.rstrip('/'):
            return jsonify(error='Request origin is not allowed.'), 403

@app.after_request
def headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    if request.path.startswith('/api/'):
        response.headers['Cache-Control'] = 'no-store'
    return response

@app.errorhandler(413)
def too_large(_):
    return jsonify(error='Choose a file under 100 MB.'), 413

@app.errorhandler(Exception)
def unexpected(error):
    from werkzeug.exceptions import HTTPException
    if isinstance(error, HTTPException):
        return jsonify(error=error.description), error.code
    logging.exception('API request failed')
    return jsonify(error='Processing failed. Try another file or check the server log.'), 500

@app.get('/')
def index():
    return send_from_directory(ROOT / 'static', 'index.html')

@app.get('/api/health')
def health():
    return jsonify(status='ok', engine='YOLOv8', frontend='HTML / CSS / JavaScript')

@app.get('/api/session')
def current_session():
    return jsonify(user=session.get('user'))

@app.post('/api/login')
def login():
    data = request.get_json(silent=True) or {}
    username, password = str(data.get('username', '')), str(data.get('password', ''))
    stored = users().get(username, '')
    if not stored or not hmac.compare_digest(stored, hashlib.sha256(password.encode()).hexdigest()):
        return jsonify(error='Username or password is incorrect.'), 401
    session.clear()
    session['user'] = username
    return jsonify(user=username)

@app.post('/api/logout')
def logout():
    session.clear()
    return jsonify(ok=True)

@app.route('/api/users', methods=['GET', 'POST', 'DELETE', 'PUT'])
@authenticated
def manage_users():
    if session['user'] != 'admin':
        return jsonify(error='Administrator access required.'), 403
    with users_lock:
        records = users()
        data = request.get_json(silent=True) or {}
        name, password = str(data.get('username', '')).strip(), str(data.get('password', ''))
        if request.method == 'POST':
            if not name or len(name) > 60 or name in records or len(password) < 8:
                return jsonify(error='Use a new username and a password of at least 8 characters.'), 400
            records[name] = hashlib.sha256(password.encode()).hexdigest()
        elif request.method == 'DELETE':
            if name == 'admin':
                return jsonify(error='The administrator cannot be deleted.'), 400
            records.pop(name, None)
        elif request.method == 'PUT':
            if not hmac.compare_digest(records['admin'], hashlib.sha256(str(data.get('current', '')).encode()).hexdigest()) or len(password) < 8:
                return jsonify(error='Check the current password. The new password needs at least 8 characters.'), 400
            records['admin'] = hashlib.sha256(password.encode()).hexdigest()
        if request.method != 'GET':
            temp = ROOT / 'users.json.tmp'
            temp.write_text(json.dumps(records, indent=2))
            temp.replace(ROOT / 'users.json')
    return jsonify(users=list(records))

def model_for(task):
    if task not in models:
        models[task] = YOLO(ROOT / 'weights' / ('yolov8n-seg.pt' if task == 'segment' else 'yolov8n.pt'))
    return models[task]

@app.get('/api/meta')
@authenticated
def meta():
    with model_lock:
        names = model_for('detect').names
    return jsonify(classes=[{'id': k, 'name': v} for k, v in names.items()], max_upload_mb=100)

@app.get('/api/sample')
@authenticated
def sample():
    return send_file(ROOT / 'images' / 'office_4.jpg')

def options():
    task = request.form.get('task', 'detect')
    confidence = float(request.form.get('confidence', '0.4'))
    classes = json.loads(request.form.get('classes', '[]'))
    if task not in ('detect', 'segment') or not math.isfinite(confidence) or not .1 <= confidence <= 1:
        raise ValueError('Invalid task or confidence.')
    if not isinstance(classes, list) or any(type(c) is not int or not 0 <= c < 80 for c in classes):
        raise ValueError('Invalid object class selection.')
    return task, confidence, classes or None

def decode_image(raw):
    # Keep Pillow's original decoder: Ultralytics patches Image.open with a
    # HEIF fallback that can attempt package installation for malformed uploads.
    with pillow_open(io.BytesIO(raw), formats=['JPEG', 'PNG', 'WEBP', 'BMP']) as img:
        if img.width * img.height > 30_000_000:
            raise ValueError('Use an image below 30 megapixels.')
        img = ImageOps.exif_transpose(img).convert('RGB')
        img.thumbnail((1920, 1920))
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def pack_result(result, elapsed):
    detections = []
    masks = result.masks.xy if result.masks is not None else []
    for i, box in enumerate(result.boxes):
        cls = int(box.cls.item())
        item = dict(id=i, class_id=cls, name=result.names[cls], confidence=round(float(box.conf.item()), 4),
                    box=[round(float(v), 1) for v in box.xyxy[0].tolist()])
        if i < len(masks):
            item['polygon'] = masks[i].round(1).tolist()
        detections.append(item)
    height, width = result.orig_img.shape[:2]
    return dict(detections=detections, counts=dict(Counter(d['name'] for d in detections)),
                width=width, height=height, elapsed_ms=round(elapsed * 1000))

@app.post('/api/detect')
@authenticated
def detect():
    try:
        task, confidence, classes = options()
        file = request.files.get('file')
        if not file:
            raise ValueError('Choose an image first.')
        frame = decode_image(file.read())
    except (ValueError, TypeError, UnidentifiedImageError, OSError, Image.DecompressionBombError):
        return jsonify(error='Invalid image or settings. Use a JPG, PNG, WEBP or BMP below 30 megapixels.'), 400
    start = time.perf_counter()
    with model_lock:
        result = model_for(task).predict(frame, conf=confidence, classes=classes, verbose=False)[0]
    payload = pack_result(result, time.perf_counter() - start)
    ok, jpeg = cv2.imencode('.jpg', frame)
    if not ok:
        raise RuntimeError('Image encoding failed')
    payload.update(image='data:image/jpeg;base64,' + base64.b64encode(jpeg).decode(), task=task)
    return jsonify(payload)

def update_job(job_id, **values):
    with jobs_lock:
        jobs[job_id].update(values)

def process_video(job_id, source, task, confidence, classes, tracking):
    capture, writer = None, None
    raw_output, output = OUTPUT / f'{job_id}.raw.mp4', OUTPUT / f'{job_id}.mp4'
    try:
        import imageio_ffmpeg
        update_job(job_id, status='processing')
        capture = cv2.VideoCapture(str(source))
        fps = capture.get(cv2.CAP_PROP_FPS)
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if not capture.isOpened() or not math.isfinite(fps) or fps <= 0 or total <= 0:
            raise ValueError('This video could not be decoded. Try an MP4 video.')
        if total / fps > 120:
            raise ValueError('Please use a video up to 2 minutes long.')
        local_model = YOLO(ROOT / 'weights' / ('yolov8n-seg.pt' if task == 'segment' else 'yolov8n.pt'))
        counts, unique_ids, frames = Counter(), set(), 0
        start = time.perf_counter()
        while True:
            with jobs_lock:
                cancelled = jobs[job_id].get('cancel', False)
            if cancelled:
                update_job(job_id, status='cancelled')
                return
            ok, frame = capture.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = min(1, 960 / max(h, w))
            frame = cv2.resize(frame, (max(2, int(w*scale)//2*2), max(2, int(h*scale)//2*2)))
            if writer is None:
                writer = cv2.VideoWriter(str(raw_output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
                if not writer.isOpened():
                    raise ValueError('Video encoder could not start.')
            with model_lock:
                if tracking:
                    result = local_model.track(frame, conf=confidence, classes=classes, persist=True, tracker='bytetrack.yaml', verbose=False)[0]
                else:
                    result = local_model.predict(frame, conf=confidence, classes=classes, verbose=False)[0]
            counts.update(result.names[int(c)] for c in result.boxes.cls.tolist())
            if result.boxes.id is not None:
                unique_ids.update(int(v) for v in result.boxes.id.tolist())
            writer.write(result.plot())
            frames += 1
            update_job(job_id, progress=min(94, round(frames/total*94)), frames=frames, total_frames=total)
        if frames == 0:
            raise ValueError('This video does not contain readable frames.')
        writer.release()
        writer = None
        capture.release()
        capture = None
        update_job(job_id, status='encoding', progress=95)
        subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', str(raw_output), '-an', '-c:v', 'libx264',
                        '-preset', 'ultrafast', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(output)],
                       check=True, capture_output=True, timeout=180, creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
        with jobs_lock:
            cancelled = jobs[job_id].get('cancel', False)
        if cancelled:
            output.unlink(missing_ok=True)
            update_job(job_id, status='cancelled')
            return
        update_job(job_id, status='complete', progress=100, counts=dict(counts), unique_tracks=len(unique_ids), tracking=tracking,
                   elapsed_ms=round((time.perf_counter()-start)*1000), url=f'/api/videos/{job_id}/file')
    except Exception as error:
        logging.exception('Video processing failed')
        update_job(job_id, status='failed', error=str(error) if isinstance(error, ValueError) else 'Video processing failed. Try a shorter MP4 file.')
        output.unlink(missing_ok=True)
    finally:
        if capture is not None: capture.release()
        if writer is not None: writer.release()
        source.unlink(missing_ok=True)
        raw_output.unlink(missing_ok=True)

@app.post('/api/videos')
@authenticated
def upload_video():
    try:
        task, confidence, classes = options()
    except (ValueError, TypeError):
        return jsonify(error='Invalid video settings.'), 400
    file = request.files.get('file')
    if not file or Path(file.filename or '').suffix.lower() not in ('.mp4', '.mov', '.avi', '.mkv', '.webm'):
        return jsonify(error='Choose an MP4, MOV, AVI, MKV or WEBM video.'), 400
    with jobs_lock:
        if any(j['status'] in ('queued', 'processing', 'encoding') for j in jobs.values()):
            return jsonify(error='A video is already processing. Wait for it to finish or cancel it.'), 409
        for old_id in list(jobs):
            if time.time()-jobs[old_id]['created'] > 3600:
                (OUTPUT/f'{old_id}.mp4').unlink(missing_ok=True)
                del jobs[old_id]
        job_id = uuid.uuid4().hex
        jobs[job_id] = dict(id=job_id, owner=session['user'], created=time.time(), status='queued', progress=0)
    source = OUTPUT / f'{job_id}.upload{Path(file.filename).suffix.lower()}'
    try:
        file.save(source)
        executor.submit(process_video, job_id, source, task, confidence, classes, request.form.get('tracking') == 'true')
    except Exception:
        source.unlink(missing_ok=True)
        update_job(job_id, status='failed')
        raise
    return jsonify(id=job_id), 202

@app.route('/api/videos/<job_id>', methods=['GET', 'DELETE'])
@authenticated
def video_status(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job['owner'] != session['user']:
            return jsonify(error='Video not found.'), 404
        if request.method == 'DELETE': job['cancel'] = True
        return jsonify({k:v for k,v in job.items() if k not in ('owner','created','cancel')})

@app.get('/api/videos/<job_id>/file')
@authenticated
def video_file(job_id):
    with jobs_lock:
        job = jobs.get(job_id)
        if not job or job['owner'] != session['user'] or job['status'] != 'complete':
            return jsonify(error='Video is not ready.'), 404
    return send_file(OUTPUT/f'{job_id}.mp4', mimetype='video/mp4', conditional=True,
                     as_attachment=request.args.get('download') == '1', download_name='yolo-studio-result.mp4')

def run():
    from waitress import serve
    print('YOLO Studio: http://127.0.0.1:8501', flush=True)
    serve(app, host='127.0.0.1', port=8501, threads=6)

if __name__ == '__main__':
    run()
