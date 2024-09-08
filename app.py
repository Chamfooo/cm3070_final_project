from flask import Flask, render_template, redirect, Response, request, jsonify, url_for, g
from flask_socketio import SocketIO, emit

import sqlite3
import os
import json
import time
import cv2

from typing import List, Dict, Any
from security_detector import SecurityDetection

app = Flask(__name__, static_folder='static')
app.config['SECRET_KEY'] = 'secret!'
DATABASE = 'events.db'
USER_SETTIGNS_FILE = 'user_settings.json'

socketio = SocketIO(app)

def init_db():
    conn = sqlite3.connect('events.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            video_path TEXT NOT NULL,
            thumbnail_path TEXT NOT NULL,
            detected_objects TEXT
        )
    ''')
    conn.commit()
    conn.close()

def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row  # This allows us to access columns by name
    return g.db

def read_user_settings(file_path: str) -> Dict[str, Any]:
    """
    Read user settings from a JSON file.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    Dict[str, Any]: A dictionary containing the user settings.
    """
    try:
        with open(file_path, 'r') as file:
            user_settings = json.load(file)
        return user_settings
    except FileNotFoundError:
        return {}
    except json.JSONDecodeError:
        return {}

def write_user_settings(file_path: str, user_settings: Dict[str, Any]) -> None:
    """
    Write user settings to a JSON file.

    Parameters:
    file_path (str): The path to the JSON file.
    user_settings (Dict[str, Any]): A dictionary containing the user settings.
    """
    with open(file_path, 'w') as file:
        json.dump(user_settings, file, indent=4)

def update_user_settings(file_path: str, name: str, email: str, filter_list: List[str]) -> None:
    """
    Update user settings in the JSON file.

    Parameters:
    file_path (str): The path to the JSON file.
    name (str, optional): The user's name. Defaults to None.
    email (str, optional): The user's email address. Defaults to None.
    filter_list (List[str], optional): A list of filters. Defaults to None.
    """
    user_settings = read_user_settings(file_path)
    
    user_settings['name'] = name
    user_settings['email'] = email
    user_settings['filters'] = filter_list

    write_user_settings(file_path, user_settings)

@app.teardown_appcontext
def close_connection(exception):
    db = getattr(g, 'db', None)
    if db is not None:
        db.close()

@app.route('/')
def index():
    db = get_db()
    cur = db.execute('SELECT id, timestamp, video_path, thumbnail_path, detected_objects FROM events')
    events = cur.fetchall()
    return render_template('index.html', 
                           events=events,
                           security_running=security_detector.running, 
                           security_finishing=security_detector.finishing)

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/list')
def list():
    return render_template('list.html')

@app.route('/get_settings', methods=['GET'])
def get_settings():
    user_settings = read_user_settings(USER_SETTIGNS_FILE)
    return jsonify(user_settings)

@app.route('/get_security_state', methods=['GET'])
def get_security_state():
    return security_detector.running

@app.route('/update_settings', methods=['POST'])
def update_settings():
    name = request.form.get('name')
    email = request.form.get('email')
    filter_list = request.form.getlist('filter')
    
    update_user_settings(USER_SETTIGNS_FILE, name, email, filter_list)
    security_detector.update_settings(read_user_settings(USER_SETTIGNS_FILE))

    return jsonify({'status': 'success'})

@app.route('/status')
def status():
    return jsonify({
        "running": security_detector.running,
        "armed": security_detector.is_armed,
        "live_view": security_detector.is_live_view,
        "recording": security_detector.recording,
        "finishing": security_detector.finishing
    })

@app.route('/event/<int:event_id>')
def event_detail(event_id):
    conn = sqlite3.connect('events.db')
    c = conn.cursor()
    c.execute('SELECT id, timestamp, video_path, thumbnail_path, detected_objects FROM events WHERE id = ?', (event_id,))
    event = c.fetchone()
    conn.close()
    if event:
        event_dict = {
            'id': event[0],
            'timestamp': event[1],
            'video_path': event[2].lstrip('static/'), 
            'thumbnail_path': event[3].lstrip('static/'),
            'detected_objects': event[4],
        }
        return render_template('event_detail.html', event=event_dict)
    else:
        return 'Event not found', 404

@app.route('/latest_events')
def latest_events():
    limit = request.args.get('limit', default=10, type=int)

    db = get_db()
    cur = db.execute('SELECT id, timestamp, video_path, thumbnail_path, detected_objects FROM events ORDER BY id DESC')
    events = cur.fetchall()
    events_list = []
    for event in events:
        events_list.append({
            'id': event[0],
            'timestamp': event[1],
            'video_path': event[2],
            'thumbnail_path': event[3],
            'detected_objects': event[4],
        })
    return jsonify(events_list[:limit])

@app.route('/delete/<int:event_id>')
def delete_event(event_id):
    db = get_db()
    cur = db.execute('SELECT video_path FROM events WHERE id = ?', (event_id,))
    event = cur.fetchone()
    if event:
        os.remove('static/' + event['video_path'])
        db.execute('DELETE FROM events WHERE id = ?', (event_id,))
        db.commit()
    return redirect(url_for('list'))

def gen_frames():
    # Start up the system if not already running
    security_detector.is_live_view = True
    security_detector.start()
    
    try:
        while True:
            frame = security_detector.get_frame()
            if frame is not None:
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            else:
                time.sleep(0.03)
    finally:
        security_detector.is_live_view = False

        # Stop the camera if the system is not armed
        if not security_detector.is_armed:
            security_detector.stop()

@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/toggle_security')
def toggle_security():
    if security_detector.is_armed == False:
        security_detector.arm_system(True)
        security_detector.start()
    elif security_detector.is_armed:
        security_detector.arm_system(False)
        if not security_detector.finishing:
            security_detector.stop()
    return '', 200

security_detector = SecurityDetection(socketio)
security_detector.update_settings(read_user_settings(USER_SETTIGNS_FILE))

if __name__ == '__main__':
    # Initialize the database
    init_db()

    # Ensure static directory exists
    if not os.path.exists('static'):
        os.makedirs('static')

    socketio.run(app, debug=True)
