import cv2
import torch
import threading
import datetime
import sqlite3
import numpy as np
import io
import queue
from flask_socketio import SocketIO, emit

from send_mail import send_alert_email

class SecurityDetection:
    MaxAssociationDistance = 500
    AccumulatedWeightAlpha = 0.9
    DetectionThreshold = 1.0
    MinDetectionArea = 7000

    def __init__(self, socketio, model_name='yolov5s', db_path='events.db', video_folder='static/'):
        self.socketio = socketio

        self.model = torch.hub.load('ultralytics/yolov5', model_name)
        self.db_path = db_path
        self.video_folder = video_folder
        
        self.thread = None
        self.running = False
        self.finishing = False
        self.is_armed = False
        self.is_live_view = False

        self.video_writer = None
        self.recording = False
        self.start_timestamp = None
        self.video_filename = None
 
        self.closing_kernel = np.ones((40, 40), np.uint8)
        self.background = None
        self.fps = None
        self.frame_width = None

        # Initialize trackers
        self.trackers = []
        self.detected_objects_set = set()

        self.recording_frame_counter = 0

        # User settings
        self.name = ""
        self.email = ""
        self.filters = []

        # Advanced User settings
        self.confidence = 0.5

        self.new_events = []

        self.frame_queue = queue.Queue()
        self.lock = threading.Lock()    

    def update_settings(self, settings_dict):
        for key, value in settings_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)

    def start(self):
        with self.lock:
            if not self.running and not self.finishing:
                print("Starting camera")
                self.running = True
                self.thread = threading.Thread(target=self.detection_loop)
                self.thread.start()

    def arm_system(self, is_armed):
        self.is_armed = is_armed

        if is_armed:
            self.start()

    def stop(self):
        with self.lock:
            if self.running and not self.finishing and not self.is_live_view:
                print("Stopping camera")
                self.finishing = True
                self.running = False
                if self.thread is not None:
                    self.thread.join()
                self.finishing = False

    def remove_close_points(self, centroids, filters):
        return [centroid for centroid in centroids if all(
            np.linalg.norm(np.array(centroid['centroid']) - filter_dict['last_centroid']['centroid']) >= SecurityDetection.MaxAssociationDistance
            for filter_dict in filters
        )]
    
    def create_centroids(self, contours):
        centroids = []
        for contour in contours:
            if cv2.contourArea(contour) < SecurityDetection.MinDetectionArea:
                continue
            
            # Get the center of the centriod.
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                centroids.append({'centroid': (cx, cy), 'contour': contour})
        return centroids
    
    def track_objects(self, contours):
        centroids = self.create_centroids(contours)
        
        self.update_tracker(centroids)
        
        # Remove close points between centroids and predictions
        centroids = self.remove_close_points(centroids, self.trackers)
        
        # Create new tracker for unmatched centroids
        while (len(centroids) > 0):
            self.trackers.append({'frames_since_last_update': 0, 'last_centroid': centroids[0]})
            centroids.pop(0)
            # If there are now any other centroids that are still close, remove them
            centroids = self.remove_close_points(centroids, self.trackers)
            
        max_frames_since_update = self.fps
        # Delete any old trackers. This ensures the count is accurate.
        self.trackers = [tracker for tracker in self.trackers if tracker['frames_since_last_update'] <= max_frames_since_update]
    
    def update_tracker(self, centroids):
        # Predict next state and correct with measurements for each tracker
        for tracker in self.trackers:        
            # Find the closest centroid within the maximum association distance
            min_distance = float('inf')
            closest_centroid = None

            for centroid_info in centroids:
                # Get the distance between the centroid and the tracked objects.
                distance = np.linalg.norm(np.array(centroid_info['centroid']) - np.array(tracker['last_centroid']['centroid']))

                # Store the closest distance.
                if distance < min_distance:
                    min_distance = distance
                    closest_centroid = centroid_info

            # If the closest centroid is within the maximum association distance, correct prediction with measurement
            if closest_centroid is not None and min_distance < SecurityDetection.MaxAssociationDistance:
                tracker['last_centroid'] = closest_centroid
                tracker['frames_since_last_update'] = 0

                # Remove the associated centroid from the list to avoid re-association
                centroids.remove(closest_centroid)
            # Didn't update this time
            else:
                tracker['frames_since_last_update'] += 1

    def send_detection_email(self, detected_objects, image):
        if self.email:
            if not self.filters:
                subject = f"Motion detected in {self.name} Security System"
            elif (bool(set(self.filters) & self.detected_objects_set)):
                subject = f"Known object detect in {self.name} Security System"
            else:
                subject = f"Unknown object detect in {self.name} Security System"
            
            # TODO: Add more details to body (Maybe even an image)
            body = f"Found the following objects during recording: {detected_objects}"
            send_alert_email(subject, body, image, self.email)

    def start_recording(self):
        self.recording = True
        self.start_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.video_filename = f'static/event_{self.start_timestamp}.webm'
        fourcc = cv2.VideoWriter_fourcc(*'VP80')  # Specify codec
        self.video_writer = cv2.VideoWriter(self.video_filename, fourcc, 10.0, (640, 480))
        print(f"Motion detected at {self.start_timestamp}, starting recording")

    def stop_recording(self):
        print("Stopping recording - No Tracked objects")

        self.video_writer.release()  # Release VideoWriter object

        image = self.extract_halfway_frame(self.video_filename)

        video_path = self.video_filename.replace('static/', '')
        thumbnail_path = video_path.replace('.webm', '.jpg')
        detected_objects = ", ".join(self.detected_objects_set) 

        self.save_event_to_db(self.start_timestamp, video_path, detected_objects, thumbnail_path)
        self.send_detection_email(detected_objects, image)
        self.socketio.emit('new_event', {'data': 'New event detected!'})

        self.recording_frame_counter = 0
        self.detected_objects_set.clear()
        self.recording = False

    def motion_detection(self, gray):
        # Update background model
        cv2.accumulateWeighted(gray, self.background, SecurityDetection.AccumulatedWeightAlpha)

        # Compute absolute difference between current frame and background.
        frameDelta = cv2.absdiff(gray, cv2.convertScaleAbs(self.background))

        # Threshold the delta image to obtain the foreground mask.
        _, fgmask = cv2.threshold(frameDelta, SecurityDetection.DetectionThreshold, 255, cv2.THRESH_BINARY)

        # Apply morphological operations to remove noise and close gaps.
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        fgmask = cv2.morphologyEx(fgmask, cv2.MORPH_CLOSE, self.closing_kernel)

        contours, _ = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        return contours

    def add_detections_to_image(self, frame):
        # Draw bounding boxes around detected objects in the left foreground mask
        for tracker in self.trackers:
            contour = tracker['last_centroid']['contour']
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # Define the text to display
        tracked_count_text = f"Object Count: {len(self.trackers)}"

        # Define the position to display the text (top-right corner)
        text_position = (self.frame_width - 150 - len(tracked_count_text) * 8, 30)  # Adjust the position as needed

        # Display the text on the frame
        cv2.putText(frame, tracked_count_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    def run_security_detection(self, frame):
        # Change the frame from colour to grayscale.
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Initialize background model on the first frame
        if self.background is None:
            self.background = gray.copy().astype(float)
            return

        # Run motion detection.
        contours = self.motion_detection(gray)
        
        # Run object tracking.
        self.track_objects(contours)
            
        self.add_detections_to_image(frame)
        
        # Start recording
        if len(self.trackers) > 0 and not self.recording:
            self.start_recording()

        if self.recording:
            # Run yolo every self.fps frames.
            if self.recording_frame_counter % self.fps == 0:
                frame, detected_objects = self.get_detections(frame)  # Perform YOLO object detection on the frame
                self.detected_objects_set.update(detected_objects)  # Update the set with detected objects
            
            self.recording_frame_counter = self.recording_frame_counter + 1
            self.video_writer.write(frame)  # Write frame to video file

            if len(self.trackers) == 0:
                self.new_events.append("Event Details")
                self.stop_recording()

    def detection_loop(self):
        # Connect to the webcam.
        webcam = cv2.VideoCapture(0)
        # Get camera details.
        self.frame_width = int(webcam.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.fps = int(webcam.get(cv2.CAP_PROP_FPS))

        # Start thread loop.
        while self.running:
            # Read camera frame.
            ret, frame = webcam.read()
            if ret == False: break

            # Put frame in the queue for live streaming
            if self.is_live_view:
                if not self.frame_queue.full():
                    self.frame_queue.put(frame)

            if self.is_armed:
                self.run_security_detection(frame)
            elif self.recording:
                self.stop_recording()

            if cv2.waitKey(1) == 13: break
        
        if self.recording:
            self.stop_recording()

        self.frame_queue.queue.clear()

        webcam.release()
        cv2.destroyAllWindows()

    def get_detections(self, frame):
        results = self.model(frame)
        detected_objects = []
        for detection in results.xyxy[0]:  # xyxy format
            xmin, ymin, xmax, ymax, conf, class_id = detection
            if conf > self.confidence:  # Filter weak detections
                label = self.model.names[int(class_id)]
                detected_objects.append(label)
        return frame, detected_objects

    def save_event_to_db(self, timestamp, video_path, detected_objects, thumbnail):
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("INSERT INTO events (timestamp, video_path, detected_objects, thumbnail_path) VALUES (?, ?, ?, ?)",
                  (timestamp, video_path, detected_objects, thumbnail))
        conn.commit()
        conn.close()

    def extract_halfway_frame(self, video_path):
        cap = cv2.VideoCapture(video_path)
        
        # Get the total number of frames
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate the frame number for the halfway point
        halfway_frame = total_frames // 2
        
        # Set the video capture position to the halfway frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, halfway_frame)
        
        # Read the frame
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Save to file
            thumbnail_path = video_path.replace('.webm', '.jpg')
            cv2.imwrite(thumbnail_path, frame)

            # Convert frame to JPEG in memory
            is_success, img_bytes = cv2.imencode('.jpg', frame)
            if is_success:
                return io.BytesIO(img_bytes.tobytes())
        return None

    def get_frame(self):
        return self.frame_queue.get() if not self.frame_queue.empty() else None
