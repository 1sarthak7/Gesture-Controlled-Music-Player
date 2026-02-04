import os
os.environ["SDL_AUDIODRIVER"] = "coreaudio"
import cv2
import mediapipe as mp  
import pygame
import numpy as np
import math
import time
import os
from collections import deque
# ---------------------------------

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================

# Camera
CAM_WIDTH, CAM_HEIGHT = 1280, 720
FPS = 30

# Hand Tracking
MAX_HANDS = 2
DETECTION_CONF = 0.7
TRACKING_CONF = 0.7

# Volume Control
VOL_MIN_DIST = 30   # Minimum pixel distance between thumb and index (0% volume)
VOL_MAX_DIST = 200  # Maximum pixel distance (100% volume)
VOL_SMOOTHING = 0.2 # EMA Alpha (0.0 to 1.0). Lower = smoother but more lag.

# Swipe Gestures
SWIPE_THRESHOLD = 50   # Pixels needed to register a swipe
SWIPE_COOLDOWN = 1.0   # Seconds between swipes
SWIPE_HISTORY_LEN = 5  # Frames to analyze for movement

# General Cooldowns
GESTURE_COOLDOWN = 1.0 # Seconds between state changes (Play/Pause)

# Paths
SONG_DIR = "songs"

# Colors (B, G, R)
COLOR_NEON_BLUE = (255, 255, 0)
COLOR_NEON_RED = (0, 0, 255)
COLOR_NEON_GREEN = (0, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

# ==========================================
# MODULE: MUSIC PLAYER
# ==========================================

# ==========================================
# MODULE: MUSIC PLAYER (UPDATED)
# ==========================================

class MusicPlayer:
    def __init__(self, song_dir):
        self.song_dir = song_dir
        self.playlist = []
        self.current_index = 0
        self.is_playing = False
        self.volume = 0.5
        self.has_started = False  # <--- NEW: Tracks if music has ever started
        
        # Initialize Pygame Mixer with explicit settings for macOS stability
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=2048)
        except pygame.error:
            pygame.mixer.init() # Fallback
            
        self._load_songs()
        
    def _load_songs(self):
        """Loads all .mp3 files from the directory."""
        if not os.path.exists(self.song_dir):
            os.makedirs(self.song_dir)
            print(f"[WARN] '{self.song_dir}' directory created. Please add MP3 files.")
            return

        self.playlist = [f for f in os.listdir(self.song_dir) if f.lower().endswith('.mp3')]
        self.playlist.sort()
        
        if not self.playlist:
            print("[WARN] No MP3 files found in songs directory.")
        else:
            print(f"[INFO] Loaded {len(self.playlist)} songs.")
            # Preload first song
            self.load_track(0)

    def load_track(self, index):
        if not self.playlist: return
        
        try:
            track_path = os.path.join(self.song_dir, self.playlist[index])
            pygame.mixer.music.load(track_path)
            self.current_index = index
            # Reset started flag when loading a new track manually, 
            # though play() usually handles this.
        except pygame.error as e:
            print(f"[ERROR] Failed to load song: {e}")

    def play(self):
        """Starts playback from the beginning."""
        if not self.playlist: return
        pygame.mixer.music.play()
        self.is_playing = True
        self.has_started = True

    def resume(self):
        """Resumes if paused, or Starts if never played."""
        if not self.playlist: return
        
        if not self.is_playing:
            if not self.has_started:
                # FIX: If music never started, 'unpause' does nothing. We must 'play'.
                self.play()
            else:
                # Standard unpause
                pygame.mixer.music.unpause()
                self.is_playing = True

    def pause(self):
        if self.is_playing:
            pygame.mixer.music.pause()
            self.is_playing = False

    def stop(self):
        pygame.mixer.music.stop()
        self.is_playing = False
        self.has_started = False # Reset so next Open Palm starts from fresh

    def next_track(self):
        if not self.playlist: return
        self.current_index = (self.current_index + 1) % len(self.playlist)
        self.load_track(self.current_index)
        self.play()

    def prev_track(self):
        if not self.playlist: return
        self.current_index = (self.current_index - 1) % len(self.playlist)
        self.load_track(self.current_index)
        self.play()

    def set_volume(self, level):
        """Level is 0.0 to 1.0"""
        self.volume = np.clip(level, 0.0, 1.0)
        pygame.mixer.music.set_volume(self.volume)

    def get_current_song_name(self):
        if not self.playlist: return "No Songs Found"
        return self.playlist[self.current_index]

# ==========================================
# MODULE: VISUAL EFFECTS (DOCTOR STRANGE)
# ==========================================

class VisualEffects:
    def __init__(self):
        self.rotation_angle = 0
    
    def draw_magic_circle(self, image, center, is_active=True):
        """Draws a rotating magic mandala effect at the hand center."""
        if center is None: return

        cx, cy = center
        time_seed = time.time() * 2
        
        # Base colors
        color_main = (0, 165, 255) # Orange-ish
        if not is_active:
            color_main = (100, 100, 100) # Grey if inactive

        # Pulsing radius
        pulse = math.sin(time_seed * 4) * 5
        base_radius = 40 + pulse

        # 1. Outer Ring with segments
        cv2.circle(image, (cx, cy), int(base_radius), color_main, 2)
        
        # 2. Rotating Square
        self.rotation_angle += 2
        rect_size = int(base_radius * 1.5)
        
        # Calculate rotated square vertices
        pts = []
        for i in range(4):
            theta = math.radians(self.rotation_angle + (i * 90))
            x = cx + int(rect_size * 0.7 * math.cos(theta))
            y = cy + int(rect_size * 0.7 * math.sin(theta))
            pts.append([x, y])
        
        pts = np.array(pts, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(image, [pts], True, color_main, 2)

        # 3. Inner Circle
        cv2.circle(image, (cx, cy), int(base_radius * 0.5), COLOR_WHITE, 1)

        # 4. Connecting Lines (Runes)
        for pt in pts:
            cv2.line(image, (cx, cy), (pt[0][0], pt[0][1]), color_main, 1)

# ==========================================
# MODULE: GESTURE RECOGNITION & LOGIC
# ==========================================

class GestureController:
    def __init__(self, player):
        self.player = player
        self.last_gesture_time = 0
        self.last_swipe_time = 0
        
        # Swipe tracking: Store recent wrist x-positions
        self.wrist_history = deque(maxlen=SWIPE_HISTORY_LEN)
        
        # Volume smoothing
        self.prev_vol = 0.5
        
        # Hand State Constants
        self.STATE_OPEN = "OPEN"
        self.STATE_FIST = "FIST"
        self.STATE_UNKNOWN = "UNKNOWN"

    def _get_hand_state(self, landmarks):
        """
        Determines if hand is OPEN or FIST.
        Logic: Check if fingertips are below finger PIP joints (folded).
        Note: This assumes hand is upright. More robust checks involve joint angles.
        """
        # Fingertip indices: 8 (Index), 12 (Middle), 16 (Ring), 20 (Pinky)
        # PIP Joint indices: 6, 10, 14, 18
        # Thumb is ignored for simple Open/Fist detection here
        
        fingers_open = 0
        # Check Index, Middle, Ring, Pinky
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        # Wrist is 0. If tip is further from wrist than PIP, it's open.
        # Calculating distance to wrist (0)
        wrist = landmarks[0]
        
        for tip_idx, pip_idx in zip(tips, pips):
            # We use Euclidean distance comparison for rotation invariance
            tip = landmarks[tip_idx]
            pip = landmarks[pip_idx]
            
            dist_tip_wrist = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            dist_pip_wrist = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
            
            if dist_tip_wrist > dist_pip_wrist:
                fingers_open += 1
                
        if fingers_open >= 3:
            return self.STATE_OPEN
        elif fingers_open == 0:
            return self.STATE_FIST
        return self.STATE_UNKNOWN

    def _process_volume(self, landmarks, img_shape):
        """Calculates distance between Thumb (4) and Index (8)."""
        h, w, _ = img_shape.shape
        
        thumb_tip = landmarks[4]
        index_tip = landmarks[8]
        
        # Convert to pixel coordinates
        x1, y1 = int(thumb_tip.x * w), int(thumb_tip.y * h)
        x2, y2 = int(index_tip.x * w), int(index_tip.y * h)
        
        # Euclidean distance
        length = math.hypot(x2 - x1, y2 - y1)
        
        # Draw Line
        cv2.line(img_shape, (x1, y1), (x2, y2), COLOR_NEON_BLUE, 3)
        cv2.circle(img_shape, (x1, y1), 10, COLOR_NEON_BLUE, cv2.FILLED)
        cv2.circle(img_shape, (x2, y2), 10, COLOR_NEON_BLUE, cv2.FILLED)
        
        # Map length to volume
        vol = np.interp(length, [VOL_MIN_DIST, VOL_MAX_DIST], [0.0, 1.0])
        
        # EMA Smoothing
        smooth_vol = (VOL_SMOOTHING * vol) + ((1 - VOL_SMOOTHING) * self.prev_vol)
        self.prev_vol = smooth_vol
        
        return smooth_vol

    def _detect_swipe(self, current_wrist_x):
        """
        Detects significant horizontal movement.
        Returns: 'RIGHT', 'LEFT', or None
        """
        current_time = time.time()
        if current_time - self.last_swipe_time < SWIPE_COOLDOWN:
            return None
        
        self.wrist_history.append(current_wrist_x)
        if len(self.wrist_history) < SWIPE_HISTORY_LEN:
            return None
            
        # Calculate delta
        start_x = self.wrist_history[0]
        end_x = self.wrist_history[-1]
        delta = end_x - start_x
        
        if abs(delta) > SWIPE_THRESHOLD:
            self.last_swipe_time = current_time
            self.wrist_history.clear() # Reset
            if delta > 0: return "RIGHT" # Camera is mirrored, so Right on screen
            else: return "LEFT"
            
        return None

    def process_hands(self, results, img):
        """Main logic pipeline."""
        if not results.multi_hand_landmarks:
            return
            
        h, w, c = img.shape
        current_time = time.time()
        
        hands_list = results.multi_hand_landmarks
        num_hands = len(hands_list)
        
        # --- TWO HAND LOGIC ---
        if num_hands == 2:
            hand1_state = self._get_hand_state(hands_list[0].landmark)
            hand2_state = self._get_hand_state(hands_list[1].landmark)
            
            # Action: Stop (Two Open Palms)
            if hand1_state == self.STATE_OPEN and hand2_state == self.STATE_OPEN:
                if current_time - self.last_gesture_time > GESTURE_COOLDOWN:
                    self.player.stop()
                    self.last_gesture_time = current_time
                    cv2.putText(img, "STOPPED", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_NEON_RED, 3)
            
            # Action: Pause (Two Fists)
            elif hand1_state == self.STATE_FIST and hand2_state == self.STATE_FIST:
                 if current_time - self.last_gesture_time > GESTURE_COOLDOWN:
                    self.player.pause()
                    self.last_gesture_time = current_time
                    cv2.putText(img, "PAUSED", (w//2 - 100, h//2), cv2.FONT_HERSHEY_SIMPLEX, 2, COLOR_NEON_RED, 3)
            
            return # Exit to avoid conflicting single hand gestures

        # --- SINGLE HAND LOGIC ---
        hand_lms = hands_list[0]
        wrist = hand_lms.landmark[0]
        
        # 1. State Detection (Play/Pause)
        state = self._get_hand_state(hand_lms.landmark)
        
        if state == self.STATE_OPEN:
             if current_time - self.last_gesture_time > GESTURE_COOLDOWN:
                # Only resume if not already playing
                if not self.player.is_playing:
                    self.player.resume()
                    self.last_gesture_time = current_time
        
        elif state == self.STATE_FIST:
             if current_time - self.last_gesture_time > GESTURE_COOLDOWN:
                if self.player.is_playing:
                    self.player.pause()
                    self.last_gesture_time = current_time

        # 2. Volume Control (Pinch)
        # We calculate volume every frame if we are in 'OPEN' state or 'UNKNOWN' (pinching looks like unknown/fist depending on angle)
        # To be safe, we allow volume control always, but visual feedback helps.
        # Actually, let's allow volume only when NOT fully Fist to avoid conflict with Pause.
        if state != self.STATE_FIST:
            vol_level = self._process_volume(hand_lms.landmark, img)
            self.player.set_volume(vol_level)

        # 3. Swipe Detection
        wrist_x_pixel = int(wrist.x * w)
        swipe_dir = self._detect_swipe(wrist_x_pixel)
        
        if swipe_dir == "RIGHT": # Hand moves right -> Next Song
            self.player.next_track()
            cv2.putText(img, "NEXT TRACK", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_NEON_GREEN, 3)
        elif swipe_dir == "LEFT": # Hand moves left -> Prev Song
            self.player.prev_track()
            cv2.putText(img, "PREV TRACK", (w//2 - 150, h//2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, COLOR_NEON_GREEN, 3)

# ==========================================
# MAIN APPLICATION
# ==========================================

class App:
    def __init__(self):
        # Init Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, CAM_WIDTH)
        self.cap.set(4, CAM_HEIGHT)
        
        # Init Hand Tracking
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=MAX_HANDS,
            min_detection_confidence=DETECTION_CONF,
            min_tracking_confidence=TRACKING_CONF
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Init Modules
        self.player = MusicPlayer(SONG_DIR)
        self.controller = GestureController(self.player)
        self.vfx = VisualEffects()
        
    def draw_ui(self, img):
        """Draws the Overlay UI on the frame."""
        h, w, c = img.shape
        
        # Background panel for text
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, 80), (0, 0, 0), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)
        
        # Song Title
        cv2.putText(img, f"Song: {self.player.get_current_song_name()}", (20, 40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLOR_WHITE, 2)
        
        # Status
        status_text = "PLAYING" if self.player.is_playing else "PAUSED"
        color = COLOR_NEON_GREEN if self.player.is_playing else COLOR_NEON_RED
        cv2.putText(img, f"Status: {status_text}", (20, 70), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Volume Bar (Vertical)
        vol_height = int(self.player.volume * 200)
        bar_x = w - 50
        bar_y_bottom = h - 50
        bar_y_top = bar_y_bottom - 200
        
        # Draw background bar
        cv2.rectangle(img, (bar_x, bar_y_top), (bar_x + 20, bar_y_bottom), (50, 50, 50), -1)
        # Draw active volume
        cv2.rectangle(img, (bar_x, bar_y_bottom - vol_height), (bar_x + 20, bar_y_bottom), COLOR_NEON_BLUE, -1)
        cv2.rectangle(img, (bar_x, bar_y_top), (bar_x + 20, bar_y_bottom), COLOR_WHITE, 2)
        cv2.putText(img, f"{int(self.player.volume * 100)}%", (bar_x - 10, bar_y_top - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_WHITE, 2)

    def run(self):
        print("[INFO] Application Started. Press 'q' to exit.")
        
        while True:
            success, img = self.cap.read()
            if not success:
                print("[ERROR] Camera not found.")
                break
                
            # Flip image for mirror effect
            img = cv2.flip(img, 1)
            
            # Convert to RGB for MediaPipe
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = self.hands.process(img_rgb)
            
            # Draw VFX and Landmarks
            if results.multi_hand_landmarks:
                for hand_lms in results.multi_hand_landmarks:
                    # Draw Standard Landmarks
                    self.mp_draw.draw_landmarks(img, hand_lms, self.mp_hands.HAND_CONNECTIONS)
                    
                    # Calculate Center for VFX (approximate using Wrist(0) and Middle Finger MCP(9))
                    h, w, _ = img.shape
                    cx = int((hand_lms.landmark[0].x + hand_lms.landmark[9].x) / 2 * w)
                    cy = int((hand_lms.landmark[0].y + hand_lms.landmark[9].y) / 2 * h)
                    
                    # Draw Doctor Strange Effect
                    self.vfx.draw_magic_circle(img, (cx, cy), is_active=self.player.is_playing)
            
            # Process Gestures
            self.controller.process_hands(results, img)
            
            # Draw UI
            self.draw_ui(img)
            
            # Display
            cv2.imshow("Gesture Music Player", img)
            
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        pygame.quit()

if __name__ == "__main__":
    app = App()
    app.run()