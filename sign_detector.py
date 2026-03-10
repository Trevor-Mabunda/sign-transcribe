"""
Sign Detection Module - Real-time Hand Gesture Tracking with MediaPipe
"""
import cv2
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.core.base_options import BaseOptions
from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
import numpy as np
from typing import Dict, List, Tuple, Optional
import time
import base64
from io import BytesIO
from PIL import Image as PILImage
import os

class SignDetector:
    """
    Sign Language Detector using MediaPipe Hand Tracking
    Real-time hand landmark detection and gesture recognition
    """
    
    def __init__(self, model_path: str = None, confidence_threshold: float = 0.5):
        """
        Initialize sign detector with MediaPipe Hand Landmarker
        
        Args:
            model_path: Optional path to custom model
            confidence_threshold: Minimum confidence for hand detection (0.0-1.0)
        """
        self.model_path = model_path
        self.confidence_threshold = confidence_threshold
        
        # Get path to hand landmarker model
        model_file = self._get_hand_model_path()
        
        # Initialize MediaPipe Hand Landmarker
        base_options = BaseOptions(model_asset_path=model_file)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.IMAGE,
            num_hands=2,
            min_hand_detection_confidence=confidence_threshold,
            min_hand_presence_confidence=0.5
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(options)
        
        self.is_model_loaded = True
        self.last_gesture = None
        self.gesture_history = []
        self.gesture_smoothing_frames = 3
    
    def _get_hand_model_path(self) -> str:
        """Get path to hand landmarker model file"""
        # First try the local models directory
        local_model = os.path.join(
            os.path.dirname(__file__),
            'models',
            'hand_landmarker.task'
        )
        if os.path.exists(local_model):
            return local_model
        
        # Fallback to home directory
        home_model = os.path.expanduser('~/.mediapipe/hand_landmarker.task')
        if os.path.exists(home_model):
            return home_model
        
        # If no model found, raise error
        raise FileNotFoundError(
            "Hand landmarker model not found. "
            "Please download it using: mkdir -p models && "
            "curl -L 'https://storage.googleapis.com/mediapipe-assets/hand_landmarker.task' "
            "-o models/hand_landmarker.task"
        )
        
    def frame_to_cv2(self, frame_data) -> Optional[np.ndarray]:
        """
        Convert various frame formats to OpenCV format
        
        Args:
            frame_data: Frame as bytes, PIL Image, or numpy array
            
        Returns:
            OpenCV format frame or None
        """
        try:
            if isinstance(frame_data, bytes):
                pil_image = PILImage.open(BytesIO(frame_data))
                return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            elif isinstance(frame_data, PILImage.Image):
                return cv2.cvtColor(np.array(frame_data), cv2.COLOR_RGB2BGR)
            elif isinstance(frame_data, np.ndarray):
                if len(frame_data.shape) == 3 and frame_data.shape[2] == 3:
                    return frame_data
            return None
        except Exception as e:
            print(f"Error converting frame: {e}")
            return None
    
    def detect_hand_landmarks(self, frame) -> Dict:
        """
        Detect hand landmarks in a video frame using MediaPipe
        
        Args:
            frame: Video frame from webcam (numpy array, PIL Image, or bytes)
            
        Returns:
            Dict with hand detection results
        """
        result = {
            'hands_detected': 0,
            'hand_landmarks': [],
            'hand_positions': [],
            'gestures': [],
            'confidence': 0.0,
            'has_hands': False,
            'hand_keypoints': []
        }
        
        try:
            # Convert frame if necessary
            cv_frame = frame if isinstance(frame, np.ndarray) else self.frame_to_cv2(frame)
            if cv_frame is None:
                return result
            
            # Convert to RGB for MediaPipe
            frame_rgb = cv2.cvtColor(cv_frame, cv2.COLOR_BGR2RGB)
            
            # Create MediaPipe Image
            mp_image = Image(image_format=ImageFormat.SRGB, data=frame_rgb)
            
            # Run hand detection
            hand_results = self.hand_detector.detect(mp_image)
            
            if hand_results.hand_landmarks:
                result['hands_detected'] = len(hand_results.hand_landmarks)
                result['has_hands'] = True
                
                h, w, c = cv_frame.shape
                avg_confidence = 0
                
                for i, hand_landmarks in enumerate(hand_results.hand_landmarks):
                    # Extract landmarks
                    landmarks = []
                    keypoints = []
                    
                    for landmark in hand_landmarks:
                        landmarks.append({
                            'x': landmark.x,
                            'y': landmark.y,
                            'z': landmark.z,
                            'visibility': getattr(landmark, 'visibility', 1.0)
                        })
                        keypoints.append([landmark.x * w, landmark.y * h])
                    
                    result['hand_landmarks'].append(landmarks)
                    result['hand_keypoints'].append(keypoints)
                    
                    # Get handedness (left/right)
                    if hand_results.handedness and i < len(hand_results.handedness):
                        handedness = hand_results.handedness[i][0]  # Get first classification
                        result['hand_positions'].append({
                            'label': handedness.category_name,
                            'confidence': handedness.score
                        })
                        avg_confidence += handedness.score
                    else:
                        result['hand_positions'].append({
                            'label': 'Unknown',
                            'confidence': 0.0
                        })
                    
                    # Classify gesture
                    gesture = self._classify_gesture(landmarks)
                    result['gestures'].append(gesture)
                
                if result['hand_positions']:
                    result['confidence'] = avg_confidence / len(result['hand_positions'])
            
        except Exception as e:
            print(f"Error detecting hand landmarks: {e}")
            import traceback
            traceback.print_exc()
        
        return result
    
    def detect_signs(self, frame) -> Dict:
        """
        Detect sign language gestures in a video frame
        Only returns results when hands are actually detected
        
        Args:
            frame: Video frame from webcam
            
        Returns:
            Dict with detected signs, hands, and confidence
        """
        hand_detection = self.detect_hand_landmarks(frame)
        
        result = {
            'detected_sign': None,
            'confidence': hand_detection['confidence'],
            'landmarks': hand_detection['hand_landmarks'],
            'hand_positions': hand_detection['hand_positions'],
            'gestures': hand_detection['gestures'],
            'hands_detected': hand_detection['hands_detected'],
            'has_hands': hand_detection['has_hands'],
            'timestamp': time.time()
        }
        
        # Only transcribe if hands are actually detected
        if hand_detection['has_hands'] and hand_detection['gestures']:
            # Use the first detected gesture
            result['detected_sign'] = hand_detection['gestures'][0]
        
        return result
    
    def _classify_gesture(self, landmarks: List[Dict]) -> str:
        """
        Classify ASL hand gesture from landmarks
        Recognizes common American Sign Language signs
        
        Args:
            landmarks: List of 21 hand landmarks with x, y, z coordinates
            
        Returns:
            ASL sign name
        """
        if not landmarks or len(landmarks) < 21:
            return 'unknown'
        
        try:
            # Extract hand shape and position features
            hand_shape = self._get_hand_shape(landmarks)
            hand_position = self._get_hand_position(landmarks)
            palm_facing = self._get_palm_orientation(landmarks)
            hand_movement = self._get_hand_movement_pattern(landmarks)
            
            # Match to ASL signs based on feature combination
            asl_sign = self._match_asl_sign(
                hand_shape=hand_shape,
                position=hand_position,
                palm_facing=palm_facing,
                movement=hand_movement,
                landmarks=landmarks
            )
            
            return asl_sign
            
        except Exception as e:
            print(f"Error classifying gesture: {e}")
            return 'unknown'
    
    def _get_hand_shape(self, landmarks: List[Dict]) -> Dict:
        """
        Determine hand shape based on finger positions
        Returns dict with finger states
        """
        shape = {
            'index_up': False,
            'middle_up': False,
            'ring_up': False,
            'pinky_up': False,
            'thumb_extended': False,
            'fingers_spread': False,
            'fingers_touching': False,
            'closed_fist': False
        }
        
        try:
            # Check each finger
            shape['index_up'] = self._is_finger_up(landmarks, 1)
            shape['middle_up'] = self._is_finger_up(landmarks, 2)
            shape['ring_up'] = self._is_finger_up(landmarks, 3)
            shape['pinky_up'] = self._is_finger_up(landmarks, 4)
            
            # Check thumb
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            thumb_mcp = landmarks[2]
            shape['thumb_extended'] = thumb_tip['x'] < thumb_ip['x'] - 0.03
            
            # Check if fingers are spread
            index_pos = landmarks[9]
            middle_pos = landmarks[13]
            ring_pos = landmarks[17]
            pinky_pos = landmarks[21] if len(landmarks) > 20 else landmarks[20]
            
            spread_dist = [
                abs(index_pos['x'] - middle_pos['x']),
                abs(middle_pos['x'] - ring_pos['x']),
                abs(ring_pos['x'] - pinky_pos['x'])
            ]
            shape['fingers_spread'] = sum(spread_dist) > 0.15
            
            # Check if fingers are touching
            fingers_up_count = sum([
                shape['index_up'],
                shape['middle_up'],
                shape['ring_up'],
                shape['pinky_up']
            ])
            
            if fingers_up_count == 0:
                shape['closed_fist'] = True
            
            # Check if fingers are held close together
            if not shape['fingers_spread'] and fingers_up_count > 0:
                shape['fingers_touching'] = True
                
        except Exception as e:
            print(f"Error getting hand shape: {e}")
        
        return shape
    
    def _get_hand_position(self, landmarks: List[Dict]) -> Dict:
        """
        Determine hand position relative to face and body
        """
        position = {
            'height': 'middle',  # top, middle, bottom
            'side': 'center',    # left, center, right
            'distance': 'normal' # close, normal, far
        }
        
        try:
            palm_base = landmarks[0]
            
            # Determine height (0=top, 1=bottom)
            if palm_base['y'] < 0.33:
                position['height'] = 'top'
            elif palm_base['y'] > 0.66:
                position['height'] = 'bottom'
            else:
                position['height'] = 'middle'
            
            # Determine side (0=left, 1=right)
            if palm_base['x'] < 0.33:
                position['side'] = 'left'
            elif palm_base['x'] > 0.66:
                position['side'] = 'right'
            else:
                position['side'] = 'center'
            
            # Determine distance (z-depth)
            if palm_base['z'] < -0.1:
                position['distance'] = 'far'
            elif palm_base['z'] > 0.1:
                position['distance'] = 'close'
            else:
                position['distance'] = 'normal'
                
        except Exception as e:
            print(f"Error getting hand position: {e}")
        
        return position
    
    def _get_palm_orientation(self, landmarks: List[Dict]) -> str:
        """
        Determine which direction palm is facing
        """
        try:
            # Use hand landmarks to determine palm normal direction
            palm_base = landmarks[0]
            middle_finger_mcp = landmarks[9]
            
            # Vector from palm to middle finger
            vector_x = middle_finger_mcp['x'] - palm_base['x']
            vector_y = middle_finger_mcp['y'] - palm_base['y']
            vector_z = middle_finger_mcp['z'] - palm_base['z']
            
            # Determine primary orientation
            if abs(vector_z) > abs(vector_x) and abs(vector_z) > abs(vector_y):
                if vector_z > 0:
                    return 'away'  # Palm facing away
                else:
                    return 'toward'  # Palm facing toward
            elif abs(vector_y) > abs(vector_x):
                if vector_y > 0:
                    return 'down'
                else:
                    return 'up'
            else:
                if vector_x > 0:
                    return 'right'
                else:
                    return 'left'
                    
        except Exception as e:
            print(f"Error getting palm orientation: {e}")
            return 'unknown'
    
    def _get_hand_movement_pattern(self, landmarks: List[Dict]) -> str:
        """
        Analyze hand movement pattern
        This would ideally track landmarks over multiple frames
        """
        # This is simplified for single frame; could be enhanced with frame history
        return 'static'
    
    def _match_asl_sign(self, hand_shape: Dict, position: Dict, 
                       palm_facing: str, movement: str, landmarks: List[Dict]) -> str:
        """
        Match detected hand pose to ASL sign
        Based on hand shape, position, and orientation
        Checks custom trained gestures first, then built-in ASL signs
        """
        
        # First, try to match against custom trained gestures
        custom_match = self.recognize_custom_gesture(
            hand_shape, position, palm_facing, landmarks
        )
        if custom_match:
            return custom_match.upper()
        
        fingers_up = sum([
            hand_shape['index_up'],
            hand_shape['middle_up'],
            hand_shape['ring_up'],
            hand_shape['pinky_up']
        ])
        
        # ASL SIGN CLASSIFICATION LOGIC
        # ==============================
        
        # HELLO/WAVE - Open hand at side, moving sideways
        if (fingers_up == 5 and hand_shape['fingers_spread'] and 
            position['side'] in ['right', 'left'] and position['height'] == 'top'):
            return 'HELLO'
        
        # THANK YOU - Open hand, palm up, touching chin/mouth
        if (fingers_up == 5 and position['height'] in ['middle', 'top'] and
            palm_facing in ['up', 'toward']):
            if position['distance'] == 'close':
                return 'THANK_YOU'
        
        # LIKE/LOVE - Thumb and middle finger pinched, hand at chest
        if (hand_shape['thumb_extended'] and hand_shape['middle_up'] and
            not hand_shape['index_up'] and not hand_shape['ring_up'] and
            not hand_shape['pinky_up'] and position['distance'] == 'close'):
            return 'LIKE'
        
        # YES - Fist with thumb up, nodding motion
        if (hand_shape['closed_fist'] and hand_shape['thumb_extended'] and
            position['height'] in ['middle', 'top']):
            return 'YES'
        
        # NO - Index and middle finger pointing up side to side
        if (hand_shape['index_up'] and hand_shape['middle_up'] and
            not hand_shape['ring_up'] and not hand_shape['pinky_up'] and
            not hand_shape['thumb_extended']):
            return 'NO'
        
        # STOP - Open hand, palm facing forward/away
        if (fingers_up == 5 and hand_shape['fingers_spread'] and
            palm_facing in ['away', 'toward'] and position['height'] == 'middle'):
            return 'STOP'
        
        # PLEASE - Open hand, palm on chest, circular motion
        if (fingers_up == 5 and position['distance'] == 'close' and
            position['height'] in ['middle', 'bottom'] and palm_facing == 'toward'):
            return 'PLEASE'
        
        # SORRY - Fist, hand at chest with circular motion
        if (hand_shape['closed_fist'] and position['distance'] == 'close' and
            position['height'] in ['middle', 'top']):
            return 'SORRY'
        
        # HELP - Open hand over fist, supporting motion
        if (fingers_up == 5 and hand_shape['fingers_spread'] and
            position['height'] in ['middle', 'bottom']):
            return 'HELP'
        
        # GOOD - Open hand at chin, moving down
        if (fingers_up == 5 and position['height'] in ['top', 'middle'] and
            palm_facing in ['away', 'down'] and position['distance'] == 'close'):
            return 'GOOD'
        
        # BAD - Open hand at chin, twisting motion (opposite of good)
        if (fingers_up == 5 and position['height'] in ['top', 'middle'] and
            palm_facing in ['toward', 'up'] and position['distance'] == 'close'):
            return 'BAD'
        
        # SIGN - Index and middle finger pointing up, circular outward motion
        if (hand_shape['index_up'] and hand_shape['middle_up'] and
            not hand_shape['ring_up'] and not hand_shape['pinky_up'] and
            position['side'] in ['center', 'left']):
            return 'SIGN'
        
        # UNDERSTAND/KNOW - Index finger pointing at temple
        if (hand_shape['index_up'] and not hand_shape['middle_up'] and
            not hand_shape['ring_up'] and not hand_shape['pinky_up'] and
            position['height'] == 'top'):
            return 'UNDERSTAND'
        
        # THINK - Index finger pointing at head
        if (hand_shape['index_up'] and position['height'] == 'top' and
            position['distance'] == 'close'):
            return 'THINK'
        
        # YOU - Index finger pointing outward
        if (hand_shape['index_up'] and not hand_shape['middle_up'] and
            position['side'] in ['center', 'right'] and palm_facing in ['away', 'right']):
            return 'YOU'
        
        # ME/I - Index finger pointing to self
        if (hand_shape['index_up'] and not hand_shape['middle_up'] and
            position['distance'] == 'close' and position['height'] in ['middle', 'bottom']):
            return 'ME'
        
        # WHAT/QUESTION - Index and thumb separated, wiggling motion
        if (hand_shape['index_up'] and hand_shape['thumb_extended'] and
            not hand_shape['middle_up'] and position['height'] == 'middle'):
            return 'WHAT'
        
        # WHERE - Index finger wiggling while pointing
        if (hand_shape['index_up'] and position['height'] in ['middle', 'top'] and
            palm_facing == 'away'):
            return 'WHERE'
        
        # CONTINUE/GO - Both hands open, moving forward
        if (fingers_up == 5 and hand_shape['fingers_spread'] and
            palm_facing in ['away', 'forward']):
            return 'GO'
        
        # COME - Open hand with fingers pointing inward, pulling motion
        if (fingers_up == 5 and palm_facing in ['toward', 'left', 'right']):
            return 'COME'
        
        # SMILE/HAPPY - Tracing smile with index fingers at mouth
        if (hand_shape['index_up'] and not hand_shape['middle_up'] and
            position['height'] == 'top' and position['distance'] == 'close'):
            return 'HAPPY'
        
        # DEFAULT: Return detected hand shape
        if fingers_up == 0:
            return 'FIST'
        elif fingers_up == 1 and hand_shape['index_up']:
            return 'POINTING'
        elif fingers_up == 2 and hand_shape['index_up'] and hand_shape['middle_up']:
            return 'PEACE'
        elif fingers_up == 5:
            return 'OPEN_HAND'
        else:
            return f'GESTURE_{fingers_up}_FINGERS'
    
    def _distance(self, point1: Dict, point2: Dict) -> float:
        """Calculate Euclidean distance between two points"""
        dx = point1['x'] - point2['x']
        dy = point1['y'] - point2['y']
        dz = point1['z'] - point2['z']
        return np.sqrt(dx*dx + dy*dy + dz*dz)
    
    def _is_finger_up(self, landmarks: List[Dict], finger_index: int) -> bool:
        """Check if a finger is extended up"""
        tip = landmarks[finger_index * 4 + 4]
        pip = landmarks[finger_index * 4 + 2]
        return tip['y'] < pip['y']
    
    def _count_fingers_up(self, landmarks: List[Dict]) -> int:
        """
        Count how many fingers are extended up
        Improved for ASL recognition
        """
        if not landmarks or len(landmarks) < 21:
            return 0
        
        count = 0
        try:
            # Check each finger (index, middle, ring, pinky)
            # Fingers 1-4, tips are at indices 8, 12, 16, 20
            for finger_idx in range(1, 5):
                if self._is_finger_up(landmarks, finger_idx):
                    count += 1
            
            # Check thumb separately - it has different orientation
            if self._is_thumb_extended(landmarks):
                count += 1
            
        except Exception as e:
            print(f"Error counting fingers up: {e}")
        
        return count
    
    def _is_thumb_extended(self, landmarks: List[Dict]) -> bool:
        """
        Check if thumb is extended
        Thumb is extended when tip is away from palm more than usual
        """
        try:
            thumb_tip = landmarks[4]
            thumb_ip = landmarks[3]
            thumb_mcp = landmarks[2]
            palm = landmarks[0]
            
            # Thumb is extended if tip is further from palm than MCP
            thumb_tip_dist = self._distance(thumb_tip, palm)
            thumb_mcp_dist = self._distance(thumb_mcp, palm)
            
            return thumb_tip_dist > thumb_mcp_dist + 0.02
        except:
            return False
    
    def transcribe_signs(self, detections: List[Dict]) -> str:
        """
        Convert sequence of detected signs into readable text
        Only processes frames where hands were detected
        
        Args:
            detections: List of detected signs with confidences
            
        Returns:
            Transcribed text
        """
        # Filter only detections with hands
        hand_detections = [d for d in detections if d.get('hands_detected', 0) > 0]
        
        if not hand_detections:
            return "No hand gestures detected"
        
        # Filter by confidence threshold
        high_confidence = [d for d in hand_detections if d['confidence'] > self.confidence_threshold]
        
        if not high_confidence:
            return "Gestures detected but confidence too low"
        
        # Convert signs to text
        signs = []
        for detection in high_confidence:
            if detection.get('detected_sign'):
                signs.append(detection['detected_sign'])
        
        if not signs:
            return "No clear gestures detected"
        
        # Remove duplicates but maintain order
        unique_signs = []
        for sign in signs:
            if sign not in unique_signs:
                unique_signs.append(sign)
        
        return ' '.join(unique_signs)
    
    def get_hand_detection_status(self, frame) -> Dict:
        """
        Get real-time hand detection status for UI feedback
        
        Args:
            frame: Current video frame
            
        Returns:
            Dict with detection status and hand count
        """
        detection = self.detect_signs(frame)
        return {
            'hands_detected': detection['hands_detected'],
            'has_hands': detection['has_hands'],
            'gestures': detection['gestures'],
            'confidence': detection['confidence']
        }
    
    def train_gesture(self, gesture_name: str, landmarks: List[Dict], 
                     hand_shape: Dict = None, position: Dict = None) -> None:
        """
        Train the detector to recognize a new ASL gesture
        Stores the gesture pattern for recognition
        
        Args:
            gesture_name: Name of the ASL sign to train
            landmarks: Hand landmarks from training sample
            hand_shape: Pre-calculated hand shape features
            position: Pre-calculated hand position features
        """
        try:
            # Initialize training data storage if needed
            if not hasattr(self, 'custom_gestures'):
                self.custom_gestures = {}
            
            # Calculate features if not provided
            if hand_shape is None:
                hand_shape = self._get_hand_shape(landmarks)
            if position is None:
                position = self._get_hand_position(landmarks)
            
            palm_facing = self._get_palm_orientation(landmarks)
            
            # Store gesture template
            gesture_template = {
                'name': gesture_name,
                'hand_shape': hand_shape,
                'position': position,
                'palm_facing': palm_facing,
                'landmark_template': landmarks,
                'trained_at': time.time()
            }
            
            # Store multiple training samples to improve recognition
            if gesture_name not in self.custom_gestures:
                self.custom_gestures[gesture_name] = []
            
            self.custom_gestures[gesture_name].append(gesture_template)
            
            print(f"Trained gesture: {gesture_name} ({len(self.custom_gestures[gesture_name])} samples)")
            
        except Exception as e:
            print(f"Error training gesture: {e}")
    
    def recognize_custom_gesture(self, hand_shape: Dict, position: Dict, 
                                palm_facing: str, landmarks: List[Dict]) -> Optional[str]:
        """
        Recognize a custom trained gesture
        
        Args:
            hand_shape: Detected hand shape features
            position: Detected hand position
            palm_facing: Detected palm orientation
            landmarks: Hand landmarks
            
        Returns:
            Recognized gesture name if match found, None otherwise
        """
        if not hasattr(self, 'custom_gestures') or not self.custom_gestures:
            return None
        
        try:
            best_match = None
            best_score = 0.0
            threshold = 0.6  # Minimum confidence to recognize
            
            # Compare against all trained gestures
            for gesture_name, templates in self.custom_gestures.items():
                for template in templates:
                    # Calculate similarity to template
                    score = self._calculate_gesture_similarity(
                        hand_shape, template['hand_shape'],
                        position, template['position'],
                        palm_facing, template['palm_facing'],
                        landmarks, template['landmark_template']
                    )
                    
                    if score > best_score:
                        best_score = score
                        best_match = gesture_name
            
            # Return match if confidence is high enough
            if best_score > threshold:
                return best_match
            
            return None
            
        except Exception as e:
            print(f"Error recognizing custom gesture: {e}")
            return None
    
    def _calculate_gesture_similarity(self, shape1: Dict, shape2: Dict,
                                     pos1: Dict, pos2: Dict,
                                     palm1: str, palm2: str,
                                     lm1: List[Dict], lm2: List[Dict]) -> float:
        """
        Calculate similarity between two gestures
        Returns score between 0.0 and 1.0
        """
        score = 0.0
        weight_sum = 0
        
        try:
            # Compare hand shape (weight: 0.4)
            shape_match = sum(1 for key in shape1 if shape1[key] == shape2.get(key))
            shape_score = shape_match / len(shape1) if len(shape1) > 0 else 0
            score += shape_score * 0.4
            weight_sum += 0.4
            
            # Compare position (weight: 0.3)
            pos_match = 0
            if pos1.get('height') == pos2.get('height'):
                pos_match += 1
            if pos1.get('side') == pos2.get('side'):
                pos_match += 1
            pos_score = pos_match / 2.0
            score += pos_score * 0.3
            weight_sum += 0.3
            
            # Compare palm facing (weight: 0.2)
            palm_score = 1.0 if palm1 == palm2 else 0.5
            score += palm_score * 0.2
            weight_sum += 0.2
            
            # Compare landmark positions (weight: 0.1)
            if len(lm1) == len(lm2):
                landmark_diffs = []
                for i in range(len(lm1)):
                    dist = self._distance(lm1[i], lm2[i])
                    landmark_diffs.append(dist)
                
                avg_diff = np.mean(landmark_diffs)
                # If average difference is small (< 0.1), high similarity
                landmark_score = max(0, 1.0 - (avg_diff / 0.1))
                score += landmark_score * 0.1
                weight_sum += 0.1
            
            # Normalize score
            if weight_sum > 0:
                score = score / weight_sum
            
            return min(1.0, max(0.0, score))
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return 0.0
    
    def get_trained_gestures(self) -> List[str]:
        """
        Get list of trained custom gestures
        
        Returns:
            List of custom gesture names
        """
        if not hasattr(self, 'custom_gestures'):
            return []
        
        return list(self.custom_gestures.keys())
    
    def get_gesture_samples(self, gesture_name: str) -> int:
        """
        Get number of training samples for a gesture
        """
        if not hasattr(self, 'custom_gestures'):
            return 0
        
        return len(self.custom_gestures.get(gesture_name, []))
    
    def clear_gesture_training(self) -> None:
        """
        Clear all custom gesture training data
        """
        self.custom_gestures = {}
        print("Cleared all trained gestures")

    def process_video_stream(self, video_source: str = 0) -> List[Dict]:
        """
        Process video stream from webcam or file
        Only yields frames with detected hands
        
        Args:
            video_source: 0 for webcam or path to video file
            
        Yields:
            Detected signs with timestamps (only when hands detected)
        """
        try:
            cap = cv2.VideoCapture(video_source)
            
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                detection = self.detect_signs(frame)
                
                # Only yield when hands are detected
                if detection['has_hands']:
                    yield detection
            
            cap.release()
            
        except Exception as e:
            print(f"Error processing video: {e}")
            raise


def initialize_detector(use_mock: bool = False) -> SignDetector:
    """
    Initialize sign detector with real MediaPipe hand tracking
    
    Args:
        use_mock: Legacy parameter (ignored - always uses real detector now)
        
    Returns:
        SignDetector instance with MediaPipe hand tracking
    """
    # Always use real hand tracking with MediaPipe
    return SignDetector(confidence_threshold=0.7)
