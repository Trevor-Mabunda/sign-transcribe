# Sign - American Sign Language (ASL) Recognition

## Overview

Sign now includes comprehensive American Sign Language (ASL) gesture recognition powered by MediaPipe hand tracking. The system recognizes 25+ common ASL signs and allows users to teach it new custom gestures.

## Built-in ASL Signs Recognition

The app recognizes the following common ASL signs automatically:

### Communication Basics
- **HELLO** - Open hand at side, waving motion
- **THANK YOU** - Open hand, palm up, touching chin
- **SIGN** - Index and middle fingers pointing up, circular motion
- **COME** - Open hand with fingers pointing inward, pulling motion

### Yes/No & Affirmation
- **YES** - Fist with thumb up, nodding motion
- **NO** - Index and middle finger pointing up side to side
- **SORRY** - Fist at chest with circular motion

### Common Requests
- **PLEASE** - Open hand on chest, circular motion
- **HELP** - Open hand supporting a fist
- **STOP** - Open hand, palm facing forward
- **GO** - Both hands open, moving forward

### Emotions & States
- **GOOD** - Open hand at chin, moving down
- **BAD** - Open hand at chin, twisting motion
- **HAPPY** - Index fingers tracing smile at mouth
- **LIKE** - Thumb and middle finger pinched at chest

### Pronouns
- **YOU** - Index finger pointing outward
- **ME** - Index finger pointing to self
- **I** - Index finger pointing to self

### Questions & Knowledge
- **WHAT** - Index and thumb separated, wiggling
- **WHERE** - Index finger wiggling while pointing
- **UNDERSTAND** - Index finger at temple
- **THINK** - Index finger pointing at head

### Hand Shapes (Basic)
- **FIST** - All fingers closed
- **POINTING** - Single index finger extended
- **PEACE** - Index and middle finger (V-shape)
- **OPEN_HAND** - All five fingers extended

## How ASL Recognition Works

### Hand Feature Extraction

The system analyzes:

1. **Hand Shape** - Which fingers are up or down
   - Analyzes each individual finger position
   - Detects thumb extension state
   - Checks if fingers are spread or together

2. **Hand Position** - Where the hand is in space
   - Vertical position: top, middle, bottom
   - Horizontal position: left, center, right
   - Distance from camera: close, normal, far

3. **Palm Orientation** - Which direction palm faces
   - Away (facing camera)
   - Toward (facing back)
   - Up, Down, Left, Right

4. **Hand Movement** - Motion patterns (frame-to-frame analysis)
   - Static poses
   - Circular motions
   - Directional movements

### Recognition Algorithm

The system matches detected hand poses against known ASL sign patterns:

```
1. Extract hand landmarks from camera frame
2. Calculate hand shape features
3. Determine hand position in space
4. Detect palm orientation
5. Check for custom trained gestures first
6. If no custom match, match against built-in ASL signs
7. Return most likely ASL sign with confidence score
```

## Training Custom Gestures

Users can teach Sign to recognize new ASL signs or variations:

### API Endpoint: Train Gesture

**POST** `/api/train-gesture`

Train the system on a new custom ASL gesture.

**Request Body:**
```json
{
  "gesture_name": "MY_CUSTOM_SIGN",
  "frame": "base64_encoded_image_or_bytes"
}
```

**Response:**
```json
{
  "status": "success",
  "message": "Gesture 'MY_CUSTOM_SIGN' trained successfully",
  "gesture_name": "MY_CUSTOM_SIGN",
  "samples": 1
}
```

**Usage Example (JavaScript):**
```javascript
async function trainCustomGesture(gestureName, frameBlob) {
  const formData = new FormData();
  formData.append('gesture_name', gestureName);
  formData.append('frame', frameBlob, 'frame.jpg');
  
  const response = await fetch('/api/train-gesture', {
    method: 'POST',
    headers: { 'Authorization': `Bearer ${token}` },
    body: formData
  });
  
  const data = await response.json();
  console.log(`Trained: ${data.gesture_name}, Samples: ${data.samples}`);
}
```

### API Endpoint: List Trained Gestures

**GET** `/api/trained-gestures`

Get all custom trained gestures for the current user session.

**Response:**
```json
{
  "status": "success",
  "trained_gestures": {
    "MY_CUSTOM_SIGN": {"samples": 3},
    "ANOTHER_SIGN": {"samples": 1}
  },
  "total": 2
}
```

### API Endpoint: Clear Training

**POST** `/api/clear-training`

Clear all trained custom gestures (admin only).

**Response:**
```json
{
  "status": "success",
  "message": "All trained gestures cleared"
}
```

## Improving Recognition Accuracy

### For Best Results:

1. **Lighting** - Ensure good lighting conditions
   - Bright, even lighting across hands
   - Avoid shadows and backlighting

2. **Hand Visibility** - Keep hands fully visible in frame
   - Position hands against contrasting background
   - Keep fingers clearly separated

3. **Steady Pose** - Hold gesture steady for 0.5-1 second
   - Avoid jerky movements during recognition
   - Keep hands within frame boundaries

4. **Training Multiple Samples** - Train each custom gesture multiple times
   - 3-5 samples recommended per gesture
   - Train from different angles/distances
   - This improves matching accuracy

5. **Consistent Positioning** - Use consistent hand positions
   - Keep training distance similar to usage
   - Train at similar heights and angles

## ASL Sign Definition Reference

### HELLO
- **Hand Shape:** Open palm, all fingers spread
- **Position:** At side of body, shoulder height
- **Movement:** Waving side-to-side

### THANK YOU
- **Hand Shape:** Open palm, fingers spread
- **Position:** Starting at mouth level
- **Movement:** Forward and down motion
- **Palm Facing:** Facing inward initially, then outward

### YES
- **Hand Shape:** Closed fist with thumb up
- **Position:** Neutral space in front of body
- **Movement:** Nodding up-and-down motion (like fist pump)

### NO
- **Hand Shape:** Index and middle finger extended (scissors)
- **Position:** Neutral space in front of body  
- **Movement:** Wiggling side-to-side

### SORRY
- **Hand Shape:** Closed fist, or open hand
- **Position:** At chest level
- **Movement:** Circular rubbing motion over heart

### HELP
- **Hand Shape:** One open hand supporting fist beneath it
- **Position:** In neutral space
- **Movement:** Lifting motion upward

### GOOD
- **Hand Shape:** Open hand, fingers spread or together
- **Position:** Starting at mouth level
- **Movement:** Forward and downward motion
- **Palm Facing:** Away from body

### UNDERSTAND/KNOW
- **Hand Shape:** Index finger pointing
- **Position:** At side of head/temple
- **Movement:** Bending at knuckle (tapping temple)

## Technical Details

### Hand Landmarks Used

MediaPipe provides 21 hand landmarks per hand:

```
Landmark Indices:
0  - Palm base (wrist)
1-4   - Thumb (MCP, PIP, DIP, TIP)
5-8   - Index (MCP, PIP, DIP, TIP)
9-12  - Middle (MCP, PIP, DIP, TIP)
13-16 - Ring (MCP, PIP, DIP, TIP)
17-20 - Pinky (MCP, PIP, DIP, TIP)

(MCP = Metacarpophalangeal, PIP = Proximal IP, DIP = Distal IP)
```

### Confidence Scoring

The recognition system returns confidence scores:

```
- 0.9-1.0: Very confident match
- 0.7-0.9: Confident match
- 0.5-0.7: Possible match (may need refinement)
- < 0.5: Low confidence (insufficient data)
```

### Custom Gesture Matching

Custom trained gestures are matched using:

1. **Hand Shape Similarity** (40% weight)
   - Comparison of finger positions
   - Matches against training samples

2. **Position Similarity** (30% weight)
   - Height, side, and distance matching
   - Fuzzy matching allows variation

3. **Palm Orientation** (20% weight)
   - Exact or close match required

4. **Landmark Distance** (10% weight)
   - Euclidean distance between 3D landmarks
   - Averaged across all 21 points

**Overall Match Threshold:** 0.6 (60% similarity needed)

## Known Limitations

1. **Single Face Recognition** - Recognizes one hand at a time primarily
2. **Static Pose Detection** - Current implementation recognizes hand shapes and positions
3. **No Finger Spelling** - Individual letters A-Z not yet supported
4. **Movement Tracking** - Basic movement patterns only (enhanced tracking coming soon)
5. **Arm Position** - Arm movement not currently analyzed
6. **Facial Expressions** - Not yet integrated (planned for future)

## Future Enhancements

Planned improvements to ASL recognition:

- [ ] Multi-hand coordination recognition
- [ ] Arm movement and position tracking
- [ ] Facial expression integration
- [ ] American English alphabet fingerspelling
- [ ] Number signs (1-10, etc.)
- [ ] Temporal/aspect modifiers
- [ ] Classifiers and spatial agreement
- [ ] Machine learning model fine-tuning on ASL corpus
- [ ] Real-time continuous signing recognition
- [ ] Context-aware sign disambiguation

## Troubleshooting

### Issue: Gesture not recognized
**Solution:**
- Ensure hands are clearly visible
- Check lighting conditions
- Hold pose steady for 1 second
- Train multiple samples for custom gestures
- Verify hand position (correct height, distance)

### Issue: Wrong gesture recognized
**Solution:**
- Train with multiple angle variations
- Increase hand clarity in frame
- Avoid partial hand visibility
- Check that hand shape matches sign definition

### Issue: Custom gesture not saving
**Solution:**
- Verify hands are detected in training frame
- Check gesture name isn't already in use
- Ensure you're logged in
- Try clearing browser cache

## Integration Examples

### JavaScript - Real-time Recognition

```javascript
const video = document.getElementById('video');
const canvas = document.getElementById('canvas');

async function detectASLSign() {
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0);
  
  canvas.toBlob(async (blob) => {
    const formData = new FormData();
    formData.append('frame', blob);
    
    const response = await fetch('/api/transcribe', {
      method: 'POST',
      body: formData
    });
    
    const data = await response.json();
    
    if (data.status === 'hand_detected') {
      console.log(`Detected: ${data.detected_sign}`);
      console.log(`Confidence: ${(data.confidence * 100).toFixed(1)}%`);
    }
  });
}

// Call regularly (e.g., every 250ms)
setInterval(detectASLSign, 250);
```

### Python - Gesture Training

```python
import requests

def train_asl_sign(gesture_name, image_path, token):
    """Train a new ASL gesture"""
    
    with open(image_path, 'rb') as f:
        files = {'frame': f}
        data = {'gesture_name': gesture_name}
        
        response = requests.post(
            'http://localhost:5000/api/train-gesture',
            files=files,
            data=data,
            headers={'Authorization': f'Bearer {token}'}
        )
        
    return response.json()

# Usage
result = train_asl_sign('HELLO', 'hello_gesture.jpg', 'your_token')
print(f"Status: {result['status']}")
print(f"Samples: {result['samples']}")
```

## Credits & References

- **MediaPipe**: Hand tracking model - https://mediapipe.dev
- **ASL Resources**: 
  - ASL Dictionary - https://www.handspeak.com
  - American Sign Language University - https://www.lifeprint.com
  - National Association of the Deaf - https://www.nad.org

## Support

For issues or feature requests related to ASL recognition:
- Create an issue on GitHub
- Contact: support@sign.com
- Documentation: See README.md and docs/ folder
