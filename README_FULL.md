# Sign - AI-Powered Sign Language Transcriber

## Project Overview

Sign is a production-oriented web application that enables real-time transcription of sign language to text using artificial intelligence. It serves both sign language users who need transcription services and administrators who manage the system.

## Features

### For Sign Language Users
- 👁️ Real-time sign language detection via webcam
- 📝 Automatic conversion to readable text
- 💾 Save and manage transcription sessions
- 📋 View transcription history
- ✏️ Edit transcripts before saving
- 🔍 Search and filter past transcriptions

### For Administrators
- 👥 User management (create, edit, delete users)
- 📊 Dashboard with analytics
- 📋 Manage all transcripts in the system
- ⚠️ Flag inappropriate content
- 📈 View detailed statistics
- 🗂️ Audit logs for compliance

## Technology Stack

- **Backend**: Flask 2.3.3
- **Database**: SQLAlchemy + SQLite (production: PostgreSQL)
- **Authentication**: Flask-Login + Werkzeug
- **Computer Vision**: OpenCV + MediaPipe (ready for integration)
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Deployment Ready**: Gunicorn-compatible

## Project Structure

```
sign-transcriber/
├── app.py                 # Main Flask application
├── config.py              # Configuration settings
├── models.py              # Database models
├── forms.py               # WTForms for validation
├── sign_detector.py       # Sign recognition module
├── requirements.txt       # Python dependencies
├── templates/
│   ├── base.html         # Base template
│   ├── index.html        # Landing page
│   ├── auth/
│   │   ├── login.html
│   │   └── register.html
│   ├── user/
│   │   ├── dashboard.html
│   │   ├── transcribe.html
│   │   └── view_transcript.html
│   ├── admin/
│   │   ├── dashboard.html
│   │   ├── manage_users.html
│   │   ├── create_user.html
│   │   ├── edit_user.html
│   │   ├── manage_transcripts.html
│   │   └── analytics.html
│   └── errors/
│       ├── 404.html
│       ├── 403.html
│       └── 500.html
├── static/
│   ├── css/style.css
│   └── js/script.js
└── .gitignore
```

## Installation

### Prerequisites
- Python 3.8+
- pip
- Virtual environment (recommended)

### Setup Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/ily6ix/sign-transcriber.git
   cd sign-transcriber
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Initialize the database**
   ```bash
   flask db-init
   flask seed-signs
   ```

5. **Run the application**
   ```bash
   python app.py
   ```

   The app will be available at `http://localhost:5000`

## Default Credentials

When initialized, the system creates a default admin user:
- **Username**: admin
- **Password**: admin123

⚠️ **Important**: Change these credentials in production!

## User Roles

### Sign Language User
- Active status required
- Can record sign language
- Can view and manage own transcripts
- Cannot access admin features

### Administrator
- Full system access
- Can manage users
- Can view all transcripts
- Can flag inappropriate content
- Access to analytics

## Database Models

### User
- Full name, username, email
- Password (hashed with Werkzeug)
- Role (user/admin)
- Active status
- Created/updated timestamps
- Last login timestamp

### Transcript
- Title, content (text), raw content (JSON)
- Confidence scores
- Session duration
- User reference
- Status (draft/completed/flagged)
- Created/updated timestamps

### SignDataset
- Sign name, description
- Image/video URLs
- Gesture type (letter/number/phrase)

### AuditLog
- Action performed
- Target type and ID
- Admin user reference
- Timestamp

## API Endpoints

### Authentication
- `POST /register` - User registration
- `POST /login` - User login
- `GET /logout` - User logout

### User Routes
- `GET /dashboard` - User dashboard
- `GET /transcribe` - Start new transcription
- `GET /transcript/<id>` - View transcript
- `POST /transcript/<id>` - Update transcript
- `POST /transcript/<id>/delete` - Delete transcript

### Admin Routes
- `GET /admin/dashboard` - Admin dashboard
- `GET /admin/users` - Manage users
- `POST /admin/users/create` - Create user
- `GET /admin/users/<id>/edit` - Edit user
- `POST /admin/users/<id>/delete` - Delete user
- `GET /admin/transcripts` - Manage transcripts
- `POST /admin/transcripts/<id>/flag` - Flag transcript
- `POST /admin/transcripts/<id>/delete` - Delete transcript
- `GET /admin/analytics` - View analytics

### API Endpoints
- `POST /api/transcribe` - Real-time sign detection
- `POST /api/start-session` - Start transcription session
- `POST /api/save-session/<id>` - Save session

## Configuration

Edit `config.py` to customize:

```python
# Database
SQLALCHEMY_DATABASE_URI = 'sqlite:///sign.db'

# Sign Detection
MODEL_CONFIG = {
    'confidence_threshold': 0.7,
    'detection_fps': 30,
    'supported_signs': 'A-Z'
}

# Session
PERMANENT_SESSION_LIFETIME = timedelta(days=7)
```

## Sign Detection Module

The application includes a placeholder `sign_detector.py` module ready for ML integration:

- `SignDetector` class: Main detector interface
- `MockSignDetector` class: Development/testing mock
- `initialize_detector()`: Factory function

### Integrating ML Models

To add actual sign detection:

1. Import your model (TensorFlow, MediaPipe, PyTorch, etc.)
2. Implement `detect_signs()` method in `SignDetector`
3. Return detection results in the expected format
4. Update configuration thresholds

Example structure:
```python
def detect_signs(self, frame):
    # Run model inference
    # Extract hand landmarks
    # Classify sign gesture
    return {
        'detected_sign': 'HELLO',
        'confidence': 0.95,
        'landmarks': [...],
        'hand_positions': [...]
    }
```

## Features Implemented

✅ User registration and authentication
✅ Role-based access control (RBAC)
✅ User management dashboard
✅ Transcript CRUD operations
✅ Admin dashboard with analytics
✅ Real-time transcription interface
✅ Error handling with custom pages
✅ Form validation
✅ Audit logging
✅ Responsive design

## Features Ready for Development

- 🔄 Full WebRTC video streaming (partial)
- 🧠 ML model integration for sign detection
- 📊 Advanced analytics and reporting
- 🔔 Email notifications
- 📱 Mobile app support
- 🌍 Multi-language support
- 🔐 Two-factor authentication
- 📤 Export to PDF/Word
- 🗣️ Text-to-speech conversion
- 🎤 Speech-to-text integration

## Environment Variables

Create a `.env` file:

```bash
FLASK_ENV=development
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///sign.db
DEBUG=True
```

## Deployment (Production)

### Using Gunicorn

```bash
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

### Using Docker

```bash
docker build -t sign .
docker run -p 5000:5000 sign
```

### Environment Setup
```bash
# Production database (PostgreSQL)
DATABASE_URL=postgresql://user:password@localhost/sign

# Production secret
SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')

# Disable debug
DEBUG=False

# HTTPS in production
SESSION_COOKIE_SECURE=True
```

## Testing

Run the test suite:
```bash
pytest
```

Run with coverage:
```bash
pytest --cov=. --cov-report=html
```

## Security

- ✅ Password hashing with Werkzeug
- ✅ CSRF protectionwith Flask-WTF
- ✅ SQL injection prevention via ORM
- ✅ Session security
- ✅ Role-based access control
- ⚠️ TODO: HTTPS enforcement in production
- ⚠️ TODO: Rate limiting
- ⚠️ TODO: Input sanitization

## Troubleshooting

**Database errors on first run**
```bash
flask init-db
```

**Import errors**
```bash
pip install -r requirements.txt --force-reinstall
```

**Permission denied errors**
```bash
chmod +x app.py
```

## Contributing

1. Create a new branch for your feature
2. Make changes and test thoroughly
3. Submit a pull request with description

## License

MIT License - Feel free to use in production

## Support

For issues and feature requests, open an issue on GitHub.

---

**Sign** - Breaking Communication Barriers with AI 🤟
