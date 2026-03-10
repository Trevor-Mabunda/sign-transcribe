# Sign - AI-Powered Sign Language Transcriber

> A production-ready web application for real-time sign language transcription powered by artificial intelligence.

## 🎯 Project Overview

Sign is a full-stack Flask application designed to make sign language accessible through AI. It enables:

- **Sign Language Users** to transcribe gestures into text in real-time
- **Administrators** to manage users, transcripts, and system analytics
- **Developers** to integrate state-of-the-art ML models for sign recognition

## ✨ Key Features

### 👤 User Features
- Sign up and login with secure authentication
- Start real-time sign language transcription sessions
- Save, edit, and delete personal transcripts
- View transcription history with detailed metadata
- User-friendly dashboard with statistics

### 🛡️ Admin Features
- Complete user management system
- View and manage all transcripts
- Flag inappropriate content
- Access comprehensive analytics dashboard
- Audit logging for compliance

### 🔧 Developer Features
- Production-ready Flask architecture
- SQLAlchemy ORM with multiple database support
- Role-based access control (RBAC)
- Modular sign detection system ready for ML integration
- RESTful API endpoints
- Comprehensive error handling

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip
- Virtual environment

### Installation (5 minutes)

```bash
# 1. Clone repository
git clone https://github.com/ily6ix/sign-transcriber.git
cd sign-transcriber

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Initialize database
flask init-db
flask seed-signs

# 5. Run application
python app.py
```

Visit `http://localhost:5000` in your browser.

### Default Credentials
- **Username**: `admin`
- **Password**: `admin123`

⚠️ Change these in production!

## 📊 System Architecture

### Database Schema

```
Users (with roles: user/admin)
├── Transcripts (owned by users)
│   ├── Content (text)
│   ├── Raw Data (JSON)
│   ├── Confidence Scores
│   └── Status (draft/completed/flagged)
├── SignDataset (vocabulary)
│   ├── Sign Names
│   ├── Images/Videos
│   └── Gesture Types
└── AuditLogs (admin actions)
```

### User Roles

| Feature | Sign Language User | Administrator |
|---------|:--:|:--:|
| Record Signs | ✅ | ✅ |
| Manage Own Transcripts | ✅ | ❌ |
| View All Transcripts | ❌ | ✅ |
| Manage Users | ❌ | ✅ |
| View Analytics | ❌ | ✅ |
| Flag Content | ❌ | ✅ |

## 📁 Project Structure

```
sign-transcriber/
├── app.py                    # Main Flask application (630+ lines)
├── config.py                 # Configuration settings
├── models.py                 # Database models (4 models)
├── forms.py                  # Form validation (6 forms)
├── sign_detector.py          # Sign detection module
├── requirements.txt          # Dependencies
├── templates/                # HTML templates (13 files)
│   ├── base.html
│   ├── index.html (landing page)
│   ├── auth/ (login/register)
│   ├── user/ (dashboard/transcribe)
│   ├── admin/ (management pages)
│   └── errors/ (error pages)
├── static/
│   ├── css/style.css        # Responsive styling
│   └── js/script.js         # Client-side logic
├── README.md                # This file
└── .gitignore              # Git exclusions
```

## 🔌 API Endpoints

### Authentication
```
POST   /register                    # User registration
POST   /login                       # User login
GET    /logout                      # User logout
```

### User (Sign Language Users)
```
GET    /dashboard                   # View dashboard
GET    /transcribe                  # Transcription interface
GET    /transcript/<id>             # View transcript
POST   /transcript/<id>             # Update transcript
POST   /transcript/<id>/delete      # Delete transcript
```

### Admin
```
GET    /admin/dashboard             # Admin dashboard
GET    /admin/users                 # Manage users
POST   /admin/users/create          # Create user
GET    /admin/users/<id>/edit       # Edit user
POST   /admin/users/<id>/delete     # Delete user
GET    /admin/transcripts           # Manage transcripts
POST   /admin/transcripts/<id>/flag # Flag content
POST   /admin/transcripts/<id>/delete # Delete transcript
GET    /admin/analytics             # View analytics
```

### API
```
POST   /api/transcribe              # Real-time sign detection
POST   /api/start-session           # Start session
POST   /api/save-session/<id>       # Save session
```

## 🧠 ML Integration Ready

The application includes a modular sign detection system:

```python
# In sign_detector.py
class SignDetector:
    def detect_signs(self, frame) -> Dict:
        # Implement your ML model here
        # Returns: detected_sign, confidence, landmarks
        pass
```

**Supported ML Frameworks:**
- MediaPipe HandTrack
- TensorFlow
- PyTorch
- OpenCV
- Custom models

## 🔒 Security Features

- ✅ Password hashing (Werkzeug)
- ✅ CSRF protection (Flask-WTF)
- ✅ SQL injection prevention (SQLAlchemy ORM)
- ✅ Session management
- ✅ Role-based access control
- ✅ Input validation

## 📱 Responsive Design

Works perfectly on:
- 🖥️ Desktop (1920px+)
- 💻 Laptop (1024px+)
- 📱 Tablet (768px+)
- 📱 Mobile (320px+)

## 🔧 Configuration

Edit `config.py` or use environment variables:

```python
# Database
DATABASE_URL = 'sqlite:///sign.db'

# Sign Detection
MODEL_CONFIG = {
    'confidence_threshold': 0.7,
    'detection_fps': 30,
}

# Session
PERMANENT_SESSION_LIFETIME = timedelta(days=7)
```

## 🚢 Deployment

### Development
```bash
python app.py
```

### Production with Gunicorn
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
# Use .env file
cp .env.example .env
# Edit .env with production settings
```

## 📊 Database Models

### User
- username (unique)
- email (unique)
- password_hash
- full_name
- role (user/admin)
- is_active
- timestamps

### Transcript
- title
- content
- raw_content (JSON)
- confidence_scores
- session_duration
- status (draft/completed/flagged)
- user_id (foreign key)
- timestamps

### SignDataset
- sign_name
- description
- image_url, video_url
- gesture_type

### AuditLog
- action
- target_type, target_id
- admin_id
- timestamp

## 🧪 Testing

```bash
pytest
pytest --cov=. --cov-report=html
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit: `git commit -m 'Add amazing feature'`
4. Push: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📝 License

MIT License - See LICENSE file for details

## 🎓 Student Project Notes

This project demonstrates:
- Full-stack web development
- Database design and ORM
- Authentication and authorization
- RESTful API design
- Frontend-backend integration
- Production-ready practices
- Software engineering principles

## 🔗 Technology Stack

| Layer | Technology |
|-------|-----------|
| Backend | Flask 2.3.3 |
| Database | SQLAlchemy + SQLite |
| Authentication | Flask-Login |
| Forms | Flask-WTF |
| Frontend | HTML5, CSS3, JavaScript |
| Computer Vision | OpenCV (ready) |
| Deployment | Gunicorn-ready |

## 📞 Support & Contact

- 📧 Email: support@sign.com
- 🐛 Issues: [GitHub Issues](https://github.com/ily6ix/sign-transcriber/issues)
- 💬 Discussions: [GitHub Discussions](https://github.com/ily6ix/sign-transcriber/discussions)

---

**Sign** - Making Sign Language Accessible Through AI 🤟

Built with ❤️ for accessibility and inclusion.

[⬆ Back to Top](#sign---ai-powered-sign-language-transcriber)