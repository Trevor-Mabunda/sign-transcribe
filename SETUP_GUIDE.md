# Sign - Complete Setup Guide

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Git
- 100MB free disk space
- Modern web browser

## 🚀 Installation Steps

### Step 1: Clone Repository

```bash
git clone https://github.com/ily6ix/sign-transcriber.git
cd sign-transcriber
```

### Step 2: Create Virtual Environment

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Initialize Database

```bash
python init_db.py
```

This will:
- Create the SQLite database
- Create all tables
- Create default admin user (admin/admin123)
- Seed sample sign language data

### Step 5: Run Application

```bash
python app.py
```

You should see:
```
 * Running on http://0.0.0.0:5000
 * Debug mode: on
```

### Step 6: Access Application

Open your browser and visit: **http://localhost:5000**

## 🔐 Default Credentials

```
Username: admin
Password: admin123
```

⚠️ **IMPORTANT**: Change these credentials in production!

## 📊 First Steps After Login

1. **As Admin:**
   - Visit `/admin/dashboard` - View system overview
   - Go to `Manage Users` - Create additional users
   - Check `Analytics` - View system statistics

2. **As Regular User:**
   - Visit `/dashboard` - See your transcripts
   - Click `Start Transcription` - Begin recording signs
   - View past transcripts - Edit or delete

## 🛠️ Configuration

### Environment Variables

Create a `.env` file in the project root:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
FLASK_ENV=development
DEBUG=True
SECRET_KEY=your-secret-key-here
DATABASE_URL=sqlite:///signatranslate.db
```

### Database

**Development (default):**
```python
DATABASE_URL = 'sqlite:///signatranslate.db'
```

**Production (PostgreSQL):**
```python
DATABASE_URL = 'postgresql://user:password@localhost/signatranslate'
```

## 📱 Application Structure

### Routes Overview

| Route | Purpose | Requires Auth |
|-------|---------|:--:|
| `/` | Landing page | ❌ |
| `/register` | User registration | ❌ |
| `/login` | User login | ❌ |
| `/dashboard` | User dashboard | ✅ |
| `/transcribe` | Transcription interface | ✅ |
| `/admin/dashboard` | Admin dashboard | ✅ (admin) |
| `/admin/users` | Manage users | ✅ (admin) |
| `/admin/transcripts` | Manage transcripts | ✅ (admin) |
| `/admin/analytics` | Analytics dashboard | ✅ (admin) |

### Database Tables

```
users (4,000+ bytes per record)
├── id
├── username (unique)
├── email (unique)
├── password_hash
├── full_name
├── role (user/admin)
├── is_active
├── created_at
├── updated_at
└── last_login

transcripts (2,000+ bytes per record)
├── id
├── user_id (FK)
├── title
├── content
├── raw_content (JSON)
├── confidence_scores (JSON)
├── session_duration
├── status
├── created_at
└── updated_at

sign_dataset
├── id
├── sign_name
├── description
├── image_url
├── video_url
├── gesture_type

audit_logs
├── id
├── admin_id (FK)
├── action
├── target_type
├── target_id
└── timestamp
```

## 🔄 Common Tasks

### Reset Database

Remove the database file and reinitialize:

```bash
rm signatranslate.db
python init_db.py
```

### Create New Admin User

```bash
python
>>> from app import app, db
>>> from models import User
>>> with app.app_context():
...     user = User(username='newadmin', email='admin2@example.com', 
...                 full_name='Admin Two', role='admin')
...     user.set_password('password123')
...     db.session.add(user)
...     db.session.commit()
>>> exit()
```

### Backup Database

```bash
cp signatranslate.db signatranslate.db.backup
```

### Check Database Contents

```bash
sqlite3 signatranslate.db
sqlite> SELECT * FROM users;
sqlite> .quit
```

## 🚢 Deployment

### Using Gunicorn (Recommended for Production)

1. **Install Gunicorn:**
   ```bash
   pip install gunicorn
   ```

2. **Run with Gunicorn:**
   ```bash
   gunicorn -w 4 -b 0.0.0.0:5000 app:app
   ```

### Using Docker

1. **Build Docker Image:**
   ```bash
   docker build -t signatranslate .
   ```

2. **Run Container:**
   ```bash
   docker run -p 5000:5000 signatranslate
   ```

### Using Heroku

1. **Install Heroku CLI**

2. **Login and Deploy:**
   ```bash
   heroku login
   heroku create your-app-name
   git push heroku main
   ```

### Production Checklist

- [ ] Change admin password
- [ ] Set `DEBUG = False` in config.py
- [ ] Generate strong `SECRET_KEY`
- [ ] Use PostgreSQL database
- [ ] Enable HTTPS/SSL
- [ ] Set up regular backups
- [ ] Configure logging
- [ ] Set up monitoring
- [ ] Enable rate limiting
- [ ] Set up email notifications

## 🐛 Troubleshooting

### "ModuleNotFoundError: No module named 'flask'"

```bash
pip install -r requirements.txt
```

### "Database is locked"

This occurs when multiple processes access SQLite. Use PostgreSQL for production:

```bash
pip install psycopg2
# Update DATABASE_URL in config.py
```

### "Port 5000 already in use"

Use a different port:

```bash
python app.py --port 5001
# Or in config:
app.run(port=5001)
```

### Static files not loading

Clear browser cache (Ctrl+Shift+R or Cmd+Shift+R)

### Login not working

1. Check database initialized: `python init_db.py`
2. Verify credentials: Admin / admin123
3. Check browser cookies enabled

## 📚 Learning Resources

### Flask Documentation
- [Flask Official Docs](https://flask.palletsprojects.com/)
- [SQLAlchemy ORM](https://docs.sqlalchemy.org/)
- [Flask-Login](https://flask-login.readthedocs.io/)

### Sign Language ML
- [MediaPipe Hands](https://google.github.io/mediapipe/solutions/hands)
- [TensorFlow Lite](https://www.tensorflow.org/lite)
- [OpenCV Tutorials](https://docs.opencv.org/)

### Web Development
- [MDN Web Docs](https://developer.mozilla.org/)
- [HTML/CSS Guide](https://web.dev/)
- [JavaScript Basics](https://javascript.info/)

## 📞 Support

### Getting Help

1. Check [GitHub Issues](https://github.com/ily6ix/sign-transcriber/issues)
2. Read project [documentation](./README_FULL.md)
3. Search Stack Overflow with tag `flask`
4. Contact: support@signatranslate.com

### Reporting Bugs

Create an issue on GitHub with:
- Python version
- OS (Windows/Mac/Linux)
- Steps to reproduce
- Expected vs actual behavior
- Error messages

## 🎓 Educational Use

This project is designed for students learning:
- Full-stack web development
- Database design
- Authentication and security
- REST API development
- Machine learning integration
- Software engineering best practices

## 📄 License

MIT License - Feel free to use commercially

---

**SignaTranslate** - Breaking Communication Barriers with AI

🤟 Built for accessibility and inclusion
