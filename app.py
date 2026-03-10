import os
from datetime import datetime
from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from functools import wraps
from models import db, User, Transcript, AuditLog, SignDataset
from forms import RegistrationForm, LoginForm, CreateUserForm, EditUserForm, TranscriptForm
from sign_detector import initialize_detector
from config import *

# Initialize Flask app
app = Flask(__name__)
app.config.from_object('config')

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth.login'
login_manager.login_message = 'Please log in to access this page.'

# Initialize sign detector (real-time hand tracking with MediaPipe)
sign_detector = initialize_detector()


@login_manager.user_loader
def load_user(user_id):
    """Load user from database"""
    return User.query.get(int(user_id))


def admin_required(f):
    """Decorator to require admin role"""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if not current_user.is_authenticated or not current_user.is_admin():
            flash('You need administrator privileges to access this page.', 'danger')
            return redirect(url_for('user_dashboard'))
        return f(*args, **kwargs)
    return decorated_function


def log_audit(action, target_type, target_id=None, details=None):
    """Create audit log entry"""
    if current_user.is_authenticated and current_user.is_admin():
        log = AuditLog(
            admin_id=current_user.id,
            action=action,
            target_type=target_type,
            target_id=target_id,
            details=details
        )
        db.session.add(log)
        db.session.commit()


# ============================================================================
# AUTH ROUTES
# ============================================================================

@app.route('/')
def index():
    """Landing page"""
    return render_template('index.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration"""
    if current_user.is_authenticated:
        return redirect(url_for('user_dashboard'))
    
    form = RegistrationForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            full_name=form.full_name.data,
            role='user'
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful! Please log in.', 'success')
        return redirect(url_for('login'))
    
    return render_template('auth/register.html', form=form)


@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login"""
    if current_user.is_authenticated:
        return redirect(url_for('user_dashboard'))
    
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        
        if user is None or not user.check_password(form.password.data):
            flash('Invalid username or password.', 'danger')
            return redirect(url_for('login'))
        
        if not user.is_active:
            flash('Your account has been deactivated.', 'danger')
            return redirect(url_for('login'))
        
        login_user(user)
        user.last_login = datetime.utcnow()
        db.session.commit()
        
        next_page = request.args.get('next')
        if next_page and next_page.startswith('/'):
            return redirect(next_page)
        
        return redirect(url_for('user_dashboard' if user.role == 'user' else 'admin_dashboard'))
    
    return render_template('auth/login.html', form=form)


@app.route('/logout')
@login_required
def logout():
    """User logout"""
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('index'))


# ============================================================================
# USER ROUTES - Sign Language User
# ============================================================================

@app.route('/dashboard')
@login_required
def user_dashboard():
    """User dashboard"""
    if current_user.is_admin():
        return redirect(url_for('admin_dashboard'))
    
    page = request.args.get('page', 1, type=int)
    transcripts = Transcript.query.filter_by(user_id=current_user.id).order_by(
        Transcript.created_at.desc()
    ).paginate(page=page, per_page=10)
    
    stats = {
        'total_transcripts': Transcript.query.filter_by(user_id=current_user.id).count(),
        'total_duration': db.session.query(db.func.sum(Transcript.session_duration)).filter_by(
            user_id=current_user.id
        ).scalar() or 0
    }
    
    return render_template('user/dashboard.html', transcripts=transcripts, stats=stats)


@app.route('/transcribe', methods=['GET', 'POST'])
@login_required
def transcribe():
    """Sign language transcription interface"""
    if current_user.is_admin():
        return redirect(url_for('admin_dashboard'))
    
    form = TranscriptForm()
    if form.validate_on_submit():
        transcript = Transcript(
            user_id=current_user.id,
            title=form.title.data,
            content=form.content.data,
            status='completed'
        )
        db.session.add(transcript)
        db.session.commit()
        
        flash('Transcript saved successfully!', 'success')
        return redirect(url_for('user_dashboard'))
    
    return render_template('user/transcribe.html', form=form)


@app.route('/transcript/<int:transcript_id>', methods=['GET', 'POST'])
@login_required
def view_transcript(transcript_id):
    """View specific transcript"""
    transcript = Transcript.query.get_or_404(transcript_id)
    
    # Check ownership
    if transcript.user_id != current_user.id and not current_user.is_admin():
        flash('You do not have permission to view this transcript.', 'danger')
        return redirect(url_for('user_dashboard'))
    
    form = TranscriptForm()
    if form.validate_on_submit():
        transcript.title = form.title.data
        transcript.content = form.content.data
        transcript.updated_at = datetime.utcnow()
        db.session.commit()
        
        flash('Transcript updated successfully!', 'success')
        return redirect(url_for('view_transcript', transcript_id=transcript.id))
    
    elif request.method == 'GET':
        form.title.data = transcript.title
        form.content.data = transcript.content
    
    return render_template('user/view_transcript.html', transcript=transcript, form=form)


@app.route('/transcript/<int:transcript_id>/delete', methods=['POST'])
@login_required
def delete_transcript(transcript_id):
    """Delete transcript"""
    transcript = Transcript.query.get_or_404(transcript_id)
    
    # Check ownership
    if transcript.user_id != current_user.id and not current_user.is_admin():
        return jsonify({'error': 'Unauthorized'}), 403
    
    db.session.delete(transcript)
    db.session.commit()
    
    flash('Transcript deleted successfully!', 'success')
    return redirect(url_for('user_dashboard'))


# ============================================================================
# ADMIN ROUTES
# ============================================================================

@app.route('/admin/dashboard')
@login_required
@admin_required
def admin_dashboard():
    """Admin dashboard with analytics"""
    total_users = User.query.count()
    total_transcripts = Transcript.query.count()
    active_users = User.query.filter_by(is_active=True).count()
    flagged_transcripts = Transcript.query.filter_by(status='flagged').count()
    
    # Recent activity
    recent_transcripts = Transcript.query.order_by(Transcript.created_at.desc()).limit(5).all()
    recent_users = User.query.order_by(User.created_at.desc()).limit(5).all()
    
    analytics = {
        'total_users': total_users,
        'total_transcripts': total_transcripts,
        'active_users': active_users,
        'flagged_transcripts': flagged_transcripts
    }
    
    return render_template('admin/dashboard.html', analytics=analytics, 
                         recent_transcripts=recent_transcripts,
                         recent_users=recent_users)


@app.route('/admin/users')
@login_required
@admin_required
def manage_users():
    """Manage all users"""
    page = request.args.get('page', 1, type=int)
    users = User.query.order_by(User.created_at.desc()).paginate(page=page, per_page=15)
    
    return render_template('admin/manage_users.html', users=users)


@app.route('/admin/users/create', methods=['GET', 'POST'])
@login_required
@admin_required
def create_user():
    """Create new user (admin only)"""
    form = CreateUserForm()
    if form.validate_on_submit():
        user = User(
            username=form.username.data,
            email=form.email.data,
            full_name=form.full_name.data,
            role=form.role.data
        )
        user.set_password(form.password.data)
        db.session.add(user)
        db.session.commit()
        
        log_audit('CREATE_USER', 'user', user.id, f"Created user {user.username}")
        flash(f'User {user.username} created successfully!', 'success')
        return redirect(url_for('manage_users'))
    
    return render_template('admin/create_user.html', form=form)


@app.route('/admin/users/<int:user_id>/edit', methods=['GET', 'POST'])
@login_required
@admin_required
def edit_user(user_id):
    """Edit user details (admin only)"""
    user = User.query.get_or_404(user_id)
    form = EditUserForm()
    
    if form.validate_on_submit():
        user.full_name = form.full_name.data
        user.email = form.email.data
        user.role = form.role.data
        user.is_active = form.is_active.data == 'True'
        db.session.commit()
        
        log_audit('UPDATE_USER', 'user', user.id, f"Updated user {user.username}")
        flash(f'User {user.username} updated successfully!', 'success')
        return redirect(url_for('manage_users'))
    
    elif request.method == 'GET':
        form.full_name.data = user.full_name
        form.email.data = user.email
        form.role.data = user.role
        form.is_active.data = str(user.is_active)
    
    return render_template('admin/edit_user.html', user=user, form=form)


@app.route('/admin/users/<int:user_id>/delete', methods=['POST'])
@login_required
@admin_required
def delete_user(user_id):
    """Delete user (admin only)"""
    user = User.query.get_or_404(user_id)
    
    # Prevent deleting self
    if user.id == current_user.id:
        return jsonify({'error': 'Cannot delete your own account'}), 400
    
    username = user.username
    db.session.delete(user)
    db.session.commit()
    
    log_audit('DELETE_USER', 'user', user_id, f"Deleted user {username}")
    flash(f'User {username} deleted successfully!', 'success')
    
    return redirect(url_for('manage_users'))


@app.route('/admin/transcripts')
@login_required
@admin_required
def manage_transcripts():
    """View and manage all transcripts"""
    page = request.args.get('page', 1, type=int)
    status_filter = request.args.get('status', 'all')
    
    query = Transcript.query
    if status_filter != 'all':
        query = query.filter_by(status=status_filter)
    
    transcripts = query.order_by(Transcript.created_at.desc()).paginate(page=page, per_page=15)
    
    return render_template('admin/manage_transcripts.html', transcripts=transcripts, 
                         current_status=status_filter)


@app.route('/admin/transcripts/<int:transcript_id>/flag', methods=['POST'])
@login_required
@admin_required
def flag_transcript(transcript_id):
    """Flag transcript as inappropriate"""
    transcript = Transcript.query.get_or_404(transcript_id)
    transcript.status = 'flagged'
    db.session.commit()
    
    log_audit('FLAG_TRANSCRIPT', 'transcript', transcript_id)
    flash('Transcript flagged successfully!', 'warning')
    
    return redirect(url_for('manage_transcripts'))


@app.route('/admin/transcripts/<int:transcript_id>/delete', methods=['POST'])
@login_required
@admin_required
def admin_delete_transcript(transcript_id):
    """Delete transcript (admin only)"""
    transcript = Transcript.query.get_or_404(transcript_id)
    user_id = transcript.user_id
    
    db.session.delete(transcript)
    db.session.commit()
    
    log_audit('DELETE_TRANSCRIPT', 'transcript', transcript_id, f"User: {user_id}")
    flash('Transcript deleted successfully!', 'success')
    
    return redirect(url_for('manage_transcripts'))


@app.route('/admin/analytics')
@login_required
@admin_required
def analytics():
    """Analytics and statistics"""
    users_count = User.query.count()
    transcripts_count = Transcript.query.count()
    
    # Get data for charts
    transcripts_by_status = db.session.query(
        Transcript.status,
        db.func.count(Transcript.id)
    ).group_by(Transcript.status).all()
    
    top_users = db.session.query(
        User.username,
        db.func.count(Transcript.id).label('transcript_count')
    ).join(Transcript).group_by(User.id).order_by(
        db.func.count(Transcript.id).desc()
    ).limit(10).all()
    
    return render_template('admin/analytics.html',
                         users_count=users_count,
                         transcripts_count=transcripts_count,
                         transcripts_by_status=transcripts_by_status,
                         top_users=top_users)


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/api/transcribe', methods=['POST'])
@login_required
def api_transcribe():
    """
    API endpoint for real-time transcription with hand gesture detection
    Only responds when hands are actually detected in the frame
    """
    try:
        # Check if frame data is provided
        if 'frame' not in request.files:
            return jsonify({
                'status': 'no_hand',
                'message': 'No frame data',
                'hands_detected': 0,
                'detected_sign': None
            }), 200
        
        # Read frame from request
        frame_file = request.files['frame']
        frame_stream = frame_file.stream
        frame_data = frame_stream.read()
        
        # Detect hand gestures in frame
        detection = sign_detector.detect_signs(frame_data)
        
        # Only return gesture data if hands were detected
        if detection['has_hands']:
            return jsonify({
                'status': 'hand_detected',
                'hands_detected': detection['hands_detected'],
                'detected_sign': detection['detected_sign'],
                'gestures': detection['gestures'],
                'confidence': detection['confidence'],
                'hand_positions': detection['hand_positions']
            }), 200
        else:
            return jsonify({
                'status': 'no_hand',
                'hands_detected': 0,
                'detected_sign': None,
                'confidence': 0.0
            }), 200
            
    except Exception as e:
        print(f"Error in transcription API: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e),
            'hands_detected': 0
        }), 500


@app.route('/api/start-session', methods=['POST'])
@login_required
def api_start_session():
    """Start a transcription session"""
    data = request.get_json()
    
    transcript = Transcript(
        user_id=current_user.id,
        title=data.get('title', 'Untitled Session'),
        status='draft'
    )
    db.session.add(transcript)
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'session_id': transcript.id
    })


@app.route('/api/save-session/<int:session_id>', methods=['POST'])
@login_required
def api_save_session(session_id):
    """Save transcription session"""
    transcript = Transcript.query.get_or_404(session_id)
    
    # Verify ownership
    if transcript.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    data = request.get_json()
    transcript.content = data.get('content', '')
    transcript.raw_content = data.get('raw_content', [])
    transcript.confidence_scores = data.get('confidence_scores', [])
    transcript.session_duration = data.get('duration', 0)
    transcript.status = 'completed'
    
    db.session.commit()
    
    return jsonify({
        'status': 'success',
        'message': 'Session saved successfully'
    })


# ============================================================================
# GESTURE TRAINING API
# ============================================================================

@app.route('/api/train-gesture', methods=['POST'])
@login_required
def api_train_gesture():
    """
    Train the sign detector on a new ASL gesture
    Users can teach the system custom signs
    """
    try:
        data = request.get_json()
        
        if not data or 'gesture_name' not in data or 'frame' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing gesture_name or frame data'
            }), 400
        
        gesture_name = data['gesture_name'].upper().strip()
        frame_data = data['frame']
        
        # Limit gesture name length
        if len(gesture_name) > 50:
            return jsonify({
                'status': 'error',
                'message': 'Gesture name too long (max 50 characters)'
            }), 400
        
        # Detect Hand gesture in frame
        detection = sign_detector.detect_signs(frame_data)
        
        # Only train if hands are detected
        if not detection['has_hands']:
            return jsonify({
                'status': 'error',
                'message': 'No hands detected in frame. Please show your hand clearly.'
            }), 400
        
        # Get landmarks from first detected hand
        if detection['landmarks'] and len(detection['landmarks']) > 0:
            landmarks = detection['landmarks'][0]
            
            # Train the gesture
            sign_detector.train_gesture(gesture_name, landmarks)
            
            return jsonify({
                'status': 'success',
                'message': f'Gesture "{gesture_name}" trained successfully',
                'gesture_name': gesture_name,
                'samples': sign_detector.get_gesture_samples(gesture_name)
            }), 200
        else:
            return jsonify({
                'status': 'error',
                'message': 'Could not extract hand landmarks'
            }), 400
            
    except Exception as e:
        print(f"Error training gesture: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/trained-gestures', methods=['GET'])
@login_required
def api_list_trained_gestures():
    """
    Get list of all trained custom gestures
    """
    try:
        trained = sign_detector.get_trained_gestures()
        
        # Get sample counts for each gesture
        gesture_info = {}
        for gesture in trained:
            gesture_info[gesture] = {
                'samples': sign_detector.get_gesture_samples(gesture)
            }
        
        return jsonify({
            'status': 'success',
            'trained_gestures': gesture_info,
            'total': len(trained)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


@app.route('/api/clear-training', methods=['POST'])
@login_required
@admin_required
def api_clear_training():
    """
    Clear all trained custom gestures (admin only)
    """
    try:
        sign_detector.clear_gesture_training()
        
        return jsonify({
            'status': 'success',
            'message': 'All trained gestures cleared'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500


# ============================================================================
# ERROR HANDLERS
# ============================================================================

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return render_template('errors/404.html'), 404


@app.errorhandler(403)
def forbidden(error):
    """Handle 403 errors"""
    return render_template('errors/403.html'), 403


@app.errorhandler(500)
def server_error(error):
    """Handle 500 errors"""
    db.session.rollback()
    return render_template('errors/500.html'), 500


# ============================================================================
# DATABASE INITIALIZATION
# ============================================================================

@app.before_request
def create_tables():
    """Create database tables if they don't exist"""
    db.create_all()


@app.cli.command()
def init_db():
    """Initialize database with sample data"""
    db.create_all()
    
    # Create sample admin user if it doesn't exist
    if User.query.filter_by(username='admin').first() is None:
        admin = User(
            username='admin',
            email='admin@sign.com',
            full_name='Administrator',
            role='admin'
        )
        admin.set_password('admin123')
        db.session.add(admin)
        db.session.commit()
        print('Database initialized with admin user (username: admin, password: admin123)')
    else:
        print('Database already initialized')


@app.cli.command()
def seed_signs():
    """Seed sample sign data"""
    sample_signs = [
        ('A', 'Letter A in ASL', 'letter'),
        ('B', 'Letter B in ASL', 'letter'),
        ('HELLO', 'Greeting sign', 'phrase'),
        ('THANK YOU', 'Expression of gratitude', 'phrase'),
        ('YES', 'Affirmative response', 'phrase'),
        ('NO', 'Negative response', 'phrase'),
    ]
    
    for sign_name, description, gesture_type in sample_signs:
        if SignDataset.query.filter_by(sign_name=sign_name).first() is None:
            sign = SignDataset(
                sign_name=sign_name,
                description=description,
                gesture_type=gesture_type
            )
            db.session.add(sign)
    
    db.session.commit()
    print('Sign database seeded successfully')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
