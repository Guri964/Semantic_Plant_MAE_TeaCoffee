from flask import Blueprint, request, session, redirect, url_for, render_template, jsonify
from database import get_user_by_username, create_user, verify_password

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')

    data     = request.get_json() if request.is_json else request.form
    username = data.get('username', '').strip()
    password = data.get('password', '').strip()

    if not username or not password:
        if request.is_json:
            return jsonify({'success': False, 'message': 'Username and password are required.'}), 400
        return render_template('login.html', error='Username and password are required.')

    user = get_user_by_username(username)
    if not user or not verify_password(password, user['password_hash']):
        if request.is_json:
            return jsonify({'success': False, 'message': 'Invalid username or password.'}), 401
        return render_template('login.html', error='Invalid username or password.')

    session.clear()
    session['user_id']  = user['id']
    session['username'] = user['username']

    if request.is_json:
        return jsonify({'success': True, 'redirect': url_for('predict.dashboard')})
    return redirect(url_for('predict.dashboard'))


@auth_bp.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'GET':
        return render_template('signup.html')

    data             = request.get_json() if request.is_json else request.form
    username         = data.get('username', '').strip()
    password         = data.get('password', '').strip()
    confirm_password = data.get('confirm_password', '').strip()

    if not username or not password or not confirm_password:
        if request.is_json:
            return jsonify({'success': False, 'message': 'All fields are required.'}), 400
        return render_template('signup.html', error='All fields are required.')

    if password != confirm_password:
        if request.is_json:
            return jsonify({'success': False, 'message': 'Passwords do not match.'}), 400
        return render_template('signup.html', error='Passwords do not match.')

    if len(password) < 6:
        if request.is_json:
            return jsonify({'success': False, 'message': 'Password must be at least 6 characters.'}), 400
        return render_template('signup.html', error='Password must be at least 6 characters.')

    if get_user_by_username(username):
        if request.is_json:
            return jsonify({'success': False, 'message': 'Username already exists. Please login instead.'}), 409
        return render_template('signup.html', error='Username already exists. Please login instead.')

    create_user(username, password)

    if request.is_json:
        return jsonify({'success': True, 'redirect': url_for('auth.login')})
    return redirect(url_for('auth.login'))


@auth_bp.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('main.index'))