from flask import Flask, render_template, request, jsonify, session, redirect, url_for
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user
from datetime import datetime

app = Flask(__name__)
app.secret_key = 'open123'  # Change this to a secure random key

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# Dummy user data for demonstration purposes
class User(UserMixin):
    def __init__(self, id):
        self.id = id

users = {'user1': {'password': 'password1'}}

# Store the chat messages
chat_messages = []

@login_manager.user_loader
def load_user(user_id):
    return User(user_id)

@app.route('/')
@login_required
def index():
    return render_template('index.html', chat_messages=chat_messages)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        if users.get(username) and users[username]['password'] == password:
            user = User(username)
            login_user(user)
            return redirect(url_for('index'))
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/send_message', methods=['POST'])
@login_required
def send_message():
    username = session.get('user_id')
    message = request.form.get('message')

    if username and message:
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        chat_messages.append({'username': username, 'message': message, 'timestamp': timestamp})
        return jsonify({'status': 'success'})
    else:
        return jsonify({'status': 'error', 'message': 'Message cannot be empty'})

if __name__ == '__main__':
    host = '0.0.0.0'  # Change this to your desired IP address
    port = 5022      # Change this to your desired port number

    app.run(host=host, port=port, debug=True)
