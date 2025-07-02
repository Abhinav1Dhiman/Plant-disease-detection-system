from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
from flask_login import LoginManager, login_user, login_required, logout_user, current_user
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, PlantScan
import os
import tensorflow as tf
import numpy as np
from datetime import datetime

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this to a secure key
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///plant_disease.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize extensions
db.init_app(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load the model
model = tf.keras.models.load_model('model.h5')
class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
BATCH_SIZE = 32
IMAGE_SIZE = 255
CHANNEL = 3
EPOCHS = 20

# Disease information database
DISEASE_INFO = {
    'Potato___Early_blight': {
        'symptoms': 'Dark brown spots with concentric rings that form a "bull\'s eye" pattern. Spots appear on older leaves first and can grow up to 1/2 inch in diameter. Affected leaves turn yellow and die.',
        'treatment': 'Remove infected leaves and apply appropriate fungicides like chlorothalonil or copper-based products. Apply fungicides every 7-10 days during wet weather.',
        'prevention': 'Rotate crops every 2-3 years, ensure good air circulation between plants, water at soil level to keep leaves dry, and remove plant debris after harvest.'
    },
    'Potato___Late_blight': {
        'symptoms': 'Dark green to purple-black water-soaked spots on leaves that quickly enlarge to form brown to purplish-black lesions. White fuzzy growth appears on leaf undersides in humid conditions.',
        'treatment': 'Remove and destroy infected plants immediately. Apply fungicides containing chlorothalonil, mancozeb, or copper as soon as symptoms appear.',
        'prevention': 'Plant resistant varieties, improve drainage and air circulation, avoid overhead irrigation, and destroy volunteer potato plants.'
    }
}

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Function to preprocess and predict
def predict(img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# Authentication routes
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        user = User.query.filter_by(username=username).first()
        
        if user and check_password_hash(user.password, password):
            login_user(user)
            return redirect(url_for('home'))
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username')
        email = request.form.get('email')
        password = request.form.get('password')
        
        if User.query.filter_by(username=username).first():
            flash('Username already exists')
            return redirect(url_for('register'))
            
        user = User(username=username,
                    email=email,
                    password=generate_password_hash(password))
        db.session.add(user)
        db.session.commit()
        
        flash('Registration successful!')
        return redirect(url_for('login'))
    return render_template('register.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))

@app.route('/history')
@login_required
def history():
    scans = PlantScan.query.filter_by(user_id=current_user.id).order_by(PlantScan.timestamp.desc()).all()
    return render_template('history.html', scans=scans)

@app.route('/api/save-note', methods=['POST'])
@login_required
def save_note():
    scan_id = request.form.get('scan_id')
    note = request.form.get('note')
    scan = PlantScan.query.get_or_404(scan_id)
    
    if scan.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
        
    scan.notes = note
    db.session.commit()
    return jsonify({'success': True})

# Route to the home page
@app.route('/', methods=['GET', 'POST'])
@login_required
def home():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return render_template('index.html', message='No file part')

        file = request.files['file']

        # If the user does not select a file, browser submits an empty file without a filename
        if file.filename == '':
            return render_template('index.html', message='No selected file')

        # If the file is allowed and has an allowed extension
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('static', 'uploads', filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            file.save(filepath)

            # Read the image
            img = tf.keras.preprocessing.image.load_img(filepath, target_size=(IMAGE_SIZE, IMAGE_SIZE))

            # Predict using the loaded model
            predicted_class, confidence = predict(img)

            # Save the scan to database
            scan = PlantScan(
                image_path=filepath,
                prediction=predicted_class,
                confidence=confidence,
                user_id=current_user.id
            )
            db.session.add(scan)
            db.session.commit()

            # Render the template with the uploaded image and prediction
            return render_template('index.html',
                                 image_path=filepath,
                                 predicted_label=predicted_class,
                                 confidence=confidence,
                                 user=current_user)

    return render_template('index.html', user=current_user)

# Function to check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/get_disease_info')
def get_disease_info():
    disease = request.args.get('disease')
    if disease in DISEASE_INFO:
        return jsonify(DISEASE_INFO[disease])
    return jsonify({'error': 'Disease information not available'})

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(debug=True)
