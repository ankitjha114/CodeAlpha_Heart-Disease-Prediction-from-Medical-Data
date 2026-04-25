import json
import hashlib
import os

DB_FILE = "data.json"

# ===============================
# INIT DATABASE
# ===============================
def init_db():
    if not os.path.exists(DB_FILE):
        with open(DB_FILE, "w") as f:
            json.dump({"users": [], "predictions": []}, f)

# ===============================
# LOAD / SAVE
# ===============================
def load_db():
    with open(DB_FILE, "r") as f:
        return json.load(f)

def save_db(data):
    with open(DB_FILE, "w") as f:
        json.dump(data, f, indent=4)

# ===============================
# PASSWORD HASH
# ===============================
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# ===============================
# USER FUNCTIONS
# ===============================
def signup_user(username, password):
    data = load_db()
    for user in data["users"]:
        if user["username"] == username:
            return False

    data["users"].append({
        "username": username,
        "password": hash_password(password)
    })

    save_db(data)
    return True

def login_user(username, password):
    data = load_db()
    for user in data["users"]:
        if user["username"] == username and user["password"] == hash_password(password):
            return True
    return False

# ===============================
# PREDICTION STORAGE
# ===============================
def save_prediction(username, age, chol, bp, prediction, probability):
    data = load_db()

    data["predictions"].append({
        "username": username,
        "age": age,
        "chol": chol,
        "bp": bp,
        "prediction": int(prediction),
        "probability": float(probability)
    })

    save_db(data)

def get_user_predictions(username):
    data = load_db()
    return [p for p in data["predictions"] if p["username"] == username]