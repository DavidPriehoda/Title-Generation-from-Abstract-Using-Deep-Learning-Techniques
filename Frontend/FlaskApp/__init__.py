from flask import Flask
from flask_wtf.csrf import CSRFProtect

app = Flask(__name__)
app.config.from_pyfile('config.py')

csrf = CSRFProtect(app)

from FlaskApp import routes