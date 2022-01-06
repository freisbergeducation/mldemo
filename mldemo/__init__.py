from flask import Flask, render_template
import os

def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/')
    def index():
        return render_template("index.html")
    
    if __name__ == '__main__':
        PORT = int(os.environ.get('PORT', 5000))

    return app