from flask import Flask, render_template

def create_app():
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)

    @app.route('/')
    def index():
        return render_template("index.html")

    return app