from flask import Flask, redirect, url_for, render_template, request

import os

app = Flask(__name__)

@app.route("/")
def home():
    """
    This is the home/default page.
    """
    return render_template("home.html")


if __name__ == "__main__":
    app.run(debug=True) # Set debug = True for live changes in development