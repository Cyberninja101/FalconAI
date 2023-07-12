from flask import Flask, redirect, url_for, render_template, request

import os

app = Flask(__name__)

@app.route("/")
def home():
    """
    This is the home/default page.
    """
    return render_template("home.html")

@app.route("/new_entry", methods=["POST"])
def new_entry():
    """
    This is when the user asks FALCON a new question, and enters it.
    The question should be saved as a json file, displayed in the 
    chat log, and then sent to the LLM.
    """

    if request.method == "POST":
        print("It worked")
        entry = request.form
        print(f"Here is the entry: {entry}")

        # TODO: Send info to LLM somehow

    
    return render_template("home.html")
    


if __name__ == "__main__":
    app.run(debug=True) # Set debug = True for live changes in development