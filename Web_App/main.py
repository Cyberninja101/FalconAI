from flask import Flask, redirect, url_for, render_template, request
from dummy_model import model
import os
import binascii

app = Flask(__name__)

# Chat Log
chat_log = []


@app.route("/")
def home():
    """
    This is the home/default page.
    """
    # reset chat_log
    # global chat_log
    # chat_log = []

    return render_template("home.html")

@app.route("/new_entry/<entry>", methods=["POST"])
def new_entry(entry):
    """
    This is when the user asks FALCON a new question, and enters it.
    The question should be saved as a json file, displayed in the 
    chat log, and then sent to the LLM.

    """
    print("This is the new_entry")
    if request.method == "POST":
    # making error cause there is no form
        print(f"Here is the entry: {entry}")
        # TODO: Send info to LLM somehow
        # This is temporary dummy function
        ls = []

        for i in entry.split(","):
            hex_string = i
            bytes_object = bytes.fromhex(hex_string)
            ascii_string = bytes_object.decode("ASCII")
            ls.append(ascii_string)
        output = "".join(ls)

        x = chatbot.run(output)
        print(x[1])
        return x[0]


@app.route("/upload_file", methods=["POST"])
def upload_file():
    if request.method == "POST":
        f = request.files['context_file']
        f.save(os.sep.join(["Web_APP", "contexts",f.filename]))

        return " "

if __name__ == "__main__":
    chatbot = model()
    app.run(debug=True) # Set debug = True for live changes in development
    
