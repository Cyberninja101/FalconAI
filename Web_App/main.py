from flask import Flask, redirect, url_for, render_template, request

from pdfReader import read_pdf
import os
import binascii
import sys

# caution: path[0] is reserved for script path (or '' in REPL)
# print(os.sep.join([os.getcwd(),"Web_App", "models"]))
sys.path.insert(1, os.sep.join([os.getcwd(),"Web_App", "models"])) # to get path of model functions

from test_model import gpt2, model

app = Flask(__name__)

# Chat Log
chat_log = []

chatbot = model()

@app.route("/")
def home():
    """
    This is the home/default page. ok
    """
    # Refresh the chatbot history
    chatbot.__init__()
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
        return chatbot.predict(str(output))


@app.route("/upload_file", methods=["POST"])
def upload_file():
    if request.method == "POST":
        f = request.files['context_file']
        f.save(os.sep.join(["Web_App", "contexts",f.filename]))


        # TODO: FOR JUSTIN - Convert pdf file to string using ur function
        # and send it to the chatbot model as context

        # empty return with 204 code, means its good
        x = read_pdf(f"Web_App/contexts/{f.filename}") # - content of uploaded as string TODO: implement 
        # print(x)
        return '', 204

if __name__ == "__main__":
    
    app.run(debug=True) # Set debug = True for live changes in development

