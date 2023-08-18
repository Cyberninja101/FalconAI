from flask import Flask, redirect, url_for, render_template, request, session
# The Session instance is not used for direct access, you should always use flask.session
from flask_session import Session

from models.pdfReader import read_pdf
import os, shutil
import binascii
import sys

# caution: path[0] is reserved for script path (or '' in REPL)
# print(os.sep.join([os.getcwd(),"Web_App", "models"]))


# from models.test_model import gpt2_model
# from models.hmt import HMT

app = Flask(__name__)
app.secret_key = "1234"
app.config["SESSION_PERMANENT"] = False
app.config['SESSION_TYPE'] = 'filesystem'


Session(app)

# Chat Log
chat_store = []

# Defining models
# finetuned_model = gpt2_model()

# hmt_model = HMT()

# Only load models in here
# if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
    # The reloader has already run - do what you want to do here



from models.vector_db import vectordb
from models.finetuned import radar_llama

finetuned_model = radar_llama()
vectordb_model = vectordb()



@app.route("/")
def home():
    """
    This is the home/default page. ok
    """
    print("testing ok")
    # Setting up flask sessions list
    session["chat_log"] = []

    
    # Refresh the finetuned_model history
    # finetuned_model.__init__()

    # Also delete uploaded files
    folder = 'Web_App/contexts/'
    for filename in os.listdir(folder):
        if filename != ".gitignore":
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print('Failed to delete %s. Reason: %s' % (file_path, e))

    return render_template("home.html")

@app.route("/about")
def about():
    """
    About page, description of us, APL, our project.
    Explain the usecases
    """
    return render_template("about.html")

@app.route("/how_to")
def how_to():
    """
    How to use the different features of our web app.
    """
    return render_template("how_to.html")

@app.route("/new_entry/<mode>/<entry>", methods=["POST"])
def new_entry(mode, entry):
    """
    This is when the user asks FALCON a new question, and enters it.
    The question should be saved as a json file, displayed in the 
    chat log, and then sent to the LLM.

    """
    print(f"This is the new_entry: {entry}")
    # global chat_store
    # session["chat_log"] = chat_store

    if request.method == "POST":
            print("mode: "+mode)
            # Converting hex to string
            ls = []
            for i in entry.split(","):
                hex_string = i
                bytes_object = bytes.fromhex(hex_string)
                ascii_string = bytes_object.decode("ASCII")
                ls.append(ascii_string)
            output = "".join(ls)
    
            # Check Document or Normal Mode
            if mode == "normal":
                # Normal, use finetuned_model model
                response = finetuned_model.run(str(output), session["chat_log"])

                # Store entry in session
                session["chat_log"].append(f"Human: {str(output)}\n")
                session["chat_log"].append(f"AI: {str(response)}\n")

                # print("session stuff:",session["chat_log"])

                return response

            elif mode == "document":
                # Document mode, use vectordb_model
                return vectordb_model.predict(str(output))
            
            # elif mode == "hmt":
            #     # Kenny was doing something with this before.
            #     return hmt_model.predict(str(output))
            
@app.route("/upload_file", methods=["POST"])
def upload_file():
    if request.method == "POST":
        f = request.files['context_file']
        f.save(os.sep.join(["Web_App", "contexts",f.filename]))


        # TODO: FOR JUSTIN - Convert pdf file to string using ur function
        # and send it to the finetuned_model model as context

        # empty return with 204 code, means its good
        x = read_pdf(f"Web_App/contexts/{f.filename}") # - content of uploaded as string TODO: implement 
        # print(x)
        return '', 204

if __name__ == "__main__":
    app.run(debug=False, use_reloader = False, host="0.0.0.0", port=8000) # Set debug = True for live changes in development
