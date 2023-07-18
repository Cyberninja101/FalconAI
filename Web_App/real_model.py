# Use a pipeline as a high-level helper
# Use a pipeline as a high-level helper
from transformers import pipeline, Conversation

class model():
    """
    This is a hugging face conversational model that can remember
    what you/it said.
    """
    def __init__(self):
        self.pipe = pipeline("conversational", model="microsoft/DialoGPT-medium")

        # pipe = pipeline("conversational", model="kitbear444/DialoGPT-small-kit")
        self.conversation = Conversation()

    def run(self, text):
        # Adding what user inputs to the history
        self.conversation.add_user_input(text)
        response = self.pipe(self.conversation)
        print("Response:", response.generated_responses[0])

        # Adding what model returns to the history
        self.conversation.append_response(response.generated_responses[0]) # TODO: this line is not working
        return response.generated_responses[0]