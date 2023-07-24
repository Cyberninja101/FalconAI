from nemoguardrails import LLMRails, RailsConfig

config = RailsConfig.from_path("./conf")
app = LLMRails(config)

new_message = app.generate(messages=[{
    "role": "user",
    "content": "Hello! What can you do for me?"
}])