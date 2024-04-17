from transformers import pipeline,Conversation

chatboot = pipeline(model="facebook/blenderbot-400M-distill")
conversation = chatboot("Hi I'm Shaw, how are you?")
print(conversation)