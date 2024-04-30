import replicate



while True:
    user_input = input("You: ")
    if user_input.lower() in ['bye','quit','exit']:
        print("Chatbot: Goodbye")
        break
    inputres = {
        "prompt":"{}".format(user_input),
        "max_length":8192
    }
    output = replicate.run(
        "lucataco/phi-3-mini-128k-instruct:45ba1bd0a3cf3d5254becd00d937c4ba0c01b13fa1830818f483a76aa844205e",
        input=inputres
    )
    #print("Chatboot: ")
    print("".join(output))
    #print("Chatboot: ".join(output))
