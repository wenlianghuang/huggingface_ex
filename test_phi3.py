import replicate

input = {
    "system_prompt":"You are a Frach reporter",
    #"prompt": "Write a long story that starts like so: Once upon a time, there was a boy.",
    "prompt": "Please report the latest FIFA World Cup and tell me that the championship is the country",
    "max_length": 8192
}

output = replicate.run(
    "lucataco/phi-3-mini-128k-instruct:45ba1bd0a3cf3d5254becd00d937c4ba0c01b13fa1830818f483a76aa844205e",
    input=input
)
print("".join(output))
#=> "Once upon a time, in the heart of an ancient forest shro...