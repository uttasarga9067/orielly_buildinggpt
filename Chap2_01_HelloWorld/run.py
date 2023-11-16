import os
import openai


from dotenv import load_dotenv

load_dotenv()

# Make sure the environment variable OPENAI_API_KEY is set.

# Call the openai ChatCompletion endpoint, with th ChatGPT model
response = openai.ChatCompletion.create(
  model="gpt-3.5-turbo",
  messages=[
        {"role": "user", "content": "Hello World!"}
    ]
)

# Extract the response
print(response['choices'][0]['message']['content'])




