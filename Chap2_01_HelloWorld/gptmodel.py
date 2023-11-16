from transformers import GPT2LMHeadModel, GPT2Tokenizer

def generate_response(prompt):
    # Use a smaller GPT-2 model (e.g., "gpt2-small") for better performance on your machine
    model_name = "gpt2"  # or another model identifier
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)


    # Tokenize the input prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # Generate a response using the model
    output = model.generate(
        input_ids,
        max_length=100,  # Adjust the maximum length of the generated response
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95
    )

    # Decode the generated output and remove special tokens
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    return response

# Example usage:
while True:
    user_input = input("You: ")
    
    # Exit the loop if the user enters "exit" or "quit"
    if user_input.lower() in ["exit", "quit"]:
        break
    
    # Generate a response and print it
    response = generate_response(user_input)
    print("Bot:", response)
