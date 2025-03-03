

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Change the model name to GPT-Neo
MODEL_NAME = "EleutherAI/gpt-neo-2.7B"

# Load the GPT-Neo model and tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Set the model to evaluation mode
model.eval()

# Function to chat with the AI
def chat_with_ai(prompt):
    # Add more context or structure to your prompt
    prompt = f"Answer the following question: {prompt}"

    # Tokenize the input prompt
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda" if torch.cuda.is_available() else "cpu")

    # Generate the model's response with adjusted parameters
    with torch.no_grad():  # Disable gradient calculation for inference
        outputs = model.generate(
            inputs['input_ids'], 
            max_new_tokens=150,  # Increase this if you want longer responses
            do_sample=True,      # Allow sampling (more varied responses)
            top_p=0.92,          # Nucleus sampling
            top_k=50,            # Limit the tokens to the top-k probability
            temperature=0.7      # Adjust temperature for more creative responses
        )

    # Decode the output tokens into a human-readable response
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

if __name__ == "__main__":
    print("Welcome to the Hugging Face AI Chat! Type 'exit' or 'quit' to stop.")
    
    while True:
        user_input = input("You: ")
        
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        # Get the AI response
        response = chat_with_ai(user_input)
        
        # Print the response
        if response.strip():  # If there's any response generated
            print("AI:", response)
        else:
            print("AI: Sorry, I couldn't generate a response.")
