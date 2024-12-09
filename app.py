import streamlit as st
from transformers import GPT2LMHeadModel, GPT2Tokenizer

@st.cache_resource
def load_model():
    model = GPT2LMHeadModel.from_pretrained("finetuned-distilgpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("finetuned-distilgpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer

model, tokenizer = load_model()

def chat_with_model(query):
    inputs = tokenizer.encode(query, return_tensors="pt", padding=True, truncation=True, max_length=512)
    outputs = model.generate(
        inputs,
        max_length=150,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=1.0,
        pad_token_id=tokenizer.pad_token_id,
    )
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

st.title("Chat with Akshay")
st.text("Fine-tuned GPT-2 for interactive conversations about me.")

user_input = st.text_input("You:", placeholder="Type your message here...")
if user_input:
    response = chat_with_model(user_input)
    st.text_area("GPT-2 as Akshay:", response, height=200)
