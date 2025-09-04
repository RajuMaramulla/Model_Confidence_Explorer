import streamlit as st
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import pandas as pd


st.set_page_config(page_title="Log Probabilities Explorer", layout="wide")
st.title("🔍 Log Probabilities & Confidence Explorer")


text_input = st.text_area("Enter your text:", "The moon is made of cheese.")

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")
model.eval()


inputs = tokenizer(text_input, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs, labels=inputs["input_ids"])

logits = outputs.logits
log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
confidences = torch.exp(log_probs)

token_ids = inputs["input_ids"][0]
tokens = tokenizer.convert_ids_to_tokens(token_ids)

token_data = []
for i, token_id in enumerate(token_ids):
    token = tokens[i]
    log_prob = log_probs[0, i, token_id].item()
    confidence = confidences[0, i, token_id].item()
    token_data.append({
        "Token": token,
        "Log Probability": round(log_prob, 4),
        "Confidence": round(confidence, 4),
        "Hallucination": confidence < 0.3
    })


st.subheader("📊 Token-Level Analysis")
df = pd.DataFrame(token_data)

def highlight_row(row):
    if row["Hallucination"]:
        return ['background-color: #ffcccc'] * len(row)
    elif row["Confidence"] > 0.8:
        return ['background-color: #ccffcc'] * len(row)
    else:
        return ['background-color: #ffffcc'] * len(row)

st.dataframe(df.style.apply(highlight_row, axis=1))


st.subheader("🎮 Hallucination Challenge")
if st.button("Reveal Hallucinations"):
    st.write("Tokens with confidence < 0.3 are flagged as hallucinations.")
    st.dataframe(df[df["Hallucination"] == True])
else:
    st.write("Can you guess which tokens the model is least confident about?")
