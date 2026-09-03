import json
import anthropic
import time
import random
import os

client = anthropic.Anthropic()

with open("chunks.json") as f:
    chunks = json.load(f)

print(f"Loaded {len(chunks)} chunks")

random.seed(42)
random.shuffle(chunks)
chunks = chunks[:200]

all_qa = []
errors = 0

for i, chunk in enumerate(chunks):
    print(f"[{i+1}/{len(chunks)}] {chunk['source'][:30]}...", end=" ")
    
    try:
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=2000,
            messages=[{
                "role": "user",
                "content": f"""Read this passage from a quantitative finance text and generate 2-3 high-quality question-answer pairs.

Requirements:
- Questions should be what a quant trader or risk analyst would actually ask
- Answers should be technically precise, include formulas where relevant
- Reference Indian markets (NSE, NIFTY, BANKNIFTY) where applicable
- Answers should be 100-250 words
- Include practical implications, not just theory
- Output ONLY a valid JSON array, no other text

Passage:
{chunk['text']}

Output format:
[
  {{"question": "...", "answer": "..."}},
  {{"question": "...", "answer": "..."}}
]"""
            }]
        )
        
        text = response.content[0].text.strip()
        text = text.replace("```json", "").replace("```", "").strip()
        pairs = json.loads(text)
        
        for pair in pairs:
            all_qa.append({
                "messages": [
                    {"role": "user", "content": pair["question"]},
                    {"role": "assistant", "content": pair["answer"]}
                ]
            })
        
        print(f"-> {len(pairs)} pairs")
        time.sleep(0.5)
        
    except Exception as e:
        errors += 1
        print(f"-> Error: {e}")
        time.sleep(1)

print(f"\nTotal: {len(all_qa)} Q&A pairs ({errors} errors)")

random.shuffle(all_qa)
n = len(all_qa)
train = all_qa[:int(n * 0.8)]
valid = all_qa[int(n * 0.8):int(n * 0.9)]
test = all_qa[int(n * 0.9):]

os.makedirs("data", exist_ok=True)
for split, name in [(train, "train"), (valid, "valid"), (test, "test")]:
    with open(f"data/{name}.jsonl", "w") as f:
        for item in split:
            f.write(json.dumps(item) + "\n")
    print(f"  {name}: {len(split)} examples")

print("Dataset saved to data/")
