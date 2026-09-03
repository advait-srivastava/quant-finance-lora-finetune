import fitz
import os
import json

papers_dir = "./papers"
chunks = []

for filename in os.listdir(papers_dir):
    if not filename.endswith(".pdf"):
        continue
    
    filepath = os.path.join(papers_dir, filename)
    print(f"Processing: {filename}")
    
    try:
        doc = fitz.open(filepath)
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        
        full_text = full_text.replace("\n\n\n", "\n\n")
        
        words = full_text.split()
        chunk_size = 250
        overlap = 50
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk = " ".join(words[i:i + chunk_size])
            if len(chunk) > 200:
                chunks.append({
                    "source": filename,
                    "text": chunk
                })
    except Exception as e:
        print(f"  Error: {e}")

print(f"\nExtracted {len(chunks)} chunks from PDFs")

with open("chunks.json", "w") as f:
    json.dump(chunks, f, indent=2)

print("Saved to chunks.json")
