from pypdf import PdfReader

reader = PdfReader("sleepbook.pdf")

text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

with open("sleepbook.txt", "w", encoding="utf-8") as f:
    f.write(text)

print(f"Extracted {len(reader.pages)} pages to sleepbook.txt")