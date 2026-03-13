from flask import Flask, request, jsonify
from flask_cors import CORS
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
from pythainlp.util import normalize
import re

app = Flask(__name__)
CORS(app)

model_path = "paomay542/wangchanberta-base-att-spm-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# รายชื่อคำอันตราย (อัปเดตตรงนี้ได้ตลอดโดยไม่ต้องเทรนใหม่)
DANGER_KEYWORDS = [
    "ทางรัฐ", "ดิจิทัลวอลเล็ต", "สอท.", "พัสดุตกค้าง", "ค้างชำระ", 
    "ตรวจสอบสิทธิ์", "อัปเดตข้อมูล", "ระงับบัญชี", "ยืนยันตัวตน", "คืนเงิน","แจก","แจกฟรี","สมัครด่วน"
]

def clean_text(text):
    text = " ".join(text.split())
    text = normalize(text)
    # เก็บจุด (.) ไว้บ้างเพื่อตรวจ URL
    text = re.sub(r'[^\u0E00-\u0E7Fa-zA-Z0-9\s.?:]', '', text) 
    return text

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    raw_text = data.get('text', '')
    if not raw_text:
        return jsonify({'error': 'No text'}), 400

    processed_text = clean_text(raw_text)

    # 1. ประมวลผลด้วย AI
    inputs = tokenizer(processed_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = F.softmax(outputs.logits, dim=-1)

    prob_normal = probs[0][0].item()
    prob_scam = probs[0][1].item()

    # 2. ระบบ Bonus Score (ถ้าเจอคำอันตราย ให้เพิ่มค่าน้ำหนักฝั่ง Scam)
    danger_bonus = 0
    found_words = []
    for word in DANGER_KEYWORDS:
        if word in processed_text:
            danger_bonus += 0.50  # เจอ 1 คำ เพิ่มโอกาสเป็น Scam 50%
            found_words.append(word)

    # คำนวณคะแนนใหม่ (AI Score + Bonus Score)
    final_scam_score = min(prob_scam + danger_bonus, 1.0)
    
    # 3. ตั้ง Threshold ให้เข้มงวด (ถ้าคะแนนรวมเกิน 0.35 ให้เตือนทันที)
    threshold = 0.35
    
    if final_scam_score >= threshold:
        prediction = 1
        # แสดงความมั่นใจโดยรวม
        confidence = final_scam_score * 100
    else:
        prediction = 0
        confidence = prob_normal * 100

    return jsonify({
        'is_scam': True if prediction == 1 else False,
        'confidence': f"{confidence:.2f}",
        'label': 'Smishing' if prediction == 1 else 'Normal',
        'detected_keywords': found_words  # ส่งกลับไปดูว่าเจอคำไหนบ้าง
    })

if __name__ == '__main__':
    app.run(port=5000)