from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import Optional
from fuzzywuzzy import process
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import json
import os

app = FastAPI()

# تحميل البيانات من JSON
def load_data(json_path: str):
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

data = load_data("exam.json")

# تحميل نموذج BERT
model_name = "deepset/bert-base-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# البحث عن إجابة
def find_answer(question: str, qa_list):
    questions = {qa["سؤال"]: (qa["إجابة"], qa.get("ملف_الصوت", None)) for qa in qa_list}
    match, score = process.extractOne(question, questions.keys())

    if score > 70:
        return questions[match]  # (الإجابة، ملف الصوت)
    return None, None

# بيانات الطلب
class QuestionRequest(BaseModel):
    character_name: str
    question: str

# نقطة النهاية الأساسية
@app.post("/ask")
def ask_question(req: QuestionRequest):
    characters = {char["اسم"]: char for char in data["شخصيات"]}
    
    if req.character_name not in characters:
        raise HTTPException(status_code=404, detail="الشخصية غير موجودة")
    
    character = characters[req.character_name]
    answer, audio_file = find_answer(req.question, character["أسئلة_وأجوبة"])

    if not answer:
        answer = character.get("رد_غير_مفهوم", {}).get("إجابة", "عفوًا، لم أفهم سؤالك.")
        audio_file = character.get("رد_غير_مفهوم", {}).get("ملف_الصوت", None)

    return {
        "question": req.question,
        "answer": answer,
        "audio_file": audio_file
    }
