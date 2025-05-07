import json
from fuzzywuzzy import process
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI()

# تحميل بيانات الشخصيات من JSON
def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)

# تحميل نموذج BERT للإجابة عن الأسئلة
model_name = "deepset/bert-base-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# تحميل البيانات عند بدء التطبيق
data = load_data("exam.json")
characters = {char["اسم"]: char for char in data["شخصيات"]}

# البحث عن إجابة باستخدام المطابقة التقريبية
def find_answer(input_question, qa_list):
    questions = {qa["سؤال"]: qa["إجابة"] for qa in qa_list}
    match, score = process.extractOne(input_question, questions.keys())
    
    if score > 70:
        return questions[match]
    return None

# نموذج البيانات للطلب
class QuestionRequest(BaseModel):
    character_name: str
    question: str

# نقطة النهاية للحصول على الإجابة
@app.post("/get_answer")
async def get_answer(request: QuestionRequest):
    character_name = request.character_name
    user_question = request.question

    if character_name not in characters:
        raise HTTPException(status_code=404, detail="الشخصية غير موجودة")

    character = characters[character_name]
    default_response = character.get("رد_غير_مفهوم", {}).get("إجابة", "عفوًا، لم أفهم سؤالك.")

    # البحث عن إجابة
    answer = find_answer(user_question, character["أسئلة_وأجوبة"])
    if not answer:
        answer = default_response

    return {"answer": answer}