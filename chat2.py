import json
from fuzzywuzzy import process
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, pipeline
import streamlit as st
import os

# تحميل بيانات الشخصيات من JSON
def load_data(json_file):
    with open(json_file, 'r', encoding='utf-8') as file:
        return json.load(file)

# تحميل نموذج BERT للإجابة عن الأسئلة
model_name = "deepset/bert-base-uncased-squad2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# البحث عن إجابة باستخدام المطابقة التقريبية
def find_answer(input_question, qa_list):
    questions = {qa["سؤال"]: (qa["إجابة"], qa.get("ملف_الصوت", None)) for qa in qa_list}
    match, score = process.extractOne(input_question, questions.keys())

    if score > 70:
        return questions[match]  # إرجاع الإجابة وملف الصوت إذا كان التشابه عالٍ
    return None, None

# تشغيل الملفات الصوتية
def play_audio(audio_path):
    if audio_path and os.path.exists(audio_path):
        st.audio(audio_path, format="audio/mp3")

# التطبيق الأساسي
def main():
    data = load_data("exam.json after editing.json")
    st.set_page_config(page_title="الشات بوت التفاعلي", page_icon="🤖", layout="wide")

    col1, col2 = st.columns([1, 3])

    # قائمة الشخصيات
    with col1:
        st.markdown("### 👥 قائمة الشخصيات")
        characters = {char["اسم"]: char for char in data["شخصيات"]}
        selected_character = st.radio("اختر شخصية", list(characters.keys()), index=0)

    # عرض معلومات الشخصية
    with col2:
        st.markdown(f"## 🤖 الدردشة مع {selected_character}")

        character = characters[selected_character]
        st.markdown(f"📝 {character['تعريف']}")

        # تحميل الرد غير المفهوم من ملف JSON (بما في ذلك ملف الصوت)
        default_response_text = character.get("رد_غير_مفهوم", {}).get("إجابة", "عفوًا، لم أفهم سؤالك.")
        default_audio_file = character.get("رد_غير_مفهوم", {}).get("ملف_الصوت", None)

        # جلسة المحادثة
        if "chat_history" not in st.session_state:
            st.session_state["chat_history"] = []

        user_input = st.text_input("💬 أرسل رسالة:")

        if st.button("إرسال"):
            if user_input:
                st.session_state["chat_history"].append({"role": "user", "text": user_input})

                # البحث عن إجابة في الأسئلة المخزنة
                answer, audio_file = find_answer(user_input, character["أسئلة_وأجوبة"])

                # إذا لم يتم العثور على إجابة، استخدم الرد الافتراضي مع الصوت الافتراضي (إن وجد)
                if not answer:
                    answer = default_response_text
                    audio_file = default_audio_file

                st.session_state["chat_history"].append({"role": "bot", "text": answer, "audio": audio_file})

        # عرض المحادثة السابقة
        for message in st.session_state["chat_history"]:
            role = "👤 المستخدم" if message["role"] == "user" else "🤖 البوت"
            st.markdown(f'**{role}**: {message["text"]}')

            # تشغيل الصوت إذا كان متوفرًا
            if message["role"] == "bot" and message.get("audio"):
                play_audio(message["audio"])

if __name__ == "__main__":
    main()
