import streamlit as st
import cv2
import mediapipe as mp
import pickle
import numpy as np
from google import genai
from google.genai import types
from gtts import gTTS
import io

# --- 1. SETUP & UI STYLING ---
st.set_page_config(page_title="Signly - Live Sign Language Translator", layout="wide", page_icon="🤟")

st.markdown("""
    <style>
    .main { background-color: white; }
    .word-box {
        padding: 20px;
        background: #f9f9f9;
        border-radius: 15px;
        border-left: 10px solid #007bff;
        margin-bottom: 20px;
        font-size: 32px;
        font-weight: bold;
        color: #333;
        min-height: 100px;
        display: flex;
        align-items: center;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# API & Model Loading
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"], http_options=types.HttpOptions(api_version='v1'))

# Load the Holistic Brain
model_dict = pickle.load(open('model.p', 'rb'))
hand_model = model_dict['model']
labels_dict = {i: label for i, label in enumerate(model_dict['labels'])}

# Initialize Holistic
mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_drawing = mp.solutions.drawing_utils

# Session States
if 'word_list' not in st.session_state: st.session_state.word_list = []
if 'run_cam' not in st.session_state: st.session_state.run_cam = False
if 'final_sentence' not in st.session_state: st.session_state.final_sentence = ""

# --- 2. UI LAYOUT ---
st.title("🤟 Signly Pro: Holistic Translator")
col_vid, col_res = st.columns([2, 1])

with col_res:
    st.markdown("### 📝 Live Transcript")
    transcript_placeholder = st.empty()
    
    # Display logic
    display_text = " ".join(st.session_state.word_list) if st.session_state.word_list else "Waiting for signs..."
    transcript_placeholder.markdown(f'<div class="word-box">{display_text}</div>', unsafe_allow_html=True)
    
    threshold = st.slider("Confidence Threshold", 0.40, 1.00, 0.90, 0.01)

    if st.button("✨ Generate Natural Sentence", use_container_width=True):
        if st.session_state.word_list:
            try:
                raw_text = " ".join(st.session_state.word_list)
                prompt = f"You are an ASL to English translator. Translate these glosses into a natural, conversational English sentence. If the user provides a question word (WHAT, HOW), format it as a question. Give only the sentence itself.glosses are:  {raw_text}"
                response = client.models.generate_content(model="gemini-2.5-flash", contents=[prompt])
                st.session_state.final_sentence = response.text
                st.info(f"**AI Sentence:** {st.session_state.final_sentence}")
            except Exception as e:
                st.error(f"AI Error: {e}")

    if st.session_state.final_sentence:
        if st.button("🔊 Speak Sentence", use_container_width=True):
            tts = gTTS(text=st.session_state.final_sentence, lang='en')
            audio_fp = io.BytesIO()
            tts.write_to_fp(audio_fp)
            st.audio(audio_fp.getvalue(), format="audio/mp3", autoplay=True)

    if st.button("🗑️ Clear All", use_container_width=True):
        st.session_state.word_list = []
        st.session_state.final_sentence = ""
        st.rerun()

    st.divider()
    if not st.session_state.run_cam:
        if st.button("🎥 Open Camera", type="primary", use_container_width=True):
            st.session_state.run_cam = True
            st.rerun()
    else:
        if st.button("🛑 Close Camera", use_container_width=True):
            st.session_state.run_cam = False
            st.rerun()

with col_vid:
    frame_placeholder = st.empty()

# --- 3. THE LIVE LOOP ---
if st.session_state.run_cam:
    cap = cv2.VideoCapture(0)
    last_word = ""
    
    while st.session_state.run_cam:
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(rgb_frame)

        # LANDMARK EXTRACTION
        # 1. Pose
        pose = [val for lm in (results.pose_landmarks.landmark if results.pose_landmarks else []) for val in [lm.x, lm.y, lm.z]]
        if not pose: pose = [0.0] * (33 * 3)
        
        # 2. Left Hand
        lh = [val for lm in (results.left_hand_landmarks.landmark if results.left_hand_landmarks else []) for val in [lm.x, lm.y, lm.z]]
        if not lh: lh = [0.0] * (21 * 3)

        # 3. Right Hand
        rh = [val for lm in (results.right_hand_landmarks.landmark if results.right_hand_landmarks else []) for val in [lm.x, lm.y, lm.z]]
        if not rh: rh = [0.0] * (21 * 3)

        # Combine Features
        features = np.array(pose + lh + rh).reshape(1, -1)

        # Prediction
        prediction_proba = hand_model.predict_proba(features)
        max_prob = np.max(prediction_proba)
        predicted_index = np.argmax(prediction_proba)
        current_word = labels_dict[predicted_index]

        # Draw Skeletons
        mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
        mp_drawing.draw_landmarks(frame, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

        # Logic for Transcript
        if max_prob >= threshold:
            color = (0, 255, 0)
            if current_word != last_word:
                st.session_state.word_list.append(current_word)
                last_word = current_word
                new_text = " ".join(st.session_state.word_list)
                transcript_placeholder.markdown(f'<div class="word-box">{new_text}</div>', unsafe_allow_html=True)
        else:
            color = (0, 0, 255)

        cv2.putText(frame, f"{current_word} ({max_prob*100:.1f}%)", (10, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA)
        
        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

    cap.release()