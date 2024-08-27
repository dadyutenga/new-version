import streamlit as st
from openai import OpenAI
import dotenv
import os
from PIL import Image
from audio_recorder_streamlit import audio_recorder
import base64
from io import BytesIO
import google.generativeai as genai
import random
import anthropic
import mimetypes
import json
import os
from datetime import datetime

dotenv.load_dotenv()


anthropic_models = [
    "claude-3-5-sonnet-20240620"
]

google_models = [
    "gemini-1.5-flash",
    "gemini-1.5-pro",
]

openai_models = [
    "gpt-4o", 
    "gpt-4-turbo", 
    "gpt-3.5-turbo-16k", 
    "gpt-4", 
    "gpt-4-32k",
]


# Function to convert the messages format from OpenAI and Streamlit to Gemini
def messages_to_gemini(messages):
    gemini_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            gemini_message = gemini_messages[-1]
        else:
            gemini_message = {
                "role": "model" if message["role"] == "assistant" else "user",
                "parts": [],
            }

        for content in message["content"]:
            if content["type"] == "text":
                gemini_message["parts"].append(content["text"])
            elif content["type"] == "image_url":
                gemini_message["parts"].append(base64_to_image(content["image_url"]["url"]))
            elif content["type"] == "video_file":
                gemini_message["parts"].append(genai.upload_file(content["video_file"]))
            elif content["type"] == "audio_file":
                gemini_message["parts"].append(genai.upload_file(content["audio_file"]))
            elif content["type"] == "file":
                gemini_message["parts"].append(content["file_content"])

        if prev_role != message["role"]:
            gemini_messages.append(gemini_message)

        prev_role = message["role"]
        
    return gemini_messages


# Function to convert the messages format from OpenAI and Streamlit to Anthropic (the only difference is in the image messages)
def messages_to_anthropic(messages):
    anthropic_messages = []
    prev_role = None
    for message in messages:
        if prev_role and (prev_role == message["role"]):
            anthropic_message = anthropic_messages[-1]
        else:
            anthropic_message = {
                "role": message["role"] ,
                "content": [],
            }
        
        for content in message["content"]:
            if content["type"] == "image_url":
                anthropic_message["content"].append(
                    {
                        "type": "image",
                        "source":{   
                            "type": "base64",
                            "media_type": content["image_url"]["url"].split(";")[0].split(":")[1],
                            "data": content["image_url"]["url"].split(",")[1]
                        }
                    }
                )
            elif content["type"] == "file":
                anthropic_message["content"].append(
                    {
                        "type": "text",
                        "text": f"File content:\n\n{content['file_content']}"
                    }
                )
            else:
                anthropic_message["content"].append(content)

        if prev_role != message["role"]:
            anthropic_messages.append(anthropic_message)

        prev_role = message["role"]
        
    return anthropic_messages


# Function to query and stream the response from the LLM
def stream_llm_response(model_params, model_type="openai", api_key=None):
    response_message = ""

    if model_type == "openai":
        client = OpenAI(api_key=api_key)
        for chunk in client.chat.completions.create(
            model=model_params["model"] if "model" in model_params else "gpt-4o",
            messages=st.session_state.messages,
            temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
            max_tokens=4096,
            stream=True,
        ):
            chunk_text = chunk.choices[0].delta.content or ""
            response_message += chunk_text
            yield chunk_text

    elif model_type == "google":
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name = model_params["model"],
            generation_config={
                "temperature": model_params["temperature"] if "temperature" in model_params else 0.3,
            }
        )
        gemini_messages = messages_to_gemini(st.session_state.messages)

        try:
            response = model.generate_content(contents=gemini_messages, stream=False)
            
            if response.candidates:
                for candidate in response.candidates:
                    if candidate.content and candidate.content.parts:
                        for part in candidate.content.parts:
                            if part.text:
                                response_message += part.text
                                yield part.text
                    else:
                        st.warning("Response candidate does not contain any content.")
            else:
                st.warning("No valid response candidates received from the API.")
            
            if response.prompt_feedback:
                st.info(f"Prompt feedback: {response.prompt_feedback}")
            
            if not response_message:
                safety_ratings = response.candidates[0].safety_ratings if response.candidates else []
                st.warning(f"The response may have been blocked due to safety concerns. Safety ratings: {safety_ratings}")
                yield "The response was blocked due to safety concerns. Please try rephrasing your query."

        except Exception as e:
            st.error(f"Error in Google API: {str(e)}")
            yield f"Error: {str(e)}"

    elif model_type == "anthropic":
        client = anthropic.Anthropic(api_key=api_key)
        with client.messages.stream(
            model=model_params["model"] if "model" in model_params else "claude-3-5-sonnet-20240620",
            messages=messages_to_anthropic(st.session_state.messages),
            temperature=model_params["temperature"] if "temperature" in model_params else 0.3,
            max_tokens=4096,
        ) as stream:
            for text in stream.text_stream:
                response_message += text
                yield text

    st.session_state.messages.append({
        "role": "assistant", 
        "content": [
            {
                "type": "text",
                "text": response_message,
            }
        ]})


# Function to convert file to base64
def get_image_base64(image_raw):
    buffered = BytesIO()
    image_raw.save(buffered, format=image_raw.format)
    img_byte = buffered.getvalue()

    return base64.b64encode(img_byte).decode('utf-8')

def file_to_base64(file):
    with open(file, "rb") as f:

        return base64.b64encode(f.read())

def base64_to_image(base64_string):
    base64_string = base64_string.split(",")[1]
    
    return Image.open(BytesIO(base64.b64decode(base64_string)))

def save_conversation(messages, filename):
    # Ensure the 'conversations' directory exists
    os.makedirs('conversations', exist_ok=True)
    filepath = os.path.join('conversations', filename)
    with open(filepath, 'w') as f:
        json.dump(messages, f)

def load_conversation(filename):
    filepath = os.path.join('conversations', filename)
    if os.path.exists(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    return []

def load_conversation(filename):
    if os.path.exists(filename):
        with open(filename, 'r') as f:
            return json.load(f)
    return []

def main():

    # --- Page Config ---
    st.set_page_config(
        page_title="DadI Custom Chatbot",
        page_icon="🤖",
        layout="centered",
        initial_sidebar_state="expanded",
    )

    # --- Header ---
    st.html("""<h1 style="text-align: center; color: #6ca395;">🤖 <i>The DadI custom chatbot</i> 💬</h1>""")

    # --- Side Bar ---
    with st.sidebar:
        cols_keys = st.columns(2)
        with cols_keys[0]:
            default_openai_api_key = os.getenv("OPENAI_API_KEY") if os.getenv("OPENAI_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
            with st.popover("🔐 OpenAI"):
                openai_api_key = st.text_input("Introduce your OpenAI API Key (https://platform.openai.com/)", value=default_openai_api_key, type="password")
        
        with cols_keys[1]:
            default_google_api_key = os.getenv("GOOGLE_API_KEY") if os.getenv("GOOGLE_API_KEY") is not None else ""  # only for development environment, otherwise it should return None
            with st.popover("🔐 Google"):
                google_api_key = st.text_input("Introduce your Google API Key (https://aistudio.google.com/app/apikey)", value=default_google_api_key, type="password")

        default_anthropic_api_key = os.getenv("ANTHROPIC_API_KEY") if os.getenv("ANTHROPIC_API_KEY") is not None else ""
        with st.popover("🔐 Anthropic"):
            anthropic_api_key = st.text_input("Introduce your Anthropic API Key (https://console.anthropic.com/)", value=default_anthropic_api_key, type="password")
    
    # --- Main Content ---
    # Checking if the user has introduced the OpenAI API Key, if not, a warning is displayed
    if (openai_api_key == "" or openai_api_key is None or "sk-" not in openai_api_key) and (google_api_key == "" or google_api_key is None) and (anthropic_api_key == "" or anthropic_api_key is None):
        st.write("#")
        st.warning("⬅️ Please introduce an API Key to continue...")

        
           
    else:
        client = OpenAI(api_key=openai_api_key)

        if "messages" not in st.session_state:
            st.session_state.messages = []

        if "conversations" not in st.session_state:
            st.session_state.conversations = {}

        # Displaying the previous messages if there are any
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                for content in message["content"]:
                    if content["type"] == "text":
                        st.write(content["text"])
                    elif content["type"] == "image_url":      
                        st.image(content["image_url"]["url"])
                    elif content["type"] == "video_file":
                        st.video(content["video_file"])
                    elif content["type"] == "audio_file":
                        st.audio(content["audio_file"])
                    elif content["type"] == "file":
                        st.text(f"File: {content['file_name']} ({content['file_type']})")
                        with st.expander("View file content"):
                            st.code(content['file_content'], language=content['file_type'].split('/')[-1])

        # Side bar model options and inputs
        with st.sidebar:

            st.divider()
            
            available_models = [] + (anthropic_models if anthropic_api_key else []) + (google_models if google_api_key else []) + (openai_models if openai_api_key else [])
            model = st.selectbox("Select a model:", available_models, index=0)
            model_type = None
            if model.startswith("gpt"): model_type = "openai"
            elif model.startswith("gemini"): model_type = "google"
            elif model.startswith("claude"): model_type = "anthropic"
            
            with st.popover("⚙️ Model parameters"):
                model_temp = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.3, step=0.1)

            audio_response = st.toggle("Audio response", value=False)
            if audio_response:
                cols = st.columns(2)
                with cols[0]:
                    tts_voice = st.selectbox("Select a voice:", ["alloy", "echo", "fable", "onyx", "nova", "shimmer"])
                with cols[1]:
                    tts_model = st.selectbox("Select a model:", ["tts-1", "tts-1-hd"], index=1)

            model_params = {
                "model": model,
                "temperature": model_temp,
            }

            def reset_conversation():
                if "messages" in st.session_state and len(st.session_state.messages) > 0:
                    st.session_state.pop("messages", None)

            st.button(
                "🗑️ Reset conversation", 
                on_click=reset_conversation,
            )

            st.divider()
            
            # File Upload
            st.write("### **📄 Add a file:**")
            
            def add_file_to_messages():
                if st.session_state.uploaded_file:
                    file = st.session_state.uploaded_file
                    file_content = ""
                    mime_type, _ = mimetypes.guess_type(file.name)
                    
                    if mime_type == "application/pdf":
                        pdf_reader = PdfReader(io.BytesIO(file.getvalue()))
                        for page in pdf_reader.pages:
                            file_content += page.extract_text() + "\n"
                    elif mime_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
                        doc = Document(io.BytesIO(file.getvalue()))
                        for para in doc.paragraphs:
                            file_content += para.text + "\n"
                    else:
                        try:
                            file_content = file.getvalue().decode("utf-8")
                        except UnicodeDecodeError:
                            file_content = "Binary file content (not displayed)"
                    
                    st.session_state.messages.append(
                        {
                            "role": "user", 
                            "content": [{
                                "type": "file",
                                "file_name": file.name,
                                "file_type": mime_type or "application/octet-stream",
                                "file_content": file_content
                            }]
                        }
                    )

            st.file_uploader(
                "Upload a file:", 
                type=["txt", "pdf", "docx"],  # Limit to these file types
                accept_multiple_files=False,
                key="uploaded_file",
                on_change=add_file_to_messages,
            )

            # Audio Upload
            st.write("#")
            st.write(f"### **🎤 Add an audio{' (Speech To Text)' if model_type == 'openai' else ''}:**")

            audio_prompt = None
            audio_file_added = False
            if "prev_speech_hash" not in st.session_state:
                st.session_state.prev_speech_hash = None

            speech_input = audio_recorder("Press to talk:", icon_size="3x", neutral_color="#6ca395", )
            if speech_input and st.session_state.prev_speech_hash != hash(speech_input):
                st.session_state.prev_speech_hash = hash(speech_input)
                if model_type != "google":
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", 
                        file=("audio.wav", speech_input),
                    )

                    audio_prompt = transcript.text

                elif model_type == "google":
                    # save the audio file
                    audio_id = random.randint(100000, 999999)
                    with open(f"audio_{audio_id}.wav", "wb") as f:
                        f.write(speech_input)

                    st.session_state.messages.append(
                        {
                            "role": "user", 
                            "content": [{
                                "type": "audio_file",
                                "audio_file": f"audio_{audio_id}.wav",
                            }]
                        }
                    )

                    audio_file_added = True

            

            st.divider()
            
            # Reset conversation
            if st.button("Start New Conversation"):
                st.session_state.messages = []
                st.success("Started a new conversation.")

        # Chat input
        if prompt := st.chat_input("Hi! Ask me anything...") or audio_prompt or audio_file_added:
            if not audio_file_added:
                st.session_state.messages.append(
                    {
                        "role": "user", 
                        "content": [{
                            "type": "text",
                            "text": prompt or audio_prompt,
                        }]
                    }
                )
                
                # Display the new messages
                with st.chat_message("user"):
                    st.markdown(prompt)

            else:
                # Display the audio file
                with st.chat_message("user"):
                    st.audio(f"audio_{audio_id}.wav")

            with st.chat_message("assistant"):
                model2key = {
                    "openai": openai_api_key,
                    "google": google_api_key,
                    "anthropic": anthropic_api_key,
                }
                st.write_stream(
                    stream_llm_response(
                        model_params=model_params, 
                        model_type=model_type, 
                        api_key=model2key[model_type]
                    )
                )

            # --- Added Audio Response (optional) ---
            if audio_response:
                response =  client.audio.speech.create(
                    model=tts_model,
                    voice=tts_voice,
                    input=st.session_state.messages[-1]["content"][0]["text"],
                )
                audio_base64 = base64.b64encode(response.content).decode('utf-8')
                audio_html = f"""
                <audio controls autoplay>
                    <source src="data:audio/wav;base64,{audio_base64}" type="audio/mp3">
                </audio>
                """
                st.html(audio_html)



if __name__=="__main__":
    main()