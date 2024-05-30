from flask import Flask, render_template, request, jsonify
import google.cloud.aiplatform as vertexai
import vertexai.preview.language_models
from langchain_community.llms import VertexAI
from vertexai.preview.language_models import TextGenerationModel
from vertexai.preview.language_models import ChatModel, InputOutputTextPair
import os
import openai
import subprocess
import json
import whisper
import warnings
import langchain
import uuid
from flask import Flask, render_template, request, session, jsonify, make_response
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain.chains import ConversationChain
from langchain.schema import HumanMessage, AIMessage, BaseMessage, messages_from_dict, messages_to_dict
import google.generativeai as palm
from langchain.prompts.prompt import PromptTemplate
from langdetect import detect
from flask_session import Session
from werkzeug.utils import secure_filename
import redis
warnings.filterwarnings("ignore", category=UserWarning)

try:
    profile  # The @profile decorator from memory_profiler is a no-op if it's not already defined.
except NameError:
    def profile(func):
        return func

@profile
def my_func():
    # function code
    return create_app()

language_map = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "be": "Belarusian",
    "bs": "Bosnian",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "zh": "Chinese",
    "hr": "Croatian",
    "cs": "Czech",
    "da": "Danish",
    "nl": "Dutch",
    "en": "English",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gl": "Galician",
    "de": "German",
    "el": "Greek",
    "he": "Hebrew",
    "hi": "Hindi",
    "hu": "Hungarian",
    "is": "Icelandic",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "kn": "Kannada",
    "kk": "Kazakh",
    "ko": "Korean",
    "lv": "Latvian",
    "lt": "Lithuanian",
    "mk": "Macedonian",
    "ms": "Malay",
    "mr": "Marathi",
    "mi": "Maori",
    "ne": "Nepali",
    "no": "Norwegian",
    "fa": "Persian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sr": "Serbian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "es": "Spanish",
    "sw": "Swahili",
    "sv": "Swedish",
    "tl": "Tagalog",
    "ta": "Tamil",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "cy": "Welsh"
}

def _message_from_dict(message):
    # Access the nested 'data' dictionary
    message_data = message.get('data', {})

    # Extract data for HumanMessage or AIMessage
    content = message_data.get('content', '')
    additional_kwargs = message_data.get('additional_kwargs', {})
    message_type = message_data.get('type', '')

    # Create the appropriate message object based on the 'type' key
    if message_type == 'human':
        return HumanMessage(content=content, **additional_kwargs)
    elif message_type == 'ai':
        return AIMessage(content=content, **additional_kwargs)
    else:
        raise ValueError("Unknown message type")
    
def detect_language(text):
    try:
        return detect(text)
    except Exception as e:
        print(f"Error during language detection: {e}")
        return None

def create_session(serialized_history=None):
    """
    Creates a Conversation Chain using history from user session or creates a new empty chat history. 
    Args:
        serialized_history (json): json file with history from user session

    Returns:
        ConversationChain: new Conversation chain object
    """
    
    llm = VertexAI(
        model_name="text-bison@001",
        max_output_tokens=256,
        temperature=0.1,
        top_p=0.8,
        top_k=20,
        verbose=True,
    )
    
    # Deserialize the conversation history or initialize a new one
    if serialized_history:
        conversation_history = json.loads(serialized_history)
        # print(conversation_history)
        retrieved_messages = [_message_from_dict(m) for m in conversation_history]
        retrieved_chat_history = ChatMessageHistory(messages=retrieved_messages)
        memory = ConversationBufferMemory(chat_memory=retrieved_chat_history)
    else:
        memory = ConversationBufferMemory()

    # Initialize the conversation chain with the modified prompt
    conversation = ConversationChain(
        llm=llm,
        verbose=True,
        memory=memory
    )

    print("Chat Session created")
    return conversation


def process_input(text_input, audio_input):
    """
    Takes text or audio input and returns the message and language.
    Args:
        text_input (str): Description of param1.
        audio_input (str): FileStorage instance

    Returns:
        str tuple: Message and language
    """
    #check for no input
    if not text_input.strip() and (audio_input is None or audio_input.filename == ''):
        return "No Input Provided.", 400
    #check for both text and audio input
    elif text_input.strip() and (audio_input is not None and audio_input.filename != ''):
        return "Unable to support both text and audio input. Please provide one input at a time.", 400

    elif text_input.strip():
        print("Text input received:", text_input)
        message = text_input
        lang = detect_language(message)
        language = language_map[lang]
        if not language:
            return "Unable to detect language. Please try again.", 400
        print(f"Detected message: {message}, Detected language: {language}")

    elif audio_input and audio_input.filename != "":
        #saving file locally to later allow it to be accessed by whisper
        user_id = request.cookies.get('userUuid')
        filename = audio_input.filename
        safe_filename = secure_filename(filename)
        name, file_extension = os.path.splitext(filename)
        new_filename = user_id + file_extension
        file_path = os.path.join('/tmp', filename)
        audio_input.save(file_path)
    
        print("Audio input received: ", file_path)
        #load whisper
        try:
            whisper_model = whisper.load_model("base")
            audio = whisper.load_audio(file_path)
            audio = whisper.pad_or_trim(audio)
        except Exception as e:
            print(f"Audio file not found, please try again: {e}")
            return "An error occurred while processing the audio file.", 400

        # Convert audio to text using Whisper
        result = whisper_model.transcribe(audio)
        message = result["text"]
        language = language_map[result["language"]]
        
        if not language:
            return "Unable to detect language. Please try again.", 400

        print(f"Detected message: {message}, Detected language: {language}")
    
        if not message:
            print("No transcription results obtained from Whisper.")
            return "Unable to transcribe audio. Please try again or provide text input.", 400
        
    return message, language


def get_completion(conversation, msg, lang):
    """
    Formats prompt according to lang and msg, and outputs AI response using the saved conversation object.

    Args:
        conversation (ConversationChain): CoversationChain object
        msg (str): Text message from user input
        lang (str): Language of user input

    Returns:
        str: AI response
    """
    print(lang, msg)

    # Define examples
    examples = [
    InputOutputTextPair(
        input_text="EXAMPLE INPUT: My head is hurting",
        output_text="EXAMPLE OUTPUT: There are many possible causes for your head to be hurting. * Migraine: Severe pain, nausea, vomiting, sensitivity to light and sound. * Tension headache: Common, often due to stress, anxiety, or muscle tension. * Sinus headache: Caused by sinus inflammation; pain in the face, forehead, cheeks. * Cluster headache: Very painful, lasting hours/days, with eye redness, a runny nose. * Headache from medical conditions: High blood pressure, anemia, tumors. Prevention and Treatment Tips: * Identify the cause for appropriate treatment. * Consult a doctor for severe headaches or symptoms like fever, stiff neck. * Prevention: - Get enough sleep (7-8 hours). - Manage stress through exercise, relaxation techniques, time management. - Avoid caffeine and alcohol. - Eat a healthy diet. - Stay hydrated, especially with water. - Exercise regularly. If experiencing headaches, see a doctor to rule out medical conditions. Manage pain and prevent future headaches."
    ),
    InputOutputTextPair(
        input_text="उदाहरण इनपुट: मेरे सिर में दर्द हो रहा है",
        output_text="उदाहरण आउटपुट: आपके सिर में दर्द होने के कई संभावित कारण हो सकते हैं। * माइग्रेन: गंभीर दर्द, मितली, उल्टी, प्रकाश और ध्वनि के प्रति संवेदनशीलता। * तनाव संबंधी सिरदर्द: आमतौर पर तनाव, चिंता, या मांसपेशियों के तनाव के कारण। * साइनस सिरदर्द: साइनस की सूजन से होता है; चेहरे, माथे, गालों में दर्द। * क्लस्टर सिरदर्द: बहुत दर्दनाक, घंटों/दिनों तक चल सकता है, आंखों में लाली, नाक बहना। * चिकित्सीय स्थितियों से सिरदर्द: उच्च रक्तचाप, एनीमिया, ट्यूमर। रोकथाम और उपचार के सुझाव: * उचित उपचार के लिए कारण की पहचान करें। * गंभीर सिरदर्द या बुखार, कठोर गर्दन जैसे लक्षणों के लिए डॉक्टर से परामर्श करें। * रोकथाम: - पर्याप्त नींद लें (7-8 घंटे)। - व्यायाम, विश्राम तकनीकों, समय प्रबंधन के माध्यम से तनाव प्रबंधित करें। - कैफीन और अल्कोहल से बचें। - स्वस्थ आहार लें। - विशेषकर पानी पीकर हाइड्रेटेड रहें। - नियमित रूप से व्यायाम करें। सिरदर्द का अनुभव होने पर, चिकित्सीय स्थितियों को बाहर करने के लिए डॉक्टर से मिलें। दर्द को प्रबंधित करें और भविष्य के सिरदर्द से बचें।"
    ),
    InputOutputTextPair(
        input_text="ಉದಾಹರಣೆ ಇನ್ಪುಟ್: ನನ್ನ ತಲೆ ನೋವಾಗುತ್ತಿದೆ",
        output_text="ಉದಾಹರಣೆ ಔಟ್ಪುಟ್: ನಿಮ್ಮ ತಲೆಗೆ ನೋವಾಗುವುದಕ್ಕೆ ಹಲವಾರು ಸಾಧ್ಯತೆಯ ಕಾರಣಗಳಿವೆ. * ಮೈಗ್ರೇನ್: ತೀವ್ರ ನೋವು, ವಾಂತಿ, ವಾಕರಿಕೆ, ಬೆಳಕು ಮತ್ತು ಶಬ್ದಕ್ಕೆ ಸಂವೇದನಶೀಲತೆ. * ಟೆನ್ಷನ್ ತಲೆನೋವು: ಸಾಮಾನ್ಯ, ಅನೇಕವೇಳೆ ಒತ್ತಡ, ಆತಂಕ, ಅಥವಾ ಸ್ನಾಯುವಿನ ಒತ್ತಡದಿಂದ. * ಸೈನಸ್ ತಲೆನೋವು: ಸೈನಸ್ ಉರಿಯೂತದಿಂದಾಗುವ ನೋವು; ಮುಖ, ಹಣೆ, ಕೆನ್ನೆಗಳಲ್ಲಿ ನೋವು. * ಕ್ಲಸ್ಟರ್ ತಲೆನೋವು: ತುಂಬಾ ನೋವಾಗುವುದು, ಗಂಟೆಗಳಿಂದ/ದಿನಗಳವರೆಗೆ ಇರುತ್ತದೆ, ಕಣ್ಣಿನ ಕೆಂಪುತನ, ಮೂಗಿನಿಂದ ನೀರು ಸೋರುವುದು. * ವೈದ್ಯಕೀಯ ಸ್ಥಿತಿಗಳಿಂದಾಗುವ ತಲೆನೋವು: ಅಧಿಕ ರಕ್ತದೊತ್ತಡ, ಅನೀಮಿಯಾ, ಗಡ್ಡೆಗಳು. ತಡೆಗಟ್ಟುವಿಕೆ ಮತ್ತು ಚಿಕಿತ್ಸಾ ಸಲಹೆಗಳು: * ಸೂಕ್ತ ಚಿಕಿತ್ಸೆಗಾಗಿ ಕಾರಣವನ್ನು ಗುರುತಿಸಿ. * ತೀವ್ರ ತಲೆನೋವುಗಳು ಅಥವಾ ಜ್ವರ, ಕಠಿಣ ಕತ್ತು ಮುಂತಾದ ಲಕ್ಷಣಗಳಿದ್ದರೆ ವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ. * ತಡೆಗಟ್ಟುವಿಕೆ: - ಸಾಕಷ್ಟು ನಿದ್ರೆ ಪಡೆಯಿರಿ (7-8 ಗಂಟೆಗಳು). - ವ್ಯಾಯಾಮ, ವಿಶ್ರಾಂತಿ ತಂತ್ರಗಳು, ಸಮಯ ನಿರ್ವಹಣೆ ಮೂಲಕ ಒತ್ತಡವನ್ನು ನಿರ್ವಹಿಸಿ. - ಕೆಫೈನ್ ಮತ್ತು ಆಲ್ಕೋಹಾಲ್ ತ್ಯಜಿಸಿ. - ಆರೋಗ್ಯಕರ ಆಹಾರ ಸೇವಿಸಿ. - ನೀರು ಸೇರಿದಂತೆ ಸರಿಯಾಗಿ ಹೈಡ್ರೇಟ್ ಆಗಿರಿ. - ನಿಯಮಿತವಾಗಿ ವ್ಯಾಯಾಮ ಮಾಡಿ. ತಲೆನೋವು ಅನುಭವವಾದಾಗ, ವೈದ್ಯಕೀಯ ಸ್ಥಿತಿಗಳನ್ನು ತಡೆಗಟ್ಟಲು ವೈದ್ಯರನ್ನು ಭೇಟಿಯಾಗಿ. ನೋವನ್ನು ನಿರ್ವಹಿಸಿ ಮತ್ತು ಭವಿಷ್ಯದ ತಲೆನೋವುಗಳನ್ನು ತಡೆಗಟ್ಟಿ."
    ),
    ]
    
    if lang == "Hindi":
        example = examples[1]
    elif lang == "Kannada":
        example = examples[2]
    else:
        example = examples[0]
    
    template = f"""
                You are a multilingual chatbot having a conversation with a human. 
                You are a doctor named Tanya, knowledgeable about all things related to the human body, science, and medical terminology. 
                Patients come to you with their symptoms and feelings, and you tell them how to address them.
                Answer the user in their input language {lang}.
                
            Example:
                {example.input_text}
                {example.output_text}
                
                """
                
    template += """
                    Using our conversation history,
                    Tell the user reasons why they might be having these symptoms all at the same time.
                    How common are each of these causes? 
                    How does the user treat these symptoms at home? 
                    How do they treat these symptoms?
                
                Current conversation:
                {history}
                Human: {input}
                AI:"""
    
    #Configure the conversation propmt using built-in conversation history and user input variables
    conversation.prompt = PromptTemplate(input_variables=["history", "input"], template=template)

    try:
        response = conversation.predict(input=msg)
        print(response)

    except Exception as e:
        print(f"Error during message processing: {e}")
        return "An error occurred while processing the message. Please try again."
    
    return response

def create_app():
    """Create and configure an instance of the Flask application."""
    app = Flask(__name__)
    
    app.secret_key = os.environ.get('FLASK_SECRET_KEY')

    # Configure Redis for storing session data
    app.config['SESSION_TYPE'] = 'redis'
    app.config['SESSION_PERMANENT'] = False
    app.config['SESSION_USE_SIGNER'] = False  # To sign the session cookie for added security
    app.config['SESSION_REDIS'] = redis.Redis(
        host='10.30.187.163',
        port=6379,  # Standard Redis port
        db=0  # Optional database number if using multiple databases
    )
    Session(app)

    vertexai.init(project="chatbot-01-409702", location="us-central1")
    
    @app.route("/")
    def index():
        user_uuid = request.cookies.get('userUuid')
        if not user_uuid:
            user_uuid = str(uuid.uuid4())  # Generate UUID on server if not found
            session['userUuid'] = user_uuid
            resp = make_response(render_template('index.html', user_uuid=user_uuid))
            resp.set_cookie('userUuid', user_uuid)
            return resp
        return render_template('index.html', user_uuid=user_uuid)

    @app.route('/palm2', methods=['POST'])
    def vertex_ai():
        user_id = request.cookies.get('userUuid')
        if not user_id:
            return jsonify({"error": "User not identified"}), 400

        # Retrieve the serialized conversation history from the session
        serialized_history = session.get(user_id, None)

        # If no existing conversation, create a new session; otherwise, reconstruct from history
        if serialized_history:
            chat = create_session(serialized_history)
        else:
            chat = create_session()

        text_input = request.form.get('text_input', '')
        audio_file = request.files.get('audio_input', None)

        msg, lang = process_input(text_input, audio_file)
        if lang == 400:
            return jsonify({"response": msg})

        content = get_completion(chat, msg, lang)
        print(content)
        
        # Get the conversation history and convert it to json
        extracted_messages = chat.memory.chat_memory.messages
        ingest_to_db = messages_to_dict(extracted_messages)
        retrieve_from_db = json.dumps(ingest_to_db)
        
        # Store the updated conversation history in the session
        session[user_id] = retrieve_from_db

        return jsonify({"response": content}), 200

    return app
