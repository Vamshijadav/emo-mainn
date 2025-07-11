import pyttsx3
import speech_recognition as sr
import datetime
import pyjokes
import pywhatkit as kit
import webbrowser
import pyautogui
import wikipedia
import os
from ultralytics import YOLO
import cv2
import cvzone
import math
import calendar
import subprocess
import requests
import google.generativeai as genai

# Initialize components
cm = open("name.txt", "r")
listener = sr.Recognizer()
engine = pyttsx3.init()
voices = engine.getProperty("voices")
engine.setProperty("voice", voices[1].id)
engine.setProperty("rate", 170)
name = cm.read()
cm.close()

def chandramuki():
    """Object detection function that detects up to 5 objects and then stops"""
    cap = cv2.VideoCapture(0)  # For WebCam
    cap.set(3, 1280)  # Width
    cap.set(4, 720)   # Height
    
    model = YOLO("../Yolo-Weights/yolov8x.pt")
    
    detected_objects = []
    detection_count = 0
    max_detections = 5
    
    talk("Starting object detection. I will detect up to 5 objects.")
    
    while detection_count < max_detections:
        success, img = cap.read()
        if not success:
            break
            
        results = model(img, stream=True)
        
        for r in results:
            boxes = r.boxes
            if boxes is not None:
                for box in boxes:
                    # Bounding Box
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    w, h = (x2-x1), (y2-y1)
                    
                    # Confidence
                    conf = math.ceil((box.conf[0]*100))
                    
                    # Class Names
                    cls = int(box.cls[0])
                    class_name = model.names[cls]  # Get class name from model
                    
                    # Only proceed if confidence is above threshold
                    if conf > 50:  # 50% confidence threshold
                        cvzone.cornerRect(img, (x1, y1, w, h))
                        cvzone.putTextRect(img, f'{class_name} {conf}%', 
                                         (max(0, x1), max(0, y1 + 35)), 
                                         scale=1, thickness=1)
                        
                        # Add to detected objects if not already detected
                        if class_name not in detected_objects:
                            detected_objects.append(class_name)
                            talk(f"Detected {class_name}")
                            detection_count += 1
                            
                            if detection_count >= max_detections:
                                break
                
                if detection_count >= max_detections:
                    break
        
        cv2.imshow("Object Detection", img)
        
        # Break on 'q' key press or max detections reached
        if cv2.waitKey(1) & 0xFF == ord('q') or detection_count >= max_detections:
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    talk(f"Object detection completed. Detected {len(detected_objects)} different objects.")
    if detected_objects:
        talk(f"Objects found: {', '.join(detected_objects)}")

def ai(prompt):
    """AI function using Gemini API with improved error handling"""
    try:
        # Your Gemini AI API key (Consider using environment variables for security)
        api = "AIzaSyCxujWomJ070Q3Om26UnlcUO7t4mJSx9-0"
        
        genai.configure(api_key=api)

        # Set up the model
        generation_config = {
           "temperature": 0.7,  # Reduced for more consistent responses
            "top_p": 1,
            "top_k": 1,
            "max_output_tokens": 1024,  # Reduced for faster responses
        }

        safety_settings = [
            {
                "category": "HARM_CATEGORY_HARASSMENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_HATE_SPEECH",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_MEDIUM_AND_ABOVE"
            },
        ]

        model = genai.GenerativeModel(model_name="gemini-2.0-flash",
                              generation_config=generation_config,
                              safety_settings=safety_settings)
        
        # Add context to make responses more conversational and concise
        enhanced_prompt = f"Please provide a brief, conversational response to: {prompt}. Keep it under 100 words and suitable for voice response."
        
        convo = model.start_chat(history=[])
        convo.send_message(enhanced_prompt)
        
        response = convo.last.text
        if response:
            print(f"AI Response: {response}")
            talk(response)
        else:
            talk("I couldn't generate a response. Let me search that for you instead.")
            website = 'https://www.google.com/search?q=' + prompt.replace(' ', '+')
            webbrowser.open_new(website)
        
    except Exception as e:
        print(f"AI Error: {e}")
        talk("Sorry, AI is not available right now. Let me search that for you.")
        website = 'https://www.google.com/search?q=' + prompt.replace(' ', '+')
        webbrowser.open_new(website)

def talk(text):
    """Text to speech function"""
    engine.say(text)
    engine.runAndWait()

def take_command():
    """Speech recognition function"""
    try:
        with sr.Microphone() as source:
            print("Listening.....")
            voice = listener.listen(source, phrase_time_limit=5)
            command = listener.recognize_google(voice)
            command = command.lower()
            print(f"You said: {command}")
            return command
    except sr.UnknownValueError:
        print("Could not understand audio")
        return ""
    except sr.RequestError as e:
        print(f"Error with speech recognition: {e}")
        return ""
    except Exception as e:
        print(f"Error: {e}")
        return ""

def greetings():
    """Greeting function based on time"""
    current_time = datetime.datetime.now()
    hour = current_time.hour
    if 1 <= hour < 12:
        talk(f"Good morning Sandeep! I am {name}, your personal assistant")
    elif 12 <= hour < 16:
        talk(f"Good afternoon Sandeep! I am {name}, your personal assistant")
    elif 16 <= hour < 20:
        talk(f"Good evening Sandeep! I am {name}, your personal assistant")
    else:
        talk(f"Hello Sandeep! I am {name}, your personal assistant")

def run_assis():
    """Main assistant function"""
    command = take_command()
    
    if not command:
        talk("Sorry, I didn't catch that. Please try again.")
        return
        
    print(f"Processing command: {command}")  # Debug line
        
    if "hello" in command or "hi" in command:
        talk("Hello Sandeep! How may I help you today?")
        
    elif "goodbye" in command or "bye" in command:
        talk("Goodbye!")
        exit()
        
    elif "joke" in command:
        talk(pyjokes.get_joke())
        
    elif "song" in command or "video" in command:
        talk(f"Playing {command}")
        kit.playonyt(command)
        
    elif "time" in command:
        time = datetime.datetime.now().strftime("%I:%M %p")
        print(time)
        talk(f"The current time is {time}")
        
    elif "open" in command:
        app_name = command.replace("open", "").strip()
        pyautogui.press("super")
        pyautogui.typewrite(app_name)
        pyautogui.sleep(1)
        pyautogui.press("enter")
        talk(f"Opening {app_name}")
        
    elif "close" in command:
        pyautogui.hotkey("alt", "f4")
        talk("Closing application")
        
    elif "camera" in command or "take a photo" in command:
        try:
            camera_port = 0
            camera = cv2.VideoCapture(camera_port)
            return_value, image = camera.read()
            if return_value:
                cv2.imwrite("photo.jpg", image)
                talk("Photo taken successfully")
            else:
                talk("Failed to take photo")
            camera.release()
        except Exception as e:
            talk("Error accessing camera")
            
    elif ("who is" in command or "what is" in command or "why" in command or 
          "how to" in command or "explain" in command or "tell me about" in command):
        try:
            # Extract the main query
            query = command
            for phrase in ["who is", "what is", "why", "how to", "explain", "tell me about"]:
                query = query.replace(phrase, "").strip()
            
            print(f"Wikipedia query: {query}")  # Debug line
            info = wikipedia.summary(query, sentences=3)
            print(info)
            talk(info)
        except wikipedia.exceptions.DisambiguationError as e:
            talk(f"Multiple results found for {query}. Please be more specific.")
            # Try AI as fallback
            ai(command)
        except wikipedia.exceptions.PageError:
            talk(f"Sorry, I couldn't find information about {query} on Wikipedia. Let me use AI to help you.")
            ai(command)
        except Exception as e:
            print(f"Wikipedia error: {e}")  # Debug line
            talk("Let me use AI to answer that.")
            ai(command)
            
    elif "remember that" in command:
        remember_message = command.replace("remember that", "").strip()
        talk(f"You told me to remember that {remember_message}")
        try:
            with open("remember.txt", "w") as remember_file:
                remember_file.write(remember_message)
        except Exception as e:
            talk("Error saving memory")
            
    elif "what do you remember" in command:
        try:
            with open("remember.txt", "r") as remember_file:
                memory = remember_file.read()
                if memory:
                    talk(f"You told me to remember that {memory}")
                else:
                    talk("I don't have anything to remember")
        except FileNotFoundError:
            talk("I don't have anything to remember")
            
    elif "weather" in command:
        api_key = "ee6b438edb3cb44ce721340ed29abfe7"
        base_url = "https://api.openweathermap.org/data/2.5/weather?"
        talk("What is the city name?")
        city_name = take_command()
        if city_name:
            complete_url = base_url + "appid=" + api_key + "&q=" + city_name
            try:
                response = requests.get(complete_url)
                x = response.json()
                if x["cod"] != "404":
                    main_data = x["main"]
                    current_temperature = main_data["temp"]
                    current_humidity = main_data["humidity"]
                    weather_description = x["weather"][0]["description"]
                    
                    # Convert Kelvin to Celsius
                    temp_celsius = current_temperature - 273.15
                    
                    talk(f"Temperature: {temp_celsius:.1f} degrees Celsius")
                    talk(f"Humidity: {current_humidity} percent")
                    talk(f"Weather description: {weather_description}")
                else:
                    talk("City not found")
            except Exception as e:
                talk("Error getting weather information")
                
    elif "clear file" in command:
        try:
            with open("remember.txt", "w") as file:
                file.write("")
            talk("Memory cleared successfully!")
        except Exception as e:
            talk("Error clearing memory")
            
    elif "shutdown" in command:
        talk("Shutting down the system")
        talk("3... 2... 1...")
        subprocess.call(['shutdown', '/s', '/f'])
        
    elif "restart" in command:
        talk("Restarting the system")
        talk("3... 2... 1...")
        os.system("shutdown /r /t 1")
        
    elif "news" in command:
        webbrowser.open_new_tab("https://timesofindia.indiatimes.com/india")
        talk("Here are some latest news!")
        
    elif "change your name" in command:
        new_name = command.replace("change your name to", "").strip()
        talk(f"You told me to change my name to {new_name}")
        try:
            with open("name.txt", "w") as name_file:
                name_file.write(new_name)
            talk("Name changed successfully")
        except Exception as e:
            talk("Error changing name")
            
    elif "what is your name" in command or "can i know your name" in command:
        try:
            with open("name.txt", "r") as name_file:
                current_name = name_file.read()
                talk(f"My name is {current_name}")
        except Exception as e:
            talk("I don't know my name")
            
    elif 'how are you' in command:
        talk("I am fine, thank you!")
        
    elif "using artificial intelligence" in command or "ai" in command or "ask ai" in command:
        ai_command = command.replace("using artificial intelligence", "").replace("ai", "").replace("ask", "").strip()
        if ai_command:
            ai(prompt=ai_command)
        else:
            talk("Please tell me what you want to ask AI")
            
    elif ("search" in command or "google" in command or "browse" in command or 
          "find" in command or "look up" in command):
        # Extract search query
        search_terms = ["search", "google", "browse", "find", "look up", "for", "about"]
        text_words = command.split()
        
        print(f"Search command detected: {command}")  # Debug line
        
        # Find the starting point for the query
        query_start = 0
        for i, word in enumerate(text_words):
            if word.lower() in search_terms[:5]:  # Only main search terms
                query_start = i + 1
                # Skip connector words
                while (query_start < len(text_words) and 
                       text_words[query_start].lower() in ['for', 'about', 'on']):
                    query_start += 1
                break
        
        if query_start < len(text_words):
            search_query = ' '.join(text_words[query_start:])
            print(f"Search query extracted: {search_query}")  # Debug line
            website = 'https://www.google.com/search?q=' + search_query.replace(' ', '+')
            talk(f"Searching for {search_query} on Google")
            webbrowser.open_new(website)
        else:
            talk("Please tell me what you want to search for")
        
    elif "detect object" in command or "object detection" in command:
        chandramuki()
        
    elif "pause" in command or "play" in command:
        pyautogui.press("k")
        talk("Done!")
        
    elif "date" in command or "day" in command:
        now = datetime.datetime.now()
        date = datetime.datetime.today()
        weekday = calendar.day_name[date.weekday()]
        month_num = now.month
        day_num = now.day
        
        month_names = ['January', 'February', 'March', 'April', 'May', 'June', 
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        ordinal_numbers = ['1st', '2nd', '3rd', '4th', '5th', '6th', '7th', '8th', '9th', '10th',
                          '11th', '12th', '13th', '14th', '15th', '16th', '17th', '18th', '19th', '20th',
                          '21st', '22nd', '23rd', '24th', '25th', '26th', '27th', '28th', '29th', '30th', '31st']
        
        talk(f'Today is {weekday}, {month_names[month_num - 1]} the {ordinal_numbers[day_num - 1]}')
        
    elif "full screen" in command:
        pyautogui.press("f")
        talk("Done!")
        
    elif "message" in command:
        talk("Please provide the phone number.")
        number = take_command()
        if number:
            talk("Please say the message.")
            message = take_command()
            if message:
                try:
                    kit.sendwhatmsg_instantly(f'+91{number}', message)
                    pyautogui.sleep(10)
                    talk("Message sent")
                except Exception as e:
                    talk("Error sending message")
                    
    # Handle general queries - decide between AI and browser search
    elif any(keyword in command for keyword in ["tell me", "can you", "help me", "advice", "opinion", "think", "suggest", "recommend"]):
        # Use AI for conversational queries
        print(f"Using AI for conversational query: {command}")  # Debug line
        ai(prompt=command)
        
    elif any(keyword in command for keyword in ["latest", "current", "today", "news", "recent", "update", "price", "stock"]):
        # Use browser for current information
        print(f"Using browser for current info: {command}")  # Debug line
        website = 'https://www.google.com/search?q=' + command.replace(' ', '+')
        talk(f"Looking up current information about {command}")
        webbrowser.open_new(website)
        
    elif command and len(command.strip()) > 0:
        # For other general queries, try to determine the best approach
        print(f"Processing general query: {command}")  # Debug line
        
        # Check if it's a simple factual query that AI can handle
        factual_keywords = ["calculate", "convert", "definition", "meaning", "history", "biography"]
        if any(keyword in command for keyword in factual_keywords):
            print("Using AI for factual query")  # Debug line
            ai(prompt=command)
        else:
            # For everything else, use browser search
            print("Using browser for general search")  # Debug line
            website = 'https://www.google.com/search?q=' + command.replace(' ', '+')
            talk(f"Searching for {command}")
            webbrowser.open_new(website)
    else:
        print("No command recognized or empty command")  # Debug line
        talk("Sorry, I can't hear you clearly. Please try again.")

def authenticate_user(known_voiceprint):
    """User authentication function"""
    microphone = sr.Microphone()

    with microphone as source:
        print("Please say the passphrase for authentication:")
        talk("Please say the passphrase for authentication:")
        listener.adjust_for_ambient_noise(source)
        try:
            audio_data = listener.listen(source, phrase_time_limit=5)
        except Exception as e:
            talk("Error with microphone")
            return False

    try:
        spoken_text = listener.recognize_google(audio_data)
        print("You said:", spoken_text)
        
        if spoken_text.lower() == known_voiceprint.lower():
            talk("Authentication successful!")
            talk("Now I am ready for your commands, Sandeep")
            return True
        else:
            talk("Authentication failed. Voice not recognized.")
            return False
    except sr.UnknownValueError:
        talk("Sorry, I couldn't understand what you said.")
        return False
    except sr.RequestError as e:
        talk("Error with speech recognition service")
        return False

# Main execution
if __name__ == "__main__":
    greetings()
    
    # Authenticate user
    if authenticate_user("sandeep"):
        while True:
            try:
                run_assis()
            except KeyboardInterrupt:
                talk("Goodbye!")
                break
            except Exception as e:
                print(f"Error in main loop: {e}")
                talk("An error occurred. Continuing...")
                continue