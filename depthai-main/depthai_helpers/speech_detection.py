

# import speech_recognition as sr
# from playsound import playsound

# r = sr.Recognizer()

# with sr.Microphone() as source:
# # read the audio data from the default microphone
#     # audio_data = r.record(source, duration=5)
#     print("talk...")
#     audio_data = r.listen(source)
#     print("Time over, thanks")
#     # print("Recognizing... speech ")
#     # convert speech to text
#     # text = r.recognize_google(audio_data)
#     # print(text)
#     try:
#         # using google speech recognition
#         print("Text: "+r.recognize_google(audio_data))
#     except:
#          print("Sorry, I did not get that")


# import speech_recognition as sr
# import pyttsx3
# # Initialize recognizer class (for recognizing the speech)

# r = sr.Recognizer()
# engine = pyttsx3.init()
# # Reading Microphone as source
# # listening the speech and store in audio_text variable

# with sr.Microphone() as source:
#     text = "Talk"
#     engine.say(text)
#     engine.runAndWait()
#     audio_text = r.record(source, duration=10)
#     text = "Time over thanks"
#     engine.say(text)
#     engine.runAndWait()
# # recoginize_() method will throw a request error if the API is unreachable, hence using exception handling
    
#     try:
#         # using google speech recognition
#         print("Text: "+r.recognize_google(audio_text), type(r.recognize_google(audio_text)))
#     except:
#          print("Sorry, I did not get that")




# import subprocess
# text = '"Hello world"'
# subprocess.call('echo '+text+'|festival --tts', shell=True)


# import speake3
# engine = speake3.Speake()
# engine.set('voice', 'en')
# engine.set('speed', '107')
# engine.set('pitch', '99')
# engine.say("Hello world!") #String to be spoken
# engine.talkback()


# >>>>>>>>>>>>>>>>>> online translator >>>>>>>>>>>>>>>>>>


from googletrans import Translator, constants
import gtts
from playsound import playsound
from unidecode import unidecode
translator = Translator()


translation = translator.translate("right left right hello right", dest="fr")
print(f"{translation.origin} ({translation.src}) --> {translation.text} ({translation.dest})")
print("text ", unidecode(translation.text))
textToSpeech = translation.text
tts = gtts.gTTS(text=textToSpeech) #lang='kn'/"bn"
tts.save("speech.mp3")
# tts.write_to_fp(mp3fIO) 
playsound("speech.mp3")