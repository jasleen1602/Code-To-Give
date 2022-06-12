from Bot import ChatBot as bot
import subprocess
import threading
from flask import Flask, render_template, request, json, jsonify
import os
from translate import Translator
from googletrans import Translator
from langdetect import detect

translator = Translator()

bot = bot.ChatBot.getBot()
usertext = []
botresponse = []

#InputLanguageChoice = StringVar()
#TranslateLanguageChoice = StringVar()
#LanguageChoices = {'Hindi','English','Kannad','Gujrati'}
#InputLanguageChoice.set('English')
#TranslateLanguageChoice.set('Hindi')

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('chat.html')

@app.route("/ask", methods=['POST'])
def ask():
    n = 0
    txt = (request.form['messageText'])
    lang = detect(txt)
    print(lang)
    translated_text = translator.translate(txt)
    message = translated_text.text
    print(message)
    usertext.append(message)

    if(message == 'yes' or message=='YES' or message=='Yes'):
        message+=', '
        message+=botresponse[len(botresponse)-1]
        
    elif(message == 'no' or message=='NO' or message=='No'):
        message+=', '
        message+=botresponse[len(botresponse)-1]
    
    while True:
        if message == "":
            continue

        else:
            bot_response = str(bot.response(message))
            botresponse.append(bot_response)
            n=len(botresponse)
            translated_response = translator.translate(bot_response, dest=lang)
            response = translated_response.text
            
            return jsonify({'status':'OK','answer':response, 'lang':lang})


#@app.route("/ask", methods=['POST'])
#def ask():
#    message = request.form['messageText']
#   ans = bot.get_response(message)
#   print(ans)
#   while True:
#       if message == "quit":
#           exit()
#       else:
#           class Object:
#               def __init__(self, status=None, answer=None):
#                    self.status = status
#                    self.answer = answer
#
#                def toJSON(self):
#                    return json.dumps(self, default=lambda o: o.__dict__, 
#            sort_keys=True, indent=4)
#            data = Object('OK', ans)
#            return data.toJSON()


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

