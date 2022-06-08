from chatterbot import ChatBot
from chatterbot.trainers import ListTrainer
import threading
from flask import Flask, render_template, request, json
import os

bot = ChatBot("BuzzWomen Bot")
convo = [
    'hello',
    'hi there!',
    'what is your name ?',
    'My name is Buzz Women Bot , i am created by Team B3',
    'how are you ?',
    'I am doing great these days',
    'thank you',
    'In which city you live ?',
    'I live in Karnataka',
    'In which language you talk?',
    'I can talk in many languages. You can change the language to Hindi or Kannad as per your comfort'
]

trainer = ListTrainer(bot)
trainer.train(convo)

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('chat.html')


@app.route("/ask", methods=['POST'])
def ask():
    message = request.form['messageText']
    ans = bot.get_response(message)
    print(ans)
    while True:
        if message == "quit":
            exit()
        else:
            class Object:
                def __init__(self, status=None, answer=None):
                    self.status = status
                    self.answer = answer

                def toJSON(self):
                    return json.dumps(self, default=lambda o: o.__dict__, 
            sort_keys=True, indent=4)
            data = Object('OK', ans)
            return data.toJSON()


if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=True)

