from chatterbot import ChatBot
import subprocess
from chatterbot.trainers import ListTrainer
import threading
from flask import Flask, render_template, request, json, jsonify
import os

bot = ChatBot("BuzzWomen Bot")
convo= [
    'Hi', 
    'hello',
    'Can you help me to earn money ?',
    'yes, I m there to help you',
    'what is your name ?',
    'My name is Buzz Women Bot , I am created by Team B3',
    'How are you?',
    'I am doing great these days, You say ',
    'I am also doing great',
    'Great, so how can I help you ?',
    'What is Buzz Women ?',
    'Buzz Women is the global movement by and for women.We bring transformation within reach. And enable women to ignite their personal and collective power.',
    'Okay , I get it.',
    'Are you currently working?',
    'No, but I want to earn money by doing some small business.',
    'Do you have any idea in any field or you want to explore some suggestions?',
    'no,I dont have any idea',
    'So,tell me what are your interests/skills?',
    'I am good at tailoring .',
    'Do you have any knowledge of fabric and style?',
    'Yes,I have knowledge of fabrics.'
    'Ok,great. Do you also know how to stich, zigzag or staright?',
    'Yes, I know stitching',
    'Okay , Do you know how to make measurements?',
    'Yes, I am aware of measurements',
    'Okay great, how many clothes have you stitched till now approx ?',
    'I have stitched 5 clothes',
    'okay, great! So, How many tailors do you have in your locality/village ?',
    'We have count tailors in our village',
    'So, as you know you already have this amount of tailors in your locality, Cant you think of helping these tailors only in any way',
    'Umm, Yes yes you are right. I can save their time by providing them with the raw material.',
    'Yes, you are on the right track. Now, you can proceed on this idea.'
    'Yeah, Thankyou !',
    'Welcome, any other help do you want?',
    'No,Thanks for helping !',
    'Okay, bye bye',
    'Okay, so I will help you.',
    'Yes, but I want to earn more money.',
    'How much do you earn?'
    
]

trainer = ListTrainer(bot)
trainer.train(convo)

app = Flask(__name__)

@app.route("/")
def hello():
    return render_template('chat.html')

@app.route("/ask", methods=['POST'])
def ask():
    message = (request.form['messageText'])


    while True:
        if message == "":
            continue

        else:
            bot_response =   str(bot.get_response(message))
            return jsonify({'status':'OK','answer':bot_response})


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

