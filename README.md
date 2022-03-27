# whatsappbot_nlp
A WhatsApp chatbot built using Natural Language Processing to understand the user texts and reply back with a relevant message.

The chatbot is first trained using [Natural Language Toolkit](https://www.nltk.org/) to understand the messages depending upon various tags and randomly choose the appropriate message to reply.
We use flask and connect it with twilio in order to receive and send messages through [Twilio](https://www.twilio.com/). The chatbot is then connected to the flask app to use the chatbot on whatsapp to reply to the user.

## Python libraries used:
- [NLTK](https://www.nltk.org/)
- [Pytorch (torch)](https://pytorch.org/)
- [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- [Numpy](https://numpy.org/)
- [json](https://docs.python.org/3/library/json.html)
- [Twilio](https://www.twilio.com/docs/libraries/python)

## Requirements:
- Setup a twilio account
- Download ngrok to use the localhost server as a global server and use it in the twilio sandbox settings
- You can use gunicorn to deploy the flask app online to use the trained model and then set the twilio sandbox settings
