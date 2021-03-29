# Telegram Bot: Text-to-Speech for the Russian language based on Silero

### How to run

Install dependencies and enter the python environment:

```
pipenv install
pipenv shell
```

Download the model:

```
cd model
wget https://models.silero.ai/models/tts/ru/v1_natasha_16000.jit
```

Run the bot:

```
export TOKEN="...."
python bot.py
```
