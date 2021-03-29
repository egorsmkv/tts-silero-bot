import warnings

warnings.simplefilter('ignore')

import os
import logging
import torch
from torchaudio.backend import soundfile_backend

from os import remove
from os.path import dirname, abspath

import telebot
from utils import init_jit_model, apply_tts, replace_accents

TOKEN = os.environ['TOKEN']

if not TOKEN:
    print('You must set the TOKEN environment variable')
    exit(1)

SYMBOLS = '_~абвгдеёжзийклмнопрстуфхцчшщъыьэюя +.,!?…:;–'
SAMPLE_RATE = 16000

START_MSG = '''Привет!

Этот бот создан для тестирования Silero TTS: https://github.com/snakers4/silero-models
'''

FIRST_STEP = '''Использовать бот просто:
 
1) Добавьте ударения в свой текст с помощью https://morpher.ru/accentizer/

2) Отправьте текст и ждите аудиосообщение
'''

device = torch.device('cpu')

jit_model = dirname(__file__) + '/model/v1_natasha_16000.jit'
model = init_jit_model(jit_model, device=device)

bot = telebot.TeleBot(TOKEN, parse_mode=None)


@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, START_MSG)
    bot.reply_to(message, FIRST_STEP)


@bot.message_handler(func=lambda message: True)
def process_voice_message(message):
    if not message.text:
        bot.reply_to(message, 'А где текст??!')
        return

    # normalize the input
    # text_normalized = do_norm(message.text)
    text_normalized = replace_accents(message.text)
    bot.reply_to(message, 'Нормализованный текст:')
    bot.reply_to(message, text_normalized)

    # do the synthesizing
    audios = apply_tts(texts=[text_normalized],
                       model=model,
                       sample_rate=SAMPLE_RATE,
                       symbols=SYMBOLS,
                       device=device)

    s = 0
    for n, audio_tensor in enumerate(audios):
        # form the filename
        filename = dirname(abspath('__file__')) + f'/files/file_{n}.wav'

        # save to the disk
        soundfile_backend.save(filename, audio_tensor, SAMPLE_RATE)

        # send back to the user
        audio = open(filename, 'rb')
        bot.send_voice(message.chat.id, audio)

        # remove WAV file
        remove(filename)

        # increment the counter
        s = s + 1

    if s == 0:
        bot.reply_to(message, "Ничего не смог синтезировать :(")


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    bot.polling()
