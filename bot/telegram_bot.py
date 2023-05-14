import logging
import os
import itertools
import asyncio

import telegram
from telegram import constants
from telegram import Message, MessageEntity, Update, InlineQueryResultArticle, InputTextMessageContent, BotCommand, ChatMember
from telegram.error import RetryAfter, TimedOut
from telegram.ext import ApplicationBuilder, ContextTypes, CommandHandler, MessageHandler, \
    filters, InlineQueryHandler, Application, CallbackContext, MessageHandler
from pydub import AudioSegment
from openai_helper import OpenAIHelper
from usage_tracker import UsageTracker
from telegram import ReplyKeyboardMarkup, KeyboardButton
import io

import random

import warnings
from PIL import Image
from typing import Any, Awaitable
import stability_sdk.interfaces.gooseai.generation.generation_pb2 as generation
from stability_sdk.client import StabilityInference

from stability_sdk import client

from telegram import InputMediaPhoto

import re

from telegram import InputFile

import mimetypes

import imageio

import glob

import asyncio

import types

from io import BytesIO

import cv2

from pathlib import Path

import tempfile

import time

import requests

import moviepy.editor as mp

import logging

from telegram.ext import Updater

from telegram import InlineKeyboardButton, InlineKeyboardMarkup, InputMediaPhoto

from telegram.ext import CommandHandler, ConversationHandler, MessageHandler, filters, CallbackQueryHandler

from concurrent.futures import ThreadPoolExecutor

import time

from telegram import CallbackQuery

logger = logging.getLogger(__name__)

def message_text(message: Message) -> str:
    """
    Returns the text of a message, excluding any bot commands.
    """
    message_text = message.text
    if message_text is None:
        return ''

    for _, text in sorted(message.parse_entities([MessageEntity.BOT_COMMAND]).items(), key=(lambda item: item[0].offset)):
        message_text = message_text.replace(text, '').strip()

    return message_text if len(message_text) > 0 else ''


class ChatGPTTelegramBot:
    """
    Class representing a ChatGPT Telegram Bot.
    """

    def __init__(self, config: dict, openai: OpenAIHelper):
        """
        Initializes the bot with the given configuration and GPT bot object.
        :param config: A dictionary containing the bot configuration
        :param openai: OpenAIHelper object
        """
        self.config = config
        self.openai = openai
        self.commands = [
            BotCommand(command='help', description='Show help message'),
            BotCommand(command='reset', description='Reset the conversation. Optionally pass high-level instructions '
                                                    '(e.g. /reset You are a helpful assistant)'),
            
            BotCommand(command='stable', description='Generate stable from prompt (e.g. /stable cat)'),                                         
            BotCommand(command='image_editor', description='Generate stable from prompt (e.g. /image_editor cat)'),
            BotCommand(command='stable_albom', description='Generate stable from prompt (e.g. /stable_albom cat)'), 
            BotCommand(command='resend', description='Resend the latest message')
        ]
        self.disallowed_message = "Sorry, you are not allowed to use this bot."
        self.budget_limit_message = "Sorry, you have reached your monthly usage limit."
        self.usage = {}
        self.last_message = {}
        self.seeds = {}
        self.use = {}
        self.image_generating = {}
        self.generate_result = {}
        self.responses = {}
        self.text = {}



    async def start(self, update: Update, context):
        button = InlineKeyboardButton("Click me!", callback_data='button')
        keyboard = InlineKeyboardMarkup([[button]])
        help_message = (
            "Wellcome my Dear Friend!" 
            '\n'
            "I'm a bot, Stable Diffusion."
            'I use artificial intelligence to communicate and '
            'generate different types of content.\n'
        )
        chat_id = update.effective_chat.id

        await context.bot.send_photo(chat_id=chat_id, photo=open('bot\DATA\generatsiya-kartinok.png', 'rb'))
       
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=help_message,
            reply_markup=keyboard
        )

    async def help(self, update: Update, context):
        button = InlineKeyboardButton("Click me!", callback_data='button')
        keyboard = InlineKeyboardMarkup([[button]])
        help_message = (
            "I'm a bot, Stable Diffusion. "
            'I use artificial intelligence to communicate and '
            'generate different types of content.\n'
        )
        chat_id = update.effective_chat.id
        self.use[chat_id] = 3
        await context.bot.send_photo(chat_id=chat_id, photo=open('bot\DATA\maxresdefault.png', 'rb'))
       
        
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=help_message,
            reply_markup=keyboard
        )

    async def button_callback(self, update: Update, context):
        query = update.callback_query
        await query.answer()
        chat_id = update.effective_chat.id
        

        if query.data == 'button':
            # выполняем функцию, связанную с кнопкой ""
            await self.menu(update, context)
            self.use[chat_id] = 3
            
        if query.data == 'button1':
            # выполняем функцию, связанную с кнопкой "PROMPT-E"
            self.use[chat_id] = 1
            await self.variants(update, context)

        if query.data == 'button2':
            # выполняем функцию, связанную с кнопкой "STABLE"
            self.use[chat_id] = 2
            await self.menu(update, context)
       
        if query.data == 'button3':
            # выполняем функцию, связанную с кнопкой "STABLE"
            self.use[chat_id] = 1
            await self.menu(update, context)

        if query.data == 'button4':
            # выполняем функцию, связанную с кнопкой "STABLE"
            text = self.responses[chat_id]
            message = f'{text}'
            await self.image_stable_albom1(message, query, context)
           
        if query.data == 'button5':
            # выполняем функцию, связанную с кнопкой "STABLE"
            text = self.text[chat_id]
            message = f'{text}'
            await self.image_stable1(message, query, context)

            
        if query.data == 'button6':
            # выполняем функцию, связанную с кнопкой "STABLE"
            text = self.text[chat_id]
            message = f'{text}'
            await self.image_stable_albom1(message, query, context)

            
        if query.data == 'button7':
            # выполняем функцию, связанную с кнопкой "STABLE"
            text = self.responses[chat_id]
            message = f'{text}'
            await self.image_stable1(message, query, context)

            


    async def variants(self, update: Update, context):

        chat_id = update.message.chat_id
    
        button2 = InlineKeyboardButton("Example TEXT", callback_data='button5')
        button4 = InlineKeyboardButton("PROMPT", callback_data='button4')
        button5 = InlineKeyboardButton("Example PROMPT", callback_data='button7')
        button3 = InlineKeyboardButton("TEXT", callback_data='button6')
        button1 = InlineKeyboardButton("EXIT", callback_data='button')
        markup = InlineKeyboardMarkup([[button2, button4, button5, button3, button1]])
        await context.bot.send_message(chat_id=chat_id, text=f'Result:{self.responses[chat_id]}',)
        await context.bot.send_message(chat_id=chat_id, text='Continue with:', reply_markup=markup)

    async def menu(self, update, context):
        button2 = InlineKeyboardButton("PROPMT only", callback_data='button6')
        button4 = InlineKeyboardButton("One image is STABLE", callback_data='button7')
        markup = InlineKeyboardMarkup([[button2, button4]])

        await context.bot.edit_message_text(
            chat_id=update.callback_query.message.chat_id,
            message_id=update.callback_query.message.message_id,
            text="Brief instructions\n"
            '\n'
            'Write a request to create an image.'
            '\n'
        )
        
        chat_id = update.effective_chat.id
        self.use[chat_id] = 1

    async def image_stable_upscale1(self, to_img, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using stability.ai APIs.
        """
        if not self.config['enable_image_generation'] or not await self.check_allowed_and_within_budget(update, context):
            return

        chat_id = update.effective_chat.id
        message = to_img
        image_query = message_text(update.message)
        if image_query == '':
            await context.bot.send_message(chat_id=chat_id, text='Please provide a prompt! (e.g. /stable cat)')
            return

        logging.info(f'New image generation request received from user {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})')
        await context.bot.send_message(chat_id=chat_id, text='Please wait: Usually generation takes no more than 7 seconds for each image.')
        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        STABILITY_KEY = os.getenv('STABILITY_KEY')
        os.environ['STABILITY_KEY'] = STABILITY_KEY

                
                
        def parse_message(message):
            prompt = ""
            seed = 123467598
            steps = 50
            cfg_scale = 8.0
            size = 1024

            for part in message.split():
                if ":" not in part:
                    prompt += " " + part
                    continue
                key, value = part.split(":")
                if key in ["seed", "sd", "se", "see"]:
                    try:
                        seed = int(value)
                    except ValueError:
                        pass
                elif key in ["cfg_scale", "cfg", "cfg_", "cfg_s", "cf"]:
                    try:
                        cfg_scale = float(value)
                    except ValueError:
                        pass
                elif key in ["steps", "ste", "st", "step"]:
                    try:
                        steps = int(value)
                    except ValueError:
                        pass
                elif ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["size", "s", "si", "siz"]:
                    try:
                        size = int(value)
                    except ValueError:
                        pass


            # Удаляем начальные и конечные пробелы в prompt
            prompt = prompt.strip()
            return prompt, seed, steps, cfg_scale, size
        
        prompt, seed, steps, cfg_scale = parse_message(message)

        stability_api = client.StabilityInference(
            key=os.environ['STABILITY_KEY'], # API Key reference.
            upscale_engine="esrgan-v1-x2plus", # The name of the upscaling model we want to use.
            # Available Upscaling Engines: esrgan-v1-x2plus, stable-diffusion-x4-latent-upscaler 
            verbose=True, # Print debug messages.
        )
        stability_api = StabilityInference(
            key=os.environ['STABILITY_KEY'],  # API Key reference.
            verbose=True,  # Print debug messages.
            engine="stable-diffusion-xl-beta-v2-2-2",  # Set the engine to use for generation.
        )
        
        async def _generate():
            try:
                answers = stability_api.generate(
                    prompt=prompt, 
                    seed=seed, 
                    steps=steps, 
                    cfg_scale=cfg_scale, 
                    width=512, 
                    height=512, 
                    samples=1, 
                    sampler=generation.SAMPLER_K_DPMPP_SDE)
                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.finish_reason == generation.FILTER:
                           warnings.warn(
                               "Your request activated the API's safety filters and could not be processed."
                                "Please modify the prompt and try again.")
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            img_bytes = Image.open(io.BytesIO(artifact.binary))
                            img_bytes.save("imageupscaled" + ".png")

              # Получаем изображение из API
                size = parse_message(message)
                answers = stability_api.upscale(
                init_image=img_bytes,
                width=size, # Optional parameter to specify the desired output width.    
                #prompt=message, # Optional parameter when using `stable-diffusion-x4-latent-upscaler` to specify a prompt to use for the upscaling process.
                #seed=self.seeds[chat_id], # Optional parameter when using `stable-diffusion-x4-latent-upscaler` to specify a seed to use for the upscaling process.
                #steps=20, # Optional parameter when using `stable-diffusion-x4-latent-upscaler` to specify the number of diffusion steps to use for the upscaling process. Defaults to 20 if no value is passed, with a maximum of 50.
                #cfg_scale=7 # Optional parameter when using `stable-diffusion-x4-latent-upscaler` to specify the strength of prompt in use for the upscaling process. Defaults to 7 if no value is passed.
)
                
                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.finish_reason == generation.FILTER:
                           warnings.warn(
                               "Your request activated the API's safety filters and could not be processed."
                                "Please modify the prompt and try again.")
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            big_img = io.BytesIO(artifact.binary)
                            await context.bot.send_photo(chat_id=chat_id, photo=big_img)

                    
                    
            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    text=f'Failed to generate image: {str(e)}',
                    parse_mode=constants.ParseMode.MARKDOWN
                )        
        button_exit = InlineKeyboardButton("EXIT", callback_data="button")
        reply_markup = InlineKeyboardMarkup([[button_exit]])
        await self.wrap_with_indicator(update, context, constants.ChatAction.UPLOAD_PHOTO, _generate)
        await context.bot.send_message(chat_id=chat_id, text="Что бы выйти нажмите на кнопку ниже", reply_markup=reply_markup)
   
    async def image_stable_upscale(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using stability.ai APIs.
        """
        if not self.config['enable_image_generation'] or not await self.check_allowed_and_within_budget(update, context):
            return

        chat_id = update.effective_chat.id
        message = update.message.text
        image_query = message_text(update.message)
        if image_query == '':
            await context.bot.send_message(chat_id=chat_id, text='Please provide a prompt! (e.g. /stable cat)')
            return

        logging.info(f'New image generation request received from user {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})')
        await context.bot.send_message(chat_id=chat_id, text='Please wait: Usually generation takes no more than 7 seconds for each image.')
        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        STABILITY_KEY = os.getenv('STABILITY_KEY')
        os.environ['STABILITY_KEY'] = STABILITY_KEY

                
                
        def parse_message(message):
            prompt = ""
            seed = 123467598
            steps = 50
            cfg_scale = 8.0
            size = 1024

            for part in message.split():
                if ":" not in part:
                    prompt += " " + part
                    continue
                key, value = part.split(":")
                if key in ["seed", "sd", "se", "see"]:
                    try:
                        seed = int(value)
                    except ValueError:
                        pass
                elif key in ["cfg_scale", "cfg", "cfg_", "cfg_s", "cf"]:
                    try:
                        cfg_scale = float(value)
                    except ValueError:
                        pass
                elif key in ["steps", "ste", "st", "step"]:
                    try:
                        steps = int(value)
                    except ValueError:
                        pass
                elif ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["size", "s", "si", "siz"]:
                    try:
                        size = int(value)
                    except ValueError:
                        pass


            # Удаляем начальные и конечные пробелы в prompt
            prompt = prompt.strip()
            return prompt, seed, steps, cfg_scale, size
        
        prompt, seed, steps, cfg_scale = parse_message(message)

        stability_api = client.StabilityInference(
            key=os.environ['STABILITY_KEY'], # API Key reference.
            upscale_engine="esrgan-v1-x2plus", # The name of the upscaling model we want to use.
            # Available Upscaling Engines: esrgan-v1-x2plus, stable-diffusion-x4-latent-upscaler 
            verbose=True, # Print debug messages.
        )
        stability_api = StabilityInference(
            key=os.environ['STABILITY_KEY'],  # API Key reference.
            verbose=True,  # Print debug messages.
            engine="stable-diffusion-xl-beta-v2-2-2",  # Set the engine to use for generation.
        )
        
        async def _generate():
            try:
                answers = stability_api.generate(
                    prompt=prompt, 
                    seed=seed, 
                    steps=steps, 
                    cfg_scale=cfg_scale, 
                    width=512, 
                    height=512, 
                    samples=1, 
                    sampler=generation.SAMPLER_K_DPMPP_SDE)
                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.finish_reason == generation.FILTER:
                           warnings.warn(
                               "Your request activated the API's safety filters and could not be processed."
                                "Please modify the prompt and try again.")
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            img_bytes = Image.open(io.BytesIO(artifact.binary))
                            img_bytes.save("imageupscaled" + ".png")

              # Получаем изображение из API
                size = parse_message(message)
                answers = stability_api.upscale(
                init_image=img_bytes,
                width=size, # Optional parameter to specify the desired output width.    
                #prompt=message, # Optional parameter when using `stable-diffusion-x4-latent-upscaler` to specify a prompt to use for the upscaling process.
                #seed=self.seeds[chat_id], # Optional parameter when using `stable-diffusion-x4-latent-upscaler` to specify a seed to use for the upscaling process.
                #steps=20, # Optional parameter when using `stable-diffusion-x4-latent-upscaler` to specify the number of diffusion steps to use for the upscaling process. Defaults to 20 if no value is passed, with a maximum of 50.
                #cfg_scale=7 # Optional parameter when using `stable-diffusion-x4-latent-upscaler` to specify the strength of prompt in use for the upscaling process. Defaults to 7 if no value is passed.
)
                
                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.finish_reason == generation.FILTER:
                           warnings.warn(
                               "Your request activated the API's safety filters and could not be processed."
                                "Please modify the prompt and try again.")
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            big_img = io.BytesIO(artifact.binary)
                            await context.bot.send_photo(chat_id=chat_id, photo=big_img)

                    
                    
            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    text=f'Failed to generate image: {str(e)}',
                    parse_mode=constants.ParseMode.MARKDOWN
                )
        await self.Awrap_with_indicator(update, context, constants.ChatAction.UPLOAD_PHOTO, _generate)          
        button_exit = InlineKeyboardButton("EXIT", callback_data="button")
        reply_markup = InlineKeyboardMarkup([[button_exit]])
        await self.wrap_with_indicator(update, context, constants.ChatAction.UPLOAD_PHOTO, _generate)
        await context.bot.send_message(chat_id=chat_id, text="Что бы выйти нажмите на кнопку ниже", reply_markup=reply_markup)
 
    async def image_stable(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using stability.ai APIs.
        """
        if not self.config['enable_image_generation'] or not await self.check_allowed_and_within_budget(update, context):
            return

        chat_id = update.effective_chat.id
        message = update.message.text
        image_query = message_text(update.message)
        if image_query == '':
            await context.bot.send_message(chat_id=chat_id, text='Please provide a prompt! (e.g. /stable cat)')
            return

        logging.info(f'New image generation request received from user {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})')
        await context.bot.send_message(chat_id=chat_id, text='Please wait: Usually generation takes no more than 7 seconds for each image.')
        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        STABILITY_KEY = os.getenv('STABILITY_KEY')
        os.environ['STABILITY_KEY'] = STABILITY_KEY

                
        def parse_message(message):
            prompt = "NO"
            seed = 123467598
            steps = 50
            cfg_scale = 8.0

            for part in message.split():
                if ":" not in part:
                    prompt += " " + part
                    continue
                key, value = part.split(":")
                if key in ["seed", "sd", "se", "see"]:
                    try:
                        seed = int(value)
                    except ValueError:
                        pass
                elif key in ["cfg_scale", "cfg", "cfg_", "cfg_s", "cf"]:
                    try:
                        cfg_scale = float(value)
                    except ValueError:
                        pass
                elif key in ["steps", "ste", "st", "step"]:
                    try:
                        steps = int(value)
                    except ValueError:
                        pass

            # Удаляем начальные и конечные пробелы в prompt
            prompt = prompt.strip()
            return prompt, seed, steps, cfg_scale
        
        prompt, seed, steps, cfg_scale = parse_message(message)

        stability_api = StabilityInference(
            key=os.environ['STABILITY_KEY'],  # API Key reference.
            verbose=True,  # Print debug messages.
            engine="stable-diffusion-xl-beta-v2-2-2",  # Set the engine to use for generation.
        )
        async def _generate():
            try:
              # Получаем изображение из API
                answers = stability_api.generate(
                    prompt=prompt, 
                    seed=seed, 
                    steps=steps, 
                    cfg_scale=cfg_scale, 
                    width=512, 
                    height=512, 
                    samples=1, 
                    sampler=generation.SAMPLER_K_DPMPP_SDE)
                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.finish_reason == generation.FILTER:
                           warnings.warn(
                               "Your request activated the API's safety filters and could not be processed."
                                "Please modify the prompt and try again.")
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            img_bytes = io.BytesIO(artifact.binary)
                            await context.bot.send_photo(chat_id=chat_id, photo=img_bytes)

                    
                    
            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(chat_id=chat_id, text=f'{e}')
                await context.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    text=f'Failed to generate image: {str(e)}',
                    parse_mode=constants.ParseMode.MARKDOWN
                )
        await self.Awrap_with_indicator(update, context, constants.ChatAction.UPLOAD_PHOTO, _generate)  

    async def image_stable1(self, message, callback_query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE):
        
        if callback_query and callback_query.message and callback_query.message.from_user:
            logging.info(f'New image generation request received from user {callback_query.message.from_user.name} (id: {callback_query.message.from_user.id})')
        else:
            logging.error("Invalid input: 'from_user' attribute does not exist.")

        if not self.config['enable_image_generation'] or not await self.check_allowed_and_within_budget(callback_query, context):
            return

        chat_id = callback_query.message.chat_id
        await context.bot.send_message(chat_id=chat_id, text='Please wait: Usually generation takes no more than 7 seconds for each image.')
        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        STABILITY_KEY = os.getenv('STABILITY_KEY')
        os.environ['STABILITY_KEY'] = STABILITY_KEY

        def parse_message(message):
            prompt = ""
            seed = 123467598
            steps = 50
            cfg_scale = 8.0

            for part in message.split():
                if ":" not in part:
                    prompt += " " + part
                    continue
                key, value = part.split(":")
                if key in ["seed", "sd", "se", "see"]:
                    try:
                        seed = int(value)
                    except ValueError:
                        pass
                elif key in ["cfg_scale", "cfg", "cfg_", "cfg_s", "cf"]:
                    try:
                        cfg_scale = float(value)
                    except ValueError:
                        pass
                elif key in ["steps", "ste", "st", "step"]:
                    try:
                        steps = int(value)
                    except ValueError:
                        pass

            prompt = prompt.strip()
            return prompt, seed, steps, cfg_scale

        prompt, seed, steps, cfg_scale = parse_message(message)

        stability_api = StabilityInference(
            key=os.environ['STABILITY_KEY'],
            verbose=True,
            engine="stable-diffusion-xl-beta-v2-2-2",
        )

        async def _generate():
            try:
                if self.image_generating:
                    return
                else:
                    self.image_generating[chat_id] = True
                    answers = stability_api.generate(
                        prompt=prompt, 
                        seed=seed, 
                        steps=steps, 
                        cfg_scale=cfg_scale, 
                        width=512, 
                        height=512, 
                        samples=1, 
                        sampler=generation.SAMPLER_K_DPMPP_SDE)
                    for resp in answers:
                        for artifact in resp.artifacts:
                            if artifact.finish_reason == generation.FILTER:
                                warnings.warn(
                                    "Your request activated the API's safety filters and could not be processed."
                                    "Please modify the prompt and try again.")
                            if artifact.type == generation.ARTIFACT_IMAGE:
                                img_bytes = io.BytesIO(artifact.binary)
                                await context.bot.send_photo(chat_id=chat_id, photo=img_bytes)  
                                del self.image_generating[chat_id]
            except Exception as e:
                logging.exception(e)
                del self.image_generating[chat_id]
                await context.bot.send_message(chat_id=chat_id, text=f'{e}')
                await context.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=callback_query.message.message_id,
                    text=f'Failed to generate image: {str(e)}'
                )

        await self.wrap_with_indicator(callback_query, context, constants.ChatAction.UPLOAD_PHOTO, _generate)

        button_exit = InlineKeyboardButton("EXIT", callback_data="button")
        reply_markup = InlineKeyboardMarkup([[button_exit]])

        await context.bot.send_message(chat_id=chat_id, text="Что бы выйти нажмите на кнопку ниже", reply_markup=reply_markup)

    async def image_editor(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using stability.ai APIs.
        """
        if not self.config['enable_image_generation'] or not await self.check_allowed_and_within_budget(update, context):
            return

        chat_id = update.effective_chat.id
        message = update.message.text
        image_query = message_text(update.message)
        if image_query == '':
            await context.bot.send_message(chat_id=chat_id, text='Please provide a prompt! (e.g. /image_editor cat)')
            return

        logging.info(f'New image generation request received from user {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})')
        await context.bot.send_message(chat_id=chat_id, text='Please wait: Usually generation takes no more than 7 seconds for each image.')
        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        STABILITY_KEY = os.getenv('STABILITY_KEY')
        os.environ['STABILITY_KEY'] = STABILITY_KEY

                
        def parse_messag(message):
            seed = 123467598
            steps = 50
            cfg_scale = 8.0

            for part in message.split():
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["seed", "sd", "se", "see"]:
                    try:
                        seed = int(value)
                    except ValueError:
                        pass
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["cfg_scale", "cfg", "cfg_", "cfg_s", "cf"]:
                    try:
                        cfg_scale = int(value)
                    except ValueError:
                        pass
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["steps", "ste", "st", "step"]:
                    try:
                        steps = int(value)
                    except ValueError:
                        pass
            return seed, steps, cfg_scale
        
        seed, steps, cfg_scale = parse_messag(message)

        stability_api = StabilityInference(
            key=os.environ['STABILITY_KEY'],  # API Key reference.
            verbose=True,  # Print debug messages.
            engine="stable-diffusion-xl-beta-v2-2-2",  # Set the engine to use for generation.
        )

        def parse_message(message):
            # находим все скобки в сообщении
            parentheses_regex = r"\(([^(),]+),\s*(-?\d+)\)"
            matches = re.findall(parentheses_regex, message)
            
            # использовать параметры для формирования строки Prompt если найдены скобки
            if len(matches) > 0:
                output_str = f"[generation.Prompt(text='{message}', parameters=generation.PromptParameters(weight=1)), generation.Prompt(text='{matches[0][0]}', parameters=generation.PromptParameters(weight={matches[0][1]}))]"
                return output_str
            else:
                return ''  # возвращаем пустую строку, если нет скобок

        async def _generate():
            try:
                output_str = parse_message(message, chat_id)
              # Получаем изображение из API
                answers = stability_api.generate(
                    prompt=output_str, 
                    seed=seed, 
                    steps=steps, 
                    cfg_scale=cfg_scale, 
                    width=512, 
                    height=512, 
                    samples=1, 
                    sampler=generation.SAMPLER_K_DPMPP_SDE)
                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.finish_reason == generation.FILTER:
                           warnings.warn(
                               "Your request activated the API's safety filters and could not be processed."
                                "Please modify the prompt and try again.")
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            img_bytes = io.BytesIO(artifact.binary)
                            await context.bot.send_photo(chat_id=chat_id, photo=img_bytes)

                    
                    
            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    text=f'Failed to generate image: {str(e)}',
                    parse_mode=constants.ParseMode.MARKDOWN
                )
        await self.wrap_with_indicator(update, context, constants.ChatAction.UPLOAD_PHOTO, _generate)       
        
    async def image_stable_albom1(self, message, callback_query: CallbackQuery, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using stability.ai APIs.
        """
        if not self.config['enable_image_generation'] or not await self.check_allowed_and_within_budget(callback_query, context):
            return

        chat_id = callback_query.message.chat_id

        logging.info(f'New image generation request received from user {callback_query.message.from_user.name} '
            f'(id: {callback_query.message.from_user.id})')

        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        STABILITY_KEY = os.getenv('STABILITY_KEY')
        os.environ['STABILITY_KEY'] = STABILITY_KEY
        await context.bot.send_message(chat_id=chat_id, text='Please wait: Usually generation takes no more than 7 seconds for each image.')
        
        stability_api = StabilityInference(
            key=os.environ['STABILITY_KEY'],  # API Key reference.
            verbose=True,  # Print debug messages.
            engine="stable-diffusion-xl-beta-v2-2-2",  # Set the engine to use for generation.
        )
        
        def parse_messag(message):
            seed = 123467598
            steps = 50
            cfg_scale = 8.0

            for part in message.split():
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["seed", "sd", "se", "see"]:
                    try:
                        seed = int(value)
                    except ValueError:
                        pass
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["cfg_scale", "cfg", "cfg_", "cfg_s", "cf"]:
                    try:
                        cfg_scale = int(value)
                    except ValueError:
                        pass
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["steps", "ste", "st", "step"]:
                    try:
                        steps = int(value)
                    except ValueError:
                        pass
            return seed, steps, cfg_scale
        
        
        def parse_message(message):
            # находим все скобки в сообщении
            parentheses_regex = r"\(([^(),]+),\s*(-?\d+)\)"
            matches = re.findall(parentheses_regex, message)
            
            # использовать параметры для формирования строки Prompt если найдены скобки
            if len(matches) > 0:
                output_str = f"[generation.Prompt(text='{message}', parameters=generation.PromptParameters(weight=1)), generation.Prompt(text='{matches[0][0]}', parameters=generation.PromptParameters(weight={matches[0][1]}))]"
                return output_str
            else:
                return None  # возвращаем пустую строку, если нет скобок
            
        async def _generate():
            try:
                output_str = parse_message(message)
                seed, steps, cfg_scale = parse_messag(message)

                if output_str is None:
                    full_str = message
                else:
                    full_str = output_str

                # Получаем изображение из API
                answers = stability_api.generate(
                    prompt=full_str, 
                    seed=seed, 
                    steps=steps, 
                    cfg_scale=cfg_scale, 
                    width=512, 
                    height=512, 
                    samples=4, 
                    sampler=generation.SAMPLER_K_DPMPP_SDE)

                img_bytes = []  # список для сохранения байтов изображений

                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.finish_reason == generation.FILTER:
                            warnings.warn("Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again.")
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            img_bytes.append(io.BytesIO(artifact.binary))

                # отправляем все изображения как медиа-группу
                if len(img_bytes) > 0:
                    media = [InputMediaPhoto(image) for image in img_bytes]
                    await context.bot.send_media_group(chat_id=chat_id, media=media)
      
            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(chat_id=chat_id, text='An error has occurred on the server!') 
                await context.bot.send_message( chat_id=chat_id, reply_to_message_id=self.get_reply_to_message_id(callback_query), text=f'Failed to generate image: {str(e)}', parse_mode=constants.ParseMode.MARKDOWN ) 
            
        await self.wrap_with_indicator(callback_query, context, constants.ChatAction.UPLOAD_PHOTO, _generate)
        button_exit = InlineKeyboardButton("EXIT", callback_data="button")
        reply_markup = InlineKeyboardMarkup([[button_exit]])
        await context.bot.send_message(chat_id=chat_id, text="To finish, click on the button below or enter the following query", reply_markup=reply_markup)

    async def image_stable_albom(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using stability.ai APIs.
        """
        if not self.config['enable_image_generation'] or not await self.check_allowed_and_within_budget(update, context):
            return

        chat_id = update.effective_chat.id
        message = update.message.text
        image_query = message_text(update.message)
        if image_query == '':
            await context.bot.send_message(chat_id=chat_id, text='Please provide a prompt! (e.g. /stable_albom cat)')
            return

        logging.info(f'New image generation request received from user {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})')

        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        STABILITY_KEY = os.getenv('STABILITY_KEY')
        os.environ['STABILITY_KEY'] = STABILITY_KEY
        await context.bot.send_message(chat_id=chat_id, text='Please wait: Usually generation takes no more than 7 seconds for each image.')
        
        stability_api = StabilityInference(
            key=os.environ['STABILITY_KEY'],  # API Key reference.
            verbose=True,  # Print debug messages.
            engine="stable-diffusion-xl-beta-v2-2-2",  # Set the engine to use for generation.
        )
        
        def parse_messag(message):
            seed = 123467598
            steps = 50
            cfg_scale = 8.0

            for part in message.split():
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["seed", "sd", "se", "see"]:
                    try:
                        seed = int(value)
                    except ValueError:
                        pass
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["cfg_scale", "cfg", "cfg_", "cfg_s", "cf"]:
                    try:
                        cfg_scale = int(value)
                    except ValueError:
                        pass
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["steps", "ste", "st", "step"]:
                    try:
                        steps = int(value)
                    except ValueError:
                        pass
            return seed, steps, cfg_scale
        
        
        def parse_message(message):
            # находим все скобки в сообщении
            parentheses_regex = r"\(([^(),]+),\s*(-?\d+)\)"
            matches = re.findall(parentheses_regex, message)
            
            # использовать параметры для формирования строки Prompt если найдены скобки
            if len(matches) > 0:
                output_str = f"[generation.Prompt(text='{message}', parameters=generation.PromptParameters(weight=1)), generation.Prompt(text='{matches[0][0]}', parameters=generation.PromptParameters(weight={matches[0][1]}))]"
                return output_str
            else:
                return None  # возвращаем пустую строку, если нет скобок
            
        async def _generate():
            try:
                output_str = parse_message(message)
                seed, steps, cfg_scale = parse_messag(message)

                if output_str is None:
                    full_str = message
                else:
                    full_str = output_str

                # Получаем изображение из API
                answers = stability_api.generate(
                    prompt=full_str, 
                    seed=seed, 
                    steps=steps, 
                    cfg_scale=cfg_scale, 
                    width=512, 
                    height=512, 
                    samples=10, 
                    sampler=generation.SAMPLER_K_DPMPP_SDE)

                img_bytes = []  # список для сохранения байтов изображений

                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.finish_reason == generation.FILTER:
                            warnings.warn("Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again.")
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            img_bytes.append(io.BytesIO(artifact.binary))

                # отправляем все изображения как медиа-группу
                if len(img_bytes) > 0:
                    media = [InputMediaPhoto(image) for image in img_bytes]
                    await context.bot.send_media_group(chat_id=chat_id, media=media)


                    
                    
            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(chat_id=chat_id, text='Произошла ошибка на сервере!')
                await context.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    text=f'Failed to generate image: {str(e)}',
                    parse_mode=constants.ParseMode.MARKDOWN
                )
        await self.Awrap_with_indicator(update, context, constants.ChatAction.UPLOAD_PHOTO, _generate)         

    async def image_stable_clip(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using stability.ai APIs.
        """
        if not self.config['enable_image_generation'] or not await self.check_allowed_and_within_budget(update, context):
            return

        chat_id = update.effective_chat.id
        message = update.message.text
        image_query = message_text(update.message)
        if image_query == '':
            await context.bot.send_message(chat_id=chat_id, text='Please provide a prompt! (e.g. /stable_clip cat)')
            return

        logging.info(f'New image generation request received from user {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})')
        await context.bot.send_message(chat_id=chat_id, text='Please wait: Usually generation takes no more than 7 seconds for each image.')
        os.environ['STABILITY_HOST'] = 'grpc.stability.ai:443'
        STABILITY_KEY = os.getenv('STABILITY_KEY')
        os.environ['STABILITY_KEY'] = STABILITY_KEY

                
        def parse_messag(message):
            seed = 123467598
            steps = 50
            cfg_scale = 8.0

            for part in message.split():
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["seed", "sd", "se", "see"]:
                    try:
                        seed = int(value)
                    except ValueError:
                        pass
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["cfg_scale", "cfg", "cfg_", "cfg_s", "cf"]:
                    try:
                        cfg_scale = int(value)
                    except ValueError:
                        pass
                if ":" not in part:
                    continue
                key, value = part.split(":")
                if key in ["steps", "ste", "st", "step"]:
                    try:
                        steps = int(value)
                    except ValueError:
                        pass
            return seed, steps, cfg_scale
        
        seed, steps, cfg_scale = parse_messag(message)

        stability_api = StabilityInference(
            key=os.environ['STABILITY_KEY'],  # API Key reference.
            verbose=True,  # Print debug messages.
            engine="stable-diffusion-xl-beta-v2-2-2",  # Set the engine to use for generation.
        )
        def parse_message(message, chat_id):
            # находим все скобки в сообщении
            parentheses_regex = r"\(([^(),]+),\s*(-?\d+)\)"
            matches = re.findall(parentheses_regex, message)
            
            # использовать параметры для формирования строки Prompt если найдены скобки
            if len(matches) > 0:
                output_str = f"{message}"
                return output_str
            else:
                return ''  # возвращаем пустую строку, если нет скобок
            
        output_str = parse_message(message, chat_id)
        async def _generate():
            try:
                # Получаем изображение из API
                answers = stability_api.generate(
                      prompt=output_str,
                      seed=seed, 
                      steps=steps, 
                      cfg_scale=cfg_scale, 
                      width=512, 
                      height=512, 
                      samples=1, 
                      sampler=generation.SAMPLER_K_DPMPP_SDE, 
                      guidance_preset=generation.GUIDANCE_PRESET_FAST_GREEN)

                img_bytes = []  # список для сохранения байтов изображений

                for resp in answers:
                    for artifact in resp.artifacts:
                        if artifact.finish_reason == generation.FILTER:
                            warnings.warn("Your request activated the API's safety filters and could not be processed. Please modify the prompt and try again.")
                        if artifact.type == generation.ARTIFACT_IMAGE:
                            img_bytes.append(io.BytesIO(artifact.binary))

                # отправляем все изображения как медиа-группу
                if len(img_bytes) > 0:
                    media = [InputMediaPhoto(image) for image in img_bytes]
                    await context.bot.send_media_group(chat_id=chat_id, media=media)


                    
                    
            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    text=f'Failed to generate image: {str(e)}',
                    parse_mode=constants.ParseMode.MARKDOWN
                )
        await self.Awrap_with_indicator(update, context, constants.ChatAction.UPLOAD_PHOTO, _generate)       
        
    async def stats(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Returns token usage statistics for current day and month.
        """
        if not await self.is_allowed(update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                f'is not allowed to request their usage statistics')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
            f'requested their usage statistics')
        
        user_id = update.message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        tokens_today, tokens_month = self.usage[user_id].get_current_token_usage()
        images_today, images_month = self.usage[user_id].get_current_image_count()
        transcribe_durations = self.usage[user_id].get_current_transcription_duration()
        cost_today, cost_month = self.usage[user_id].get_current_cost()
        
        chat_id = update.effective_chat.id
        chat_messages, chat_token_length = self.openai.get_conversation_stats(chat_id)
        budget = await self.get_remaining_budget(update)

        text_current_conversation = f"*Current conversation:*\n"+\
                     f"{chat_messages} chat messages in history.\n"+\
                     f"{chat_token_length} chat tokens in history.\n"+\
                     f"----------------------------\n"
        text_today = f"*Usage today:*\n"+\
                     f"{tokens_today} chat tokens used.\n"+\
                     f"{images_today} images generated.\n"+\
                     f"{transcribe_durations[0]} minutes and {transcribe_durations[1]} seconds transcribed.\n"+\
                     f"💰 For a total amount of ${cost_today:.2f}\n"+\
                     f"----------------------------\n"
        text_month = f"*Usage this month:*\n"+\
                     f"{tokens_month} chat tokens used.\n"+\
                     f"{images_month} images generated.\n"+\
                     f"{transcribe_durations[2]} minutes and {transcribe_durations[3]} seconds transcribed.\n"+\
                     f"💰 For a total amount of ${cost_month:.2f}"
        # text_budget filled with conditional content
        text_budget = "\n\n"
        if budget < float('inf'):
            text_budget += f"You have a remaining budget of ${budget:.2f} this month.\n"
        # add OpenAI account information for admin request
        if self.is_admin(update):
            text_budget += f"Your OpenAI account was billed ${self.openai.get_billing_current_month():.2f} this month."
        
        usage_text = text_current_conversation + text_today + text_month + text_budget
        await update.message.reply_text(usage_text, parse_mode=constants.ParseMode.MARKDOWN)

    async def resend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resend the last request
        """
        if not await self.is_allowed(update, context):
            logging.warning(f'User {update.message.from_user.name}  (id: {update.message.from_user.id})'
                            f' is not allowed to resend the message')
            await self.send_disallowed_message(update, context)
            return

        chat_id = update.effective_chat.id
        if chat_id not in self.last_message:
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id})'
                            f' does not have anything to resend')
            await context.bot.send_message(chat_id=chat_id, text="You have nothing to resend")
            return

        # Update message text, clear self.last_message and send the request to prompt
        logging.info(f'Resending the last prompt from user: {update.message.from_user.name} '
                     f'(id: {update.message.from_user.id})')
        with update.message._unfrozen() as message:
            message.text = self.last_message.pop(chat_id)

        await self.prompt(update=update, context=context)

    async def reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Resets the conversation.
        """
        if not await self.is_allowed(update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                f'is not allowed to reset the conversation')
            await self.send_disallowed_message(update, context)
            return

        logging.info(f'Resetting the conversation for user {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})...')

        chat_id = update.effective_chat.id
        reset_content = message_text(update.message)
        self.openai.reset_chat_history(chat_id=chat_id, content=reset_content)
        await context.bot.send_message(chat_id=chat_id, text='Done!')
        
    async def image(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Generates an image for the given prompt using DALL·E APIs
        """
        if not self.config['enable_image_generation'] or not await self.check_allowed_and_within_budget(update, context):
            return
        chat_id = update.effective_chat.id
        image_query = message_text(update.message)
        if image_query == '':
            await context.bot.send_message(chat_id=chat_id, text='Please provide a prompt! (e.g. /image cat)')
            return

        logging.info(f'New image generation request received from user {update.message.from_user.name} '
            f'(id: {update.message.from_user.id})')

        async def _generate():
            try:
                image_url, image_size = await self.openai.generate_image(prompt=image_query)
                await context.bot.send_photo(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    photo=image_url
                )
                # add image request to users usage tracker
                user_id = update.message.from_user.id
                self.usage[user_id].add_image_request(image_size, self.config['image_prices'])
                # add guest chat request to guest usage tracker
                if str(user_id) not in self.config['allowed_user_ids'].split(',') and 'guests' in self.usage:
                    self.usage["guests"].add_image_request(image_size, self.config['image_prices'])

            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    text=f'Failed to generate image: {str(e)}',
                    parse_mode=constants.ParseMode.MARKDOWN
                )

        await self.wrap_with_indicator(update, context, constants.ChatAction.UPLOAD_PHOTO, _generate)

    async def transcribe(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Transcribe audio messages.
        """
        if not self.config['enable_transcription'] or not await self.check_allowed_and_within_budget(update, context):
            return

        if self.is_group_chat(update) and self.config['ignore_group_transcriptions']:
            logging.info(f'Transcription coming from group chat, ignoring...')
            return

        chat_id = update.effective_chat.id
        filename = update.message.effective_attachment.file_unique_id

        async def _execute():
            filename_mp3 = f'{filename}.mp3'

            try:
                media_file = await context.bot.get_file(update.message.effective_attachment.file_id)
                await media_file.download_to_drive(filename)
            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    text=f'Failed to download audio file: {str(e)}. Make sure the file is not too large. (max 20MB)',
                    parse_mode=constants.ParseMode.MARKDOWN
                )
                return

            # detect and extract audio from the attachment with pydub
            try:
                audio_track = AudioSegment.from_file(filename)
                audio_track.export(filename_mp3, format="mp3")
                logging.info(f'New transcribe request received from user {update.message.from_user.name} '
                    f'(id: {update.message.from_user.id})')

            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(
                    chat_id=update.effective_chat.id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    text='Unsupported file type'
                )
                if os.path.exists(filename):
                    os.remove(filename)
                return

            user_id = update.message.from_user.id
            if user_id not in self.usage:
                self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

            # send decoded audio to openai
            try:

                # Transcribe the audio file
                transcript = await self.openai.transcribe(filename_mp3)

                # add transcription seconds to usage tracker
                transcription_price = self.config['transcription_price']
                self.usage[user_id].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                # add guest chat request to guest usage tracker
                allowed_user_ids = self.config['allowed_user_ids'].split(',')
                if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                    self.usage["guests"].add_transcription_seconds(audio_track.duration_seconds, transcription_price)

                if self.config['voice_reply_transcript']:

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = f'_Transcript:_\n"{transcript}"'
                    chunks = self.split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        await context.bot.send_message(
                            chat_id=chat_id,
                            reply_to_message_id=self.get_reply_to_message_id(update) if index == 0 else None,
                            text=transcript_chunk,
                            parse_mode=constants.ParseMode.MARKDOWN
                        )
                else:
                    # Get the response of the transcript
                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=transcript)

                    # add chat request to users usage tracker
                    self.usage[user_id].add_chat_tokens(total_tokens, self.config['token_price'])
                    # add guest chat request to guest usage tracker
                    if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                        self.usage["guests"].add_chat_tokens(total_tokens, self.config['token_price'])

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    transcript_output = f'_Transcript:_\n"{transcript}"\n\n_Answer:_\n{response}'
                    chunks = self.split_into_chunks(transcript_output)

                    for index, transcript_chunk in enumerate(chunks):
                        await context.bot.send_message(
                            chat_id=chat_id,
                            reply_to_message_id=self.get_reply_to_message_id(update) if index == 0 else None,
                            text=transcript_chunk,
                            parse_mode=constants.ParseMode.MARKDOWN
                        )

            except Exception as e:
                logging.exception(e)
                await context.bot.send_message(
                    chat_id=chat_id,
                    reply_to_message_id=self.get_reply_to_message_id(update),
                    text=f'Failed to transcribe text: {str(e)}',
                    parse_mode=constants.ParseMode.MARKDOWN
                )
            finally:
                # Cleanup files
                if os.path.exists(filename_mp3):
                    os.remove(filename_mp3)
                if os.path.exists(filename):
                    os.remove(filename)

        await self.wrap_with_indicator(update, context, constants.ChatAction.TYPING, _execute)

    async def generate_prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        React to incoming messages and respond accordingly.
        """
        chat_id = update.effective_chat.id

        if not await self.check_allowed_and_within_budget(update, context):
            return
        promptD = f'Generate a prompt from : {update.message.text}. Following the following example: Ryan Gosling smiling, symmetric highly detailed eyes, trending on artstation, portrait, digital art, masterpice, by Vladimir Kush and Scott Naismith. Do you take into account that there are restrictions in the amount of 75 words'
        logging.info(f'New message received from user {update.message.from_user.name} (id: {update.message.from_user.id})')
        user_id = update.message.from_user.id
        prompt = promptD+update.message.text
        self.last_message[chat_id] = prompt

        if self.is_group_chat(update):
            trigger_keyword = self.config['group_trigger_keyword']
            if prompt.lower().startswith(trigger_keyword.lower()):
                prompt = prompt[len(trigger_keyword):].strip()
            else:
                if update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
                    logging.info('Message is a reply to the bot, allowing...')
                else:
                    logging.warning('Message does not start with trigger keyword, ignoring...')
                    return

        try:
            if self.config['stream']:
                await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
                is_group_chat = self.is_group_chat(update)

                stream_response = self.openai.get_chat_response_stream(chat_id=chat_id, query=prompt)
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                chunk = 0

                async for content, tokens in stream_response:
                    if len(content.strip()) == 0:
                        continue

                    chunks = self.split_into_chunks(content)
                    if len(chunks) > 1:
                        content = chunks[-1]
                        if chunk != len(chunks) - 1:
                            chunk += 1
                            try:
                                await self.edit_message_with_retry(context, chat_id, sent_message.message_id, chunks[-2])
                            except:
                                pass
                            try:
                                sent_message = await context.bot.send_message(
                                    chat_id=sent_message.chat_id,
                                    text=content if len(content) > 0 else "..."
                                )
                            except:
                                pass
                            continue

                    if is_group_chat:
                        # group chats have stricter flood limits
                        cutoff = 180 if len(content) > 1000 else 120 if len(content) > 200 else 90 if len(content) > 50 else 50
                    else:
                        cutoff = 90 if len(content) > 1000 else 45 if len(content) > 200 else 25 if len(content) > 50 else 15

                    cutoff += backoff

                    if i == 0:
                        try:
                            if sent_message is not None:
                                await context.bot.delete_message(chat_id=sent_message.chat_id,
                                                                 message_id=sent_message.message_id)
                            sent_message = await context.bot.send_message(
                                chat_id=chat_id,
                                reply_to_message_id=self.get_reply_to_message_id(update),
                                text=content
                            )
                        except:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content

                        try:
                            use_markdown = tokens != 'not_finished'
                            await self.edit_message_with_retry(context, chat_id, sent_message.message_id,
                                                               text=content, markdown=use_markdown)

                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue

                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

            else:
                async def _reply():
                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=prompt)

                    # Split into chunks of 4096 characters (Telegram's message limit)
                    chunks = self.split_into_chunks(response)

                    for index, chunk in enumerate(chunks):
                        try:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                reply_to_message_id=self.get_reply_to_message_id(update) if index == 0 else None,
                                text=chunk,
                                parse_mode=constants.ParseMode.MARKDOWN
                            )
                        except Exception:
                            try:
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    reply_to_message_id=self.get_reply_to_message_id(update) if index == 0 else None,
                                    text=chunk
                                )
                            except Exception as e:
                                raise e

                await self.wrap_with_indicator(update, context, constants.ChatAction.TYPING, _reply)

            try:
                # add chat request to users usage tracker
                self.usage[user_id].add_chat_tokens(total_tokens, self.config['token_price'])
                # add guest chat request to guest usage tracker
                allowed_user_ids = self.config['allowed_user_ids'].split(',')
                if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                    self.usage["guests"].add_chat_tokens(total_tokens, self.config['token_price'])
            except:
                pass

        except Exception as e:
            logging.exception(e)
            await context.bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=self.get_reply_to_message_id(update),
                text=f'Failed to get response: {str(e)}',
                parse_mode=constants.ParseMode.MARKDOWN
            )
   
    async def prompt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        React to incoming messages and respond accordingly.
        """
        chat_id = update.effective_chat.id

        
        if not await self.check_allowed_and_within_budget(update, context):
            return
        
        logging.info(f'New message received from user {update.message.from_user.name} (id: {update.message.from_user.id})')
        user_id = update.message.from_user.id

        to_img = update.message.text

        prompt = f'Generate a prompt from : {to_img}. Following the following example: Ryan Gosling smiling, symmetric highly detailed eyes, trending on artstation, portrait, digital art, masterpice, by Vladimir Kush and Scott Naismith'
                
        def parse_message(to_img):
            promp = ""
            seed = ' seed:123467598'
            steps = ' steps:50'
            cfg_scale = ' cfg_scale:8.0'

            for part in to_img.split():
                if ":" not in part:
                    promp += " " + part
                    continue
                key, value = part.split(":")
                if key in ["seed", "sd", "se", "see"]:
                    try:
                        seed = int(value)
                    except ValueError:
                        pass
                elif key in ["cfg_scale", "cfg", "cfg_", "cfg_s", "cf"]:
                    try:
                        cfg_scale = float(value)
                    except ValueError:
                        pass
                elif key in ["steps", "ste", "st", "step"]:
                    try:
                        steps = int(value)
                    except ValueError:
                        pass

            # Удаляем начальные и конечные пробелы в prompt
            promp = prompt.strip()
            return prompt, seed, steps, cfg_scale
        
        promp, seed, steps, cfg_scale = parse_message(to_img)
            
        self.last_message[chat_id] = prompt

        if self.is_group_chat(update):
            trigger_keyword = self.config['group']
            if prompt.lower().startswith(trigger_keyword.lower()):
                prompt = prompt[len(trigger_keyword):].strip()
            else:
                if update.message.reply_to_message and update.message.reply_to_message.from_user.id == context.bot.id:
                    logging.info('Message is a reply to the bot, allowing...')
                else:
                    logging.warning('Message does not start with trigger keyword, ignoring...')
                    return

        try:
            if self.config['stream']:
                await context.bot.send_chat_action(chat_id=chat_id, action=constants.ChatAction.TYPING)
                is_group_chat = self.is_group_chat(update)

                stream_response = self.openai.get_chat_response_stream(chat_id=chat_id, query=prompt)
                i = 0
                prev = ''
                sent_message = None
                backoff = 0
                chunk = 0

                async for content, tokens in stream_response:
                    if len(content.strip()) == 0:
                        continue

                    chunks = self.split_into_chunks(content)
                    if len(chunks) > 1:
                        content = chunks[-1]
                        if chunk != len(chunks) - 1:
                            chunk += 1
                            try:
                                await self.edit_message_with_retry(context, chat_id, sent_message.message_id, chunks[-2])
                            except:
                                pass
                            try:
                                sent_message = await context.bot.send_message(
                                    chat_id=sent_message.chat_id,
                                    text=content if len(content) > 0 else "..."
                                )
                            except:
                                pass
                            continue

                    if is_group_chat:
                        # group chats have stricter flood limits
                        cutoff = 180 if len(content) > 1000 else 120 if len(content) > 200 else 90 if len(content) > 50 else 50
                    else:
                        cutoff = 90 if len(content) > 1000 else 45 if len(content) > 200 else 25 if len(content) > 50 else 15

                    cutoff += backoff

                    if i == 0:
                        try:
                            if sent_message is not None:
                                await context.bot.delete_message(chat_id=sent_message.chat_id,
                                                                 message_id=sent_message.message_id)
                            sent_message = await context.bot.send_message(
                                chat_id=chat_id,
                                reply_to_message_id=self.get_reply_to_message_id(update),
                                text=content
                            )
                        except:
                            continue

                    elif abs(len(content) - len(prev)) > cutoff or tokens != 'not_finished':
                        prev = content

                        try:
                            use_markdown = tokens != 'not_finished'
                            await self.edit_message_with_retry(context, chat_id, sent_message.message_id,
                                                               text=content, markdown=use_markdown)
                            self.responses[chat_id] = f'{content}'+f'{seed}'+f'{steps}'+f'{cfg_scale}'
                            self.text[chat_id] = f'{to_img}'
                        except RetryAfter as e:
                            backoff += 5
                            await asyncio.sleep(e.retry_after)
                            continue

                        except TimedOut:
                            backoff += 5
                            await asyncio.sleep(0.5)
                            continue

                        except Exception:
                            backoff += 5
                            continue

                        await asyncio.sleep(0.01)

                    i += 1
                    if tokens != 'not_finished':
                        total_tokens = int(tokens)

            else:
                async def _reply():
                    response, total_tokens = await self.openai.get_chat_response(chat_id=chat_id, query=prompt)


                    
                    # Split into chunks of 4096 characters (Telegram's message limit)
                    chunks = self.split_into_chunks(response)

                    for index, chunk in enumerate(chunks):
                        try:
                            await context.bot.send_message(
                                chat_id=chat_id,
                                reply_to_message_id=self.get_reply_to_message_id(update) if index == 0 else None,
                                text=chunk,
                                parse_mode=constants.ParseMode.MARKDOWN
                            )

                        except Exception:
                            try:
                                await context.bot.send_message(
                                    chat_id=chat_id,
                                    reply_to_message_id=self.get_reply_to_message_id(update) if index == 0 else None,
                                    text=chunk
                                )
                                
                            except Exception as e:
                                raise e
                            
                await self.wrap_with_indicator(update, context, constants.ChatAction.TYPING, _reply)
                
               
            try:
                # add chat request to users usage tracker
                self.usage[user_id].add_chat_tokens(total_tokens, self.config['token_price'])
                # add guest chat request to guest usage tracker
                allowed_user_ids = self.config['allowed_user_ids'].split(',')
                if str(user_id) not in allowed_user_ids and 'guests' in self.usage:
                    self.usage["guests"].add_chat_tokens(total_tokens, self.config['token_price'])
            except:
                pass
            
        except Exception as e:
            logging.exception(e)
            await context.bot.send_message(
                chat_id=chat_id,
                reply_to_message_id=self.get_reply_to_message_id(update),
                text=f'Failed to get response: {str(e)}',
                parse_mode=constants.ParseMode.MARKDOWN
            )
        
        await self.variants(update, context)

    async def inline_query(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handle the inline query. This is run when you type: @botusername <query>
        """
        query = update.inline_query.query

        if query == '':
            return

        results = [
            InlineQueryResultArticle(
                id=query,
                title='Ask ChatGPT',
                input_message_content=InputTextMessageContent(query),
                description=query,
                thumb_url='https://user-images.githubusercontent.com/11541888/223106202-7576ff11-2c8e-408d-94ea-b02a7a32149a.png'
            )
        ]

        await update.inline_query.answer(results)

    async def edit_message_with_retry(self, context: ContextTypes.DEFAULT_TYPE, chat_id: int,
                                      message_id: int, text: str, markdown: bool = True):
        """
        Edit a message with retry logic in case of failure (e.g. broken markdown)
        :param context: The context to use
        :param chat_id: The chat id to edit the message in
        :param message_id: The message id to edit
        :param text: The text to edit the message with
        :param markdown: Whether to use markdown parse mode
        :return: None
        """
        try:
            await context.bot.edit_message_text(
                chat_id=chat_id,
                message_id=message_id,
                text=text,
                parse_mode=constants.ParseMode.MARKDOWN if markdown else None
            )
        except telegram.error.BadRequest as e:
            if str(e).startswith("Message is not modified"):
                return
            try:
                await context.bot.edit_message_text(
                    chat_id=chat_id,
                    message_id=message_id,
                    text=text
                )
            except Exception as e:
                logging.warning(f'Failed to edit message: {str(e)}')
                raise e

        except Exception as e:
            logging.warning(str(e))
            raise e

    async def wrap_with_indicator(self, update: Update, context: CallbackContext, chat_action: constants.ChatAction, coroutine):
        """
        Wraps a coroutine while repeatedly sending a chat action to the user.
        """
        task = context.application.create_task(coroutine(), update=update)
        while not task.done():
            context.application.create_task(update.effective_chat.send_action(chat_action))
            try:
                await asyncio.wait_for(asyncio.shield(task), 4.5)
            except asyncio.TimeoutError:
                pass

    async def send_disallowed_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Sends the disallowed message to the user.
        """
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=self.disallowed_message,
            disable_web_page_preview=True
        )

    async def send_budget_reached_message(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """
        Sends the budget reached message to the user.
        """
        await context.bot.send_message(
            chat_id=update.effective_chat.id,
            text=self.budget_limit_message
        )

    async def error_handler(self, update: object, context: ContextTypes.DEFAULT_TYPE) -> None:
        """
        Handles errors in the telegram-python-bot library.
        """
        logging.error(f'Exception while handling an update: {context.error}')

    def is_group_chat(self, update: Update) -> bool:
        """
        Checks if the message was sent from a group chat
        """
        return update.effective_chat.type in [
            constants.ChatType.GROUP,
            constants.ChatType.SUPERGROUP
        ]

    async def is_user_in_group(self, update: Update, context: CallbackContext, user_id: int) -> bool:
        """
        Checks if user_id is a member of the group
        """
        try:
            chat_member = await context.bot.get_chat_member(update.message.chat_id, user_id)
            return chat_member.status in [ChatMember.OWNER, ChatMember.ADMINISTRATOR, ChatMember.MEMBER]
        except telegram.error.BadRequest as e:
            if str(e) == "User not found":
                return False
            else:
                raise e
        except Exception as e:
            raise e

    async def is_allowed(self, update: Update, context: CallbackContext) -> bool:
        """
        Checks if the user is allowed to use the bot.
        """
        if self.config['allowed_user_ids'] == '*':
            return True
        
        if self.is_admin(update):
            return True
        
        allowed_user_ids = self.config['allowed_user_ids'].split(',')
        # Check if user is allowed
        if str(update.message.from_user.id) in allowed_user_ids:
            return True

        # Check if it's a group a chat with at least one authorized member
        if self.is_group_chat(update):
            admin_user_ids = self.config['admin_user_ids'].split(',')
            for user in itertools.chain(allowed_user_ids, admin_user_ids):
                if await self.is_user_in_group(update, context, user):
                    logging.info(f'{user} is a member. Allowing group chat message...')
                    return True
            logging.info(f'Group chat messages from user {update.message.from_user.name} '
                f'(id: {update.message.from_user.id}) are not allowed')

        return False

    def is_admin(self, update: Update) -> bool:
        """
        Checks if the user is the admin of the bot.
        The first user in the user list is the admin.
        """
        if self.config['admin_user_ids'] == '-':
            logging.info('No admin user defined.')
            return False

        admin_user_ids = self.config['admin_user_ids'].split(',')

        # Check if user is in the admin user list
        if str(update.message.from_user.id) in admin_user_ids:
            return True

        return False

    async def get_remaining_budget(self, update: Update) -> float:
        user_id = update.message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        if self.is_admin(update):
            return float('inf')

        if self.config['monthly_user_budgets'] == '*':
            return float('inf')

        allowed_user_ids = self.config['allowed_user_ids'].split(',')
        if str(user_id) in allowed_user_ids:
            # find budget for allowed user
            user_index = allowed_user_ids.index(str(user_id))
            user_budgets = self.config['monthly_user_budgets'].split(',')
            # check if user is included in budgets list
            if len(user_budgets) <= user_index:
                logging.warning(f'No budget set for user: {update.message.from_user.name} ({user_id}).')
                return 0.0
            user_budget = float(user_budgets[user_index])
            cost_month = self.usage[user_id].get_current_cost()[1]
            remaining_budget = user_budget - cost_month
            return remaining_budget
        else:
            return 0.0

    async def is_within_budget(self, update: Update, context: CallbackContext) -> bool:
        """
        Checks if the user reached their monthly usage limit.
        Initializes UsageTracker for user and guest when needed.
        """
        user_id = update.message.from_user.id
        if user_id not in self.usage:
            self.usage[user_id] = UsageTracker(user_id, update.message.from_user.name)

        if self.is_admin(update):
            return True

        if self.config['monthly_user_budgets'] == '*':
            return True

        allowed_user_ids = self.config['allowed_user_ids'].split(',')
        if str(user_id) in allowed_user_ids:
            # find budget for allowed user
            user_index = allowed_user_ids.index(str(user_id))
            user_budgets = self.config['monthly_user_budgets'].split(',')
            # check if user is included in budgets list
            if len(user_budgets) <= user_index:
                logging.warning(f'No budget set for user: {update.message.from_user.name} ({user_id}).')
                return False
            user_budget = float(user_budgets[user_index])
            cost_month = self.usage[user_id].get_current_cost()[1]
            # Check if allowed user is within budget
            return user_budget > cost_month

        # Check if group member is within budget
        if self.is_group_chat(update):
            admin_user_ids = self.config['admin_user_ids'].split(',')
            for user in itertools.chain(allowed_user_ids, admin_user_ids):
                if await self.is_user_in_group(update, context, user):
                    if 'guests' not in self.usage:
                        self.usage['guests'] = UsageTracker('guests', 'all guest users in group chats')
                    if self.config['monthly_guest_budget'] >= self.usage['guests'].get_current_cost()[1]:
                        return True
                    logging.warning('Monthly guest budget for group chats used up.')
                    return False
            logging.info(f'Group chat messages from user {update.message.from_user.name} '
                f'(id: {update.message.from_user.id}) are not allowed')
        return False

    async def check_allowed_and_within_budget(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> bool:
        """
        Checks if the user is allowed to use the bot and if they are within their budget
        :param update: Telegram update object
        :param context: Telegram context object
        :return: Boolean indicating if the user is allowed to use the bot
        """
        if not await self.is_allowed(update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                f'is not allowed to use the bot')
            await self.send_disallowed_message(update, context)
            return False

        if not await self.is_within_budget(update, context):
            logging.warning(f'User {update.message.from_user.name} (id: {update.message.from_user.id}) '
                f'reached their usage limit')
            await self.send_budget_reached_message(update, context)
            return False

        return True

    def get_reply_to_message_id(self, update: Update):
        """
        Returns the message id of the message to reply to
        :param update: Telegram update object
        :return: Message id of the message to reply to, or None if quoting is disabled
        """
        if self.config['enable_quoting'] or self.is_group_chat(update):
            return update.message.message_id
        return None

    def split_into_chunks(self, text: str, chunk_size: int = 4096) -> list[str]:
        """
        Splits a string into chunks of a given size.
        """
        return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]

    async def post_init(self, application: Application) -> None:
        """
        Post initialization hook for the bot.
        """
        await application.bot.set_my_commands(self.commands)
 
    def run(self):
        """
        Runs the bot indefinitely until the user presses Ctrl+C
        """
        application = ApplicationBuilder() \
            .token(self.config['token']) \
            .proxy_url(self.config['proxy']) \
            .get_updates_proxy_url(self.config['proxy']) \
            .post_init(self.post_init) \
            .concurrent_updates(True) \
            .build()

        application.add_handler(CommandHandler('reset', self.reset))
        application.add_handler(CommandHandler('start', self.start))
        application.add_handler(CommandHandler('help', self.help))
        application.add_handler(CallbackQueryHandler(self.button_callback, pattern='button'))
        application.add_handler(CommandHandler('image', self.image))     
        application.add_handler(CommandHandler('stable', self.image_stable))
        application.add_handler(CommandHandler('stable_albom', self.image_stable_albom))
        application.add_handler(CommandHandler('image_editor', self.image_editor))
        application.add_handler(CommandHandler('stable_clip', self.image_stable_clip))
        application.add_handler(CommandHandler('prompt', self.generate_prompt))
        application.add_handler(CommandHandler('stable_upscale', self.image_stable_upscale))
        application.add_handler(CommandHandler('start', self.help))
        application.add_handler(CommandHandler('resend', self.resend))
        application.add_handler(MessageHandler(
            filters.AUDIO | filters.VOICE | filters.Document.AUDIO,
            self.transcribe))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.prompt))
        application.add_handler(MessageHandler(filters.TEXT & (~filters.COMMAND), self.generate_prompt))
        application.add_handler(InlineQueryHandler(self.inline_query, chat_types=[
            constants.ChatType.GROUP, constants.ChatType.SUPERGROUP
        ]))


        application.add_error_handler(self.error_handler)

        application.run_polling()
