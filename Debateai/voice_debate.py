#This file contains the code to execute the debate between the bot and the user over the voice commands.
import os
import re
import queue
import time
import torch
import pyaudio
import threading
import asyncio
from groq import Groq
from google.cloud import speech
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
import warnings
from concurrent.futures import ThreadPoolExecutor

warnings.filterwarnings("ignore", category=UserWarning)

try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class DebateAI:
    def __init__(self):
        self.GOOGLE_CREDENTIALS_PATH = "YOUR LINK TO .JSON FILE CREDENTIALS"
        self.GROQ_API_KEY = "INPUT YOUR API KEY"
        self.ELEVENLABS_API_KEY = "INPUT YOUR API KEY"
        
        self.setup_environment()
        self.load_asr_corrector()
        self.initialize_apis()
        self.configure_audio()
        self.initialize_state()

        self.executor = ThreadPoolExecutor(max_workers=3)
        self.response_queue = queue.Queue()
        self.audio_buffer = []
        self.is_processing = False
        
        self.max_turns = 8
        self.current_emotion = 'neutral'
        self.listening = True
        self.response_lock = threading.Lock()
        
        self.voice_settings = {
            'neutral': {'stability': 0.7, 'similarity_boost': 0.8, 'style': 0.5, 'speed': 1.0},
            'excited': {'stability': 0.6, 'similarity_boost': 0.7, 'style': 0.8, 'speed': 1.2},
            'calm': {'stability': 0.8, 'similarity_boost': 0.9, 'style': 0.3, 'speed': 0.9},
            'assertive': {'stability': 0.6, 'similarity_boost': 0.7, 'style': 0.7, 'speed': 1.1},
            'thoughtful': {'stability': 0.9, 'similarity_boost': 0.8, 'style': 0.4, 'speed': 0.95}
        }

    def setup_environment(self):
        os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = self.GOOGLE_CREDENTIALS_PATH
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

    def analyze_emotion(self, text):
        scores = self.sentiment_analyzer.polarity_scores(text)
        compound = scores['compound']
        pos = scores['pos']
        neg = scores['neg']
        neu = scores['neu']

        text_length = len(text.split())
        exclamation_count = text.count('!')
        intensity = min(1.0, (text_length / 10) + (exclamation_count * 0.2)) 

        if compound >= 0.5:
            if pos >= 0.75:
                return 'excited' 
            elif pos >= 0.5 and intensity > 0.7:
                return 'confident'  
            else:
                return 'assertive'  
        elif compound <= -0.5:
            if neg >= 0.75:
                return 'angry'  
            elif neg >= 0.5 and intensity < 0.3:
                return 'thoughtful'  
            else:
                return 'calm' 
        else: 
            if pos > neg and pos > 0.3:
                return 'hopeful'  
            elif neg > pos and neg > 0.3:
                return 'skeptical' 
            elif neu >= 0.8 and intensity < 0.5:
                return 'neutral'  
            else:
                return 'curious'  

        return 'neutral'

    def get_response_prompt(self, text, emotion):
        prompts = {
            'excited': "Respond enthusiastically but concisely to the counterargument:",
            'calm': "Provide a measured, brief response to:",
            'assertive': "Give a confident, direct counter to:",
            'thoughtful': "Share a brief, reflective perspective on:",
            'neutral': "Respond clearly and concisely to:"
        }
        return f"{prompts[emotion]} {text[:200]}"

    def load_asr_corrector(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.tokenizer = GPT2Tokenizer.from_pretrained("./tokenizer")
            self.asr_corrector = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
            self.asr_corrector.resize_token_embeddings(len(self.tokenizer))
            self.asr_corrector.load_state_dict(torch.load("./asr_corrector.pth", map_location=device), strict=False)
            self.asr_corrector.eval()
        except Exception as e:
            print(f"Error loading ASR corrector: {e}")
            exit()

    def initialize_apis(self):
        self.speech_client = speech.SpeechClient()
        self.groq_client = Groq(api_key=self.GROQ_API_KEY)
        self.elevenlabs_client = ElevenLabs(api_key=self.ELEVENLABS_API_KEY)

    def configure_audio(self):
        self.audio_interface = pyaudio.PyAudio()
        self.stream = self.audio_interface.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=16000,
            input=True,
            frames_per_buffer=2048,
            stream_callback=self.audio_callback,
            start=False
        )
        self.audio_queue = queue.Queue()

    def initialize_state(self):
        self.debate_state = {
            'turns': 0,
            'user_arguments': [],
            'bot_arguments': [],
            'start_time': time.time(),
            'user_sentiments': [] 
        }

    def audio_callback(self, in_data, frame_count, time_info, status):
        if not self.is_processing:
            self.audio_queue.put(in_data)
        return (in_data, pyaudio.paContinue)

    async def process_audio_async(self):
        config = speech.RecognitionConfig(
            encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
            sample_rate_hertz=16000,
            language_code="en-US",
            enable_automatic_punctuation=True
        )
        streaming_config = speech.StreamingRecognitionConfig(
            config=config,
            interim_results=True
        )

        def audio_generator():
            silence_counter = 0
            while self.listening:
                try:
                    chunk = self.audio_queue.get(timeout=0.05)
                    silence_counter = 0
                    yield speech.StreamingRecognizeRequest(audio_content=chunk)
                except queue.Empty:
                    silence_counter += 1
                    if silence_counter > 5:
                        yield speech.StreamingRecognizeRequest(audio_content=b'')
                    time.sleep(0.05)

        responses = self.speech_client.streaming_recognize(streaming_config, audio_generator())
        
        try:
            for response in responses:
                if not response.results:
                    continue

                result = response.results[0]
                if result.is_final:
                    await self.process_transcript_async(result.alternatives[0].transcript)
        except Exception as e:
            print(f"Streaming error: {e}")

    async def process_transcript_async(self, text):
        print(f"\nYou: {text}", flush=True)
        self.is_processing = True
        
        # Check if user concedes defeat
        if re.search(r"(i\s*(lost|give\s*up|concede|surrender)\s*(the\s*debate)?)", text.lower()):
            print("\nYou’ve conceded the debate. Ending now...")
            self.listening = False
            self.debate_state['user_arguments'].append(text)
            self.is_processing = False
            return

        # Process user input as part of the debate
        corrected_text = await asyncio.get_event_loop().run_in_executor(
            self.executor, self.correct_asr, text
        )
        sentiment_score = self.sentiment_analyzer.polarity_scores(corrected_text)
        self.debate_state['user_arguments'].append(corrected_text)
        self.debate_state['user_sentiments'].append(sentiment_score['compound'])
        
        await self.generate_and_stream_response(corrected_text)
        self.is_processing = False

    def correct_asr(self, text):
        inputs = self.tokenizer(
            f"correct: {text} [SEP]",
            return_tensors="pt",
            max_length=256,
            truncation=True,
            padding='max_length'
        ).to(self.asr_corrector.device)

        with torch.no_grad():
            outputs = self.asr_corrector.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=300,
                num_beams=2,
                early_stopping=True
            )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).split("[SEP]")[-1].strip()

    async def generate_and_stream_response(self, text):
        user_emotion = self.analyze_emotion(text)
        prompt = self.get_response_prompt(text, user_emotion)
        
        full_response = ""
        print("\nBot: ", end='', flush=True)
        
        response = self.groq_client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[
                {"role": "system", "content": "You are a concise debate partner. Keep responses under 50 words while maintaining clarity and impact. You are not easy to convience in the ongoing debate topic."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=50,
            stream=True
        )

        for chunk in response:
            content = chunk.choices[0].delta.content
            if content:
                full_response += content
                print(content, end='', flush=True)
        
        print("\n", flush=True)
        
        bot_emotion = self.analyze_emotion(full_response)
        voice_settings = self.voice_settings[bot_emotion]
        
        try:
            print(f"Responding with {bot_emotion} tone...", flush=True)
            audio_stream = await asyncio.get_event_loop().run_in_executor(
                self.executor,
                lambda: self.elevenlabs_client.generate(
                    text=full_response,
                    voice="Rachel",
                    model="eleven_turbo_v2",
                    stream=True,
                    voice_settings=voice_settings
                )
            )
            stream(audio_stream)
            print("\nYour turn to speak.", flush=True)
        except Exception as e:
            print(f"\nAudio error: {e}")
        
        self.debate_state['turns'] += 1
        self.debate_state['bot_arguments'].append(full_response)

    async def debate_loop_async(self):
        print("\n--- Debate Started --- (Speak now)")
        self.stream.start_stream()
        
        try:
            while self.listening:
                await self.process_audio_async()
        except KeyboardInterrupt:
            self.listening = False
            print("\nDebate interrupted by user.")
        except Exception as e:
            print(f"Debate loop error: {e}")
        finally:
            self.stream.stop_stream()
            self.stream.close()
            self.audio_interface.terminate()
            self.executor.shutdown(wait=True)
            self.show_statistics()

    def debate_loop(self):
        asyncio.run(self.debate_loop_async())

    def show_statistics(self):
        duration = time.time() - self.debate_state['start_time']
        turns = self.debate_state['turns']
        user_args = len(self.debate_state['user_arguments'])
        
        if self.debate_state['user_sentiments']:
            avg_sentiment = sum(self.debate_state['user_sentiments']) / len(self.debate_state['user_sentiments'])
            sentiment_desc = "positive" if avg_sentiment > 0 else "negative" if avg_sentiment < 0 else "neutral"
        else:
            avg_sentiment = 0
            sentiment_desc = "neutral"

        avg_arg_length = sum(len(arg.split()) for arg in self.debate_state['user_arguments']) / max(user_args, 1)

        print(f"\n--- Debate Ended in {duration:.1f}s ---")
        print(f"Total Turns: {turns}")
        print(f"Your Arguments: {user_args}")
        print(f"Average Sentiment: {avg_sentiment:.2f} ({sentiment_desc})")
        print(f"Average Argument Length: {avg_arg_length:.1f} words")
        print("\nPerformance Summary:")
        if turns == 0:
            print("You didn’t make any arguments before conceding.")
        elif avg_sentiment < -0.3:
            print("Your arguments leaned negative, possibly weakening your stance.")
        elif avg_arg_length < 5:
            print("Your arguments were short, potentially lacking depth.")
        else:
            print("You held a solid debate with balanced arguments!")

if __name__ == "__main__":
    debate = DebateAI()
    debate.debate_loop()
