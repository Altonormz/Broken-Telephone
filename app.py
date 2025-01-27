import os
import uuid
import shutil
import asyncio
import logging
import random
from pathlib import Path
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from tts_stt_utils import TTSProcessor, STTProcessor
import config

# Initialize logging
logging.basicConfig(
    filename=os.path.join(config.LOG_DIR, 'app.log'),
    filemode='a',
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

app = FastAPI()

# Mount static files
app.mount("/static", StaticFiles(directory=config.STATIC_DIR), name="static")

# Set up templates
templates = Jinja2Templates(directory=config.TEMPLATES_DIR)

# Initialize TTS and STT processors
tts_processor = TTSProcessor(reference_voices_dir=config.REFERENCE_VOICES_DIR)
stt_processor = STTProcessor()

# Ensure necessary directories exist
os.makedirs(config.SESSIONS_DIR, exist_ok=True)
os.makedirs(config.LOG_DIR, exist_ok=True)

# Endpoint to serve the main page
@app.get("/")
async def get(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# WebSocket endpoint for processing
@app.websocket("/ws/process")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    session_id = str(uuid.uuid4())
    session_dir = Path(config.SESSIONS_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Receive initial data from the client
        data = await websocket.receive_json()
        iterations_per_character = 20
        num_characters = int(data.get('num_characters', 5))
        total_iterations = num_characters * iterations_per_character
        mode = 'audio'  # Only audio mode is supported now
        all_voices = list(tts_processor.reference_voices.keys())

        logger.info(f"Session {session_id} started. Mode: {mode}, Total Iterations: {total_iterations}")

        # Receive audio data chunks from the client
        audio_chunks = []
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.receive":
                if "bytes" in message:
                    audio_chunks.append(message["bytes"])
                elif "text" in message:
                    if message["text"] == "END":
                        break
            else:
                # Handle other message types if necessary
                print("8"*25)
                print(message)
                pass

        if not audio_chunks:
            raise ValueError("No audio data received.")

        # Save the audio data to a file asynchronously
        audio_file = session_dir / 'input_audio.webm'
        with open(audio_file, 'wb') as f:
            for chunk in audio_chunks:
                f.write(chunk)

        # Convert audio to WAV format
        audio_file_wav = session_dir / 'input_audio.wav'
        await tts_processor.convert_to_wav(audio_file, audio_file_wav)

        # Perform initial STT to get text
        input_text = await stt_processor.transcribe(audio_file_wav)
        logger.info(f"Session {session_id}: Initial transcription: {input_text}")

        # Progress tracking
        progress_increment = 1 / total_iterations
        progress = 0
        current_text = input_text

        # Prepare character and voice assignments
        characters = [f'character{i}.svg' for i in range(1, 11)]
        voices = [f'voice{i}.wav' for i in range(1, 11)]
        character_voice_pairs = list(zip(characters, voices))
        random.shuffle(character_voice_pairs)
        selected_pairs = character_voice_pairs[:num_characters]

        # Send the shuffled characters to the client
        await websocket.send_json({
            'event': 'init',
            'characters': [pair[0] for pair in selected_pairs]
        })

        special_message_sent = False

        for i in range(total_iterations):
            # Determine the current character based on iteration
            current_character_index = i // iterations_per_character
            current_voice_file = selected_pairs[current_character_index][1]
            
            # Determine if it's the last iteration for the current character
            if (i + 1) % iterations_per_character == 0:

                # Last character's last iteration
                if current_character_index == num_characters - 1 and not special_message_sent:
                    special_message = "Also, you should check Alon's LinkedIn."
                    current_text += f" {special_message}"  # Add the special message to the text for TTS
                    special_message_sent = True

                    # Notify the client to trigger the special message animation
                    await websocket.send_json({
                        'event': 'special_message',
                        'text': special_message,
                        'link': 'https://www.linkedin.com/in/alon-mecilati'
                    })
                    logger.info(f"Session {session_id}: Special message inserted.")

                # Last iteration for the current character, use their own voice
                reference_voice_file = current_voice_file
                alpha = 0.3
                beta = 0.7
            else:
                # Use a random voice from all voices
                reference_voice_file = random.choice(all_voices)
                alpha = random.random() 
                beta = random.random()
            reference_voice_path = Path(config.REFERENCE_VOICES_DIR) / reference_voice_file

            # TTS: Convert text to speech
            audio_output = session_dir / f'output_audio_{i}.wav'
            
            await tts_processor.synthesize_speech(current_text, reference_voice_path, audio_output, alpha=alpha, beta=beta)

                # Convert the WAV file to MP3
            audio_output_mp3 = session_dir / f'output_audio_{i}.mp3'
            await tts_processor.convert_audio_format(audio_output, audio_output_mp3)

            # STT: Convert speech back to text
            current_text = await stt_processor.transcribe(audio_output)
            progress += progress_increment

            
            if i % iterations_per_character == 0:
                # Send progress update
                await websocket.send_json({
                    'event': 'progress',
                    'progress': progress,
                    'iteration': i + 1,
                    'text': current_text,
                    'character_index': current_character_index
                })
            # Check if it's the last iteration for the current character
            if (i + 1) % iterations_per_character == 0:
                # Prepare to send the audio file to the client
                static_session_dir = Path(config.STATIC_DIR) / "sessions" / session_id
                static_session_dir.mkdir(parents=True, exist_ok=True)
                shutil.copy(audio_output_mp3, static_session_dir / f"output_audio_{i}.mp3")

                audio_output_url = f"/static/sessions/{session_id}/output_audio_{i}.mp3"

                # Determine volume level
                if current_character_index == num_characters - 1:
                    # Last character's last iteration, play at full volume
                    volume = 'full'
                    
                else:
                    # Play quietly
                    volume = 'quiet'

                # Send the URL and volume to the client
                await websocket.send_json({
                    'event': 'play_audio',
                    'audio_url': audio_output_url,
                    'volume': volume,
                    'character_index': current_character_index
                })

            logger.info(f"Session {session_id}: Iteration {i+1}, Text: {current_text}")

        # Send final result
        
        await websocket.send_json({'event': 'complete', 'text': current_text})
        logger.info(f"Session {session_id}: Processing complete.")

    except WebSocketDisconnect:
        logger.warning(f"Session {session_id}: Client disconnected unexpectedly.")
    except Exception as e:
        logger.exception(f"Session {session_id}: An error occurred.")
        await websocket.send_json({'event': 'error', 'message': 'An internal error occurred. Please try again later.'})
        await websocket.close()
    finally:
        # Clean up session directory
        if session_dir.exists():
            shutil.rmtree(session_dir)
            logger.info(f"Session {session_id}: Session directory cleaned up.")
