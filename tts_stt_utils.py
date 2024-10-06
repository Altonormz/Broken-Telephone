import os
import random
import asyncio
import subprocess
from pathlib import Path
import logging

import torch
import torchaudio
import librosa
import soundfile as sf
import yaml
from munch import Munch
from nltk.tokenize import word_tokenize
import phonemizer
from StyleTTS2.models import *
from config import STYLETTS2_DIR
from StyleTTS2.Utils.PLBERT.util import load_plbert
from StyleTTS2.Modules.diffusion.sampler import DiffusionSampler, ADPM2Sampler, KarrasSchedule
from StyleTTS2.models import build_model
from StyleTTS2.utils import recursive_munch
from StyleTTS2.text_utils import TextCleaner

# Suppress specific warning categories
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)

def length_to_mask(lengths):
    """Gets the mask of the max length"""
    mask = torch.arange(lengths.max()).unsqueeze(0).expand(lengths.shape[0], -1).type_as(lengths)
    mask = torch.gt(mask + 1, lengths.unsqueeze(1))
    return mask

class TTSProcessor:
    def __init__(self, reference_voices_dir):
        self.reference_voices_dir = Path(reference_voices_dir)
        self.reference_voices = self._load_reference_voices()
        self.initialize_model()

    def initialize_model(self):
        # Setup
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        # Initialize text cleaner
        self.text_cleaner = TextCleaner()

        # Mel Spectrogram transformation
        self.to_mel = torchaudio.transforms.MelSpectrogram(
            n_mels=80, n_fft=2048, win_length=1200, hop_length=300
        )
        self.mean, self.std = -4, 4

        # Load models and configurations
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        config_path = os.path.join(STYLETTS2_DIR, 'Models', 'LibriTTS', 'config.yml')
        config = yaml.safe_load(open(config_path))

        # Load pretrained ASR model
        ASR_config = os.path.join(STYLETTS2_DIR, config.get('ASR_config', ''))
        ASR_path = os.path.join(STYLETTS2_DIR, config.get('ASR_path', ''))
        text_aligner = load_ASR_models(ASR_path, ASR_config)

        # Load pretrained F0 model
        F0_path = os.path.join(STYLETTS2_DIR, config.get('F0_path', ''))
        pitch_extractor = load_F0_models(F0_path)

        # Load BERT model
        BERT_path = os.path.join(STYLETTS2_DIR, config.get('PLBERT_dir', ''))
        plbert = load_plbert(BERT_path)
        self.model_params = recursive_munch(config['model_params'])
        self.model = build_model(self.model_params, text_aligner, pitch_extractor, plbert)
        _ = [self.model[key].eval() for key in self.model]
        _ = [self.model[key].to(self.device) for key in self.model]

        # Load model parameters
        model_weights_path = os.path.join(STYLETTS2_DIR, 'Models', 'LibriTTS', 'epochs_2nd_00020.pth')
        params_whole = torch.load(model_weights_path, map_location='cpu')
        params = params_whole['net']

        for key in self.model:
            if key in params:
                try:
                    self.model[key].load_state_dict(params[key])
                except:
                    from collections import OrderedDict
                    state_dict = params[key]
                    new_state_dict = OrderedDict()
                    for k, v in state_dict.items():
                        name = k[7:]  # remove `module.`
                        new_state_dict[name] = v
                    # Load parameters
                    self.model[key].load_state_dict(new_state_dict, strict=False)
        _ = [self.model[key].eval() for key in self.model]

        # Load sampler
        self.sampler = DiffusionSampler(
            self.model['diffusion'].diffusion,
            sampler=ADPM2Sampler(),
            sigma_schedule=KarrasSchedule(sigma_min=0.0001, sigma_max=3.0, rho=9.0),
            clamp=False
        )

        # Load phonemizer
        self.global_phonemizer = phonemizer.backend.EspeakBackend(
            language='en-us', preserve_punctuation=True, with_stress=True
        )

    def _load_reference_voices(self):
        voices = {}
        for voice_file in self.reference_voices_dir.glob('*.wav'):
            voices[voice_file.name] = voice_file
        if not voices:
            logger.error("No reference voices found.")
            raise FileNotFoundError("No reference voices found in the directory.")
        return voices

    def preprocess(self, wave):
        """Turns wave to mel tensor"""
        wave_tensor = torch.from_numpy(wave).float()
        mel_tensor = self.to_mel(wave_tensor)
        mel_tensor = (torch.log(1e-5 + mel_tensor.unsqueeze(0)) - self.mean) / self.std
        return mel_tensor

    def compute_style(self, path):
        """Computes the style vector for a given audio file"""
        wave, sr = librosa.load(path, sr=24000)
        audio, index = librosa.effects.trim(wave, top_db=30)
        if sr != 24000:
            audio = librosa.resample(audio, sr, 24000)
        mel_tensor = self.preprocess(audio).to(self.device)
        with torch.no_grad():
            ref_s = self.model['style_encoder'](mel_tensor.unsqueeze(1))
            ref_p = self.model['predictor_encoder'](mel_tensor.unsqueeze(1))
        return torch.cat([ref_s, ref_p], dim=1)

    def inference(self, text, ref_s, alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1):
        # Preprocess text
        text = text.replace('"', '')
        ps = self.global_phonemizer.phonemize([text])
        ps = word_tokenize(ps[0])
        ps = ' '.join(ps)
        tokens = self.text_cleaner(ps)
        tokens.insert(0, 0)

        tokens = torch.LongTensor(tokens).to(self.device).unsqueeze(0)
        max_length = 512

        if tokens.shape[1] > max_length:
            tokens = tokens[:, :max_length]

        with torch.no_grad():
            # Process text
            input_lengths = torch.LongTensor([tokens.shape[-1]]).to(self.device)
            text_mask = length_to_mask(input_lengths).to(self.device)
            t_en = self.model['text_encoder'](tokens, input_lengths, text_mask)
            bert_dur = self.model['bert'](tokens, attention_mask=(~text_mask).int())
            d_en = self.model['bert_encoder'](bert_dur).transpose(-1, -2)

            s_pred = self.sampler(
                noise=torch.randn((1, 256)).unsqueeze(1).to(self.device),
                embedding=bert_dur,
                embedding_scale=embedding_scale,
                features=ref_s,  # Reference from the same speaker as the embedding
                num_steps=diffusion_steps
            ).squeeze(1)

            s = s_pred[:, 128:]
            ref = s_pred[:, :128]

            ref = alpha * ref + (1 - alpha) * ref_s[:, :128]
            s = beta * s + (1 - beta) * ref_s[:, 128:]

            d = self.model['predictor'].text_encoder(
                d_en, s, input_lengths, text_mask
            )

            x, _ = self.model['predictor'].lstm(d)
            duration = self.model['predictor'].duration_proj(x)

            duration = torch.sigmoid(duration).sum(axis=-1)
            pred_dur = torch.round(duration.squeeze()).clamp(min=1)

            pred_aln_trg = torch.zeros(
                input_lengths.item(), int(pred_dur.sum().item())
            )
            c_frame = 0
            for i in range(pred_aln_trg.size(0)):
                dur = int(pred_dur[i].item())
                pred_aln_trg[i, c_frame:c_frame + dur] = 1
                c_frame += dur

            # Encode prosody
            en = (d.transpose(-1, -2) @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(en)
                asr_new[:, :, 0] = en[:, :, 0]
                asr_new[:, :, 1:] = en[:, :, :-1]
                en = asr_new

            F0_pred, N_pred = self.model['predictor'].F0Ntrain(en, s)

            asr = (t_en @ pred_aln_trg.unsqueeze(0).to(self.device))
            if self.model_params.decoder.type == "hifigan":
                asr_new = torch.zeros_like(asr)
                asr_new[:, :, 0] = asr[:, :, 0]
                asr_new[:, :, 1:] = asr[:, :, :-1]
                asr = asr_new

            out = self.model['decoder'](
                asr, F0_pred, N_pred, ref.squeeze().unsqueeze(0)
            )

        # Return synthesized speech
        return out.squeeze().cpu().numpy()[..., :-50]

    async def synthesize_speech(self, text, reference_voice_path, output_path, alpha=0.3, beta=0.7, diffusion_steps=10, embedding_scale=1):
        try:
            # Compute style from the reference voice
            ref_s = self.compute_style(str(reference_voice_path))

            # Perform inference
            wav = self.inference(
                text,
                ref_s,
                alpha=alpha,
                beta=beta,
                diffusion_steps=diffusion_steps,
                embedding_scale=embedding_scale
            )

            # Save the output audio
            sf.write(output_path, wav, 24000)
            logger.info(f"TTS synthesis successful for text: {text}")
        except Exception as e:
            logger.exception("Error in TTS synthesis.")
            raise

    async def convert_to_wav(self, input_file, output_file):
        try:
            # Convert input audio to WAV format using FFmpeg
            command = [
                'ffmpeg',
                '-y',  # Overwrite output files without asking
                '-i', str(input_file),
                '-ar', '16000',  # Set sample rate
                '-ac', '1',      # Set number of audio channels
                str(output_file)
            ]
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                raise Exception(f"FFmpeg error: {stderr.decode()}")
            logger.info(f"Converted {input_file} to {output_file}")
        except Exception as e:
            logger.exception("Error in converting audio to WAV.")
            raise

    async def convert_audio_format(self, input_file, output_file):
        try:
            # Convert input audio to MP3 format using FFmpeg
            command = [
                'ffmpeg',
                '-y',  # Overwrite output files without asking
                '-i', str(input_file),
                '-vn',  # No video
                '-ar', '44100',  # Set sample rate
                '-ac', '2',      # Set number of audio channels
                '-b:a', '192k',  # Set bitrate
                str(output_file)
            ]
            process = await asyncio.create_subprocess_exec(
                *command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            if process.returncode != 0:
                logger.error(f"FFmpeg error: {stderr.decode()}")
                raise Exception(f"FFmpeg error: {stderr.decode()}")
            logger.info(f"Converted {input_file} to {output_file}")
        except Exception as e:
            logger.exception("Error in converting audio format.")
            raise


class STTProcessor:
    def __init__(self):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        model_id = "openai/whisper-small"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            low_cpu_mem_usage=True
        ).to(self.device)

        self.processor = AutoProcessor.from_pretrained(model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=self.processor.tokenizer,
            feature_extractor=self.processor.feature_extractor,
            torch_dtype=self.torch_dtype,
            device=self.device,
            generate_kwargs = {"language":"en"}
        )

    async def transcribe(self, audio_file):
        import asyncio
        loop = asyncio.get_event_loop()

        try:
            # Run the transcription in a separate thread to avoid blocking
            result = await loop.run_in_executor(
                None,
                self.pipe,
                str(audio_file),
            )
            text = result["text"]
            logger.info(f"Transcription successful for audio: {audio_file}")
            return text
        except Exception as e:
            logger.exception("Error in STT transcription.")
            raise