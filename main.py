
"""
Conversational AI Pipeline
A modular real-time conversation system using Whisper STT, Text-Generation-WebUI, and TTS
"""

import os
import sys
import time
import json
import wave
import threading
import queue
from pathlib import Path
from typing import Optional, Dict, Any

# Audio processing
import pyaudio
import soundfile as sf
import sounddevice as sd

# Whisper for STT
import whisper
from faster_whisper import WhisperModel

# HTTP requests for text-gen-webui
import requests

# TTS
from TTS.api import TTS


class ConversationalAI:
    def __init__(self, config_path: str = "config.json"):
        """Initialize the conversational AI system."""
        self.config = self.load_config(config_path)
        self.setup_components()
        
        # Audio settings
        self.audio_queue = queue.Queue()
        self.is_recording = False
        self.is_speaking = False
        
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file or create default."""
        default_config = {
            "whisper": {
                "model": "base.en",
                "use_faster_whisper": True,
                "device": "cpu"  # Change to "cuda" if you have GPU
            },
            "text_generation": {
                "base_url": "http://localhost:43031",
                "endpoints": {
                    "generate": "/api/v1/generate",
                    "chat": "/api/v1/chat/completions"
                },
                "parameters": {
                    "max_tokens": 150,
                    "temperature": 0.7,
                    "top_p": 0.9,
                    "stop": ["Human:", "User:"]
                }
            },
            "tts": {
                "model_name": "tts_models/en/ljspeech/tacotron2-DDC",
                "use_coqui": True,
                "fallback_espeak": True
            },
            "audio": {
                "sample_rate": 16000,
                "channels": 1,
                "chunk_size": 1024,
                "record_seconds": 5,
                "silence_threshold": 0.01,
                "silence_duration": 2.0
            }
        }
        
        if os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config = json.load(f)
            # Merge with defaults for missing keys
            return self._merge_configs(default_config, config)
        else:
            # Create default config file
            with open(config_path, 'w') as f:
                json.dump(default_config, f, indent=2)
            return default_config
    
    def _merge_configs(self, default: dict, user: dict) -> dict:
        """Recursively merge user config with defaults."""
        result = default.copy()
        for key, value in user.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value
        return result
    
    def setup_components(self):
        """Initialize all AI components."""
        print("Initializing components...")
        
        # Setup Whisper STT
        self.setup_whisper()
        
        # Setup Text Generation
        self.setup_text_generation()
        
        # Setup TTS
        self.setup_tts()
        
        # Setup Audio
        self.setup_audio()
        
        print("All components initialized successfully!")
    
    def setup_whisper(self):
        """Initialize Whisper for speech-to-text."""
        whisper_config = self.config["whisper"]
        
        if whisper_config["use_faster_whisper"]:
            print(f"Loading Faster Whisper model: {whisper_config['model']}")
            
            # Detect available device
            device = self._detect_device(whisper_config.get("device", "cpu"))
            if whisper_config.get("device") == "auto":
                device = self._detect_device("cuda")  # Try GPU first in auto mode
            compute_type = whisper_config.get("compute_type", "float16")
            
            print(f"Using device: {device} with compute_type: {compute_type}")
            
            try:
                self.whisper_model = WhisperModel(
                    whisper_config["model"],
                    device=device,
                    compute_type=compute_type if device != "cpu" else "int8"
                )
                print("Whisper model loaded successfully!")
            except Exception as e:
                print(f"Failed to load model on {device}: {e}")
                print("Falling back to CPU...")
                self.whisper_model = WhisperModel(
                    whisper_config["model"],
                    device="cpu",
                    compute_type="int8"
                )
        else:
            print(f"Loading OpenAI Whisper model: {whisper_config['model']}")
            self.whisper_model = whisper.load_model(whisper_config["model"])
    
    def _detect_device(self, preferred_device: str = "cpu") -> str:
        """Detect the best available device for inference."""
        print("Detecting available compute devices...")
        
        # Check for ROCm (AMD GPU)
        try:
            import torch
            if torch.cuda.is_available() and "AMD" in torch.cuda.get_device_name(0):
                print("ROCm (AMD GPU) detected")
                return "cuda"  # faster-whisper uses "cuda" for both NVIDIA and AMD
        except:
            pass
        
        # Check for CUDA (NVIDIA GPU) 
        try:
            import torch
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                print(f"CUDA device detected: {device_name}")
                return "cuda"
        except:
            pass
            
        # Check if user specified cuda but it's not available
        if preferred_device == "cuda":
            print("CUDA requested but not available, falling back to CPU")
        
        print("Using CPU for inference")
        return "cpu"
    
    def setup_text_generation(self):
        """Setup connection to text-generation-webui."""
        self.text_gen_config = self.config["text_generation"]
        
        # Auto-detect API endpoints
        self._detect_api_endpoints()
        
        # Test connection
        try:
            response = requests.get(f"{self.text_gen_config['base_url']}/api/v1/model")
            if response.status_code == 200:
                print("Connected to text-generation-webui successfully!")
                model_info = response.json()
                print(f"Loaded model: {model_info.get('result', 'Unknown')}")
            else:
                print(f"Warning: text-generation-webui responded with status {response.status_code}")
        except requests.exceptions.RequestException as e:
            print(f"Warning: Could not connect to text-generation-webui: {e}")
            print("Make sure ./start_linux.sh is running in your text-generation-webui folder")
    
    def _detect_api_endpoints(self):
        """Detect the correct API endpoints for the running text-gen-webui version."""
        base_url = self.text_gen_config['base_url']
        
        # Test different possible endpoints
        endpoints_to_test = [
            "/v1/completions",           # OpenAI-compatible endpoint (newer versions)
            "/api/v1/generate",          # Legacy endpoint
            "/v1/chat/completions",      # Chat endpoint
            "/api/v1/chat/completions"   # Legacy chat endpoint
        ]
        
        print("Detecting API endpoints...")
        
        for endpoint in endpoints_to_test:
            try:
                # Test with a simple request
                test_payload = {
                    "prompt" if "chat" not in endpoint else "messages": 
                        "test" if "chat" not in endpoint else [{"role": "user", "content": "test"}],
                    "max_tokens": 1,
                    "temperature": 0.1
                }
                
                response = requests.post(f"{base_url}{endpoint}", json=test_payload, timeout=5)
                
                if response.status_code in [200, 422]:  # 422 might indicate wrong format but endpoint exists
                    print(f"✓ Found working endpoint: {endpoint}")
                    self.text_gen_config['working_endpoint'] = endpoint
                    return
                    
            except requests.exceptions.RequestException:
                continue
        
        print("⚠ No working endpoints detected, using fallback")
        self.text_gen_config['working_endpoint'] = "/v1/completions"
    
    def setup_tts(self):
        """Initialize TTS system."""
        tts_config = self.config["tts"]
        
        if tts_config["use_coqui"]:
            try:
                print(f"Loading TTS model: {tts_config['model_name']}")
                self.tts = TTS(model_name=tts_config["model_name"])
                print("Coqui TTS loaded successfully!")
            except Exception as e:
                print(f"Failed to load Coqui TTS: {e}")
                if tts_config["fallback_espeak"]:
                    print("Falling back to espeak-ng")
                    self.tts = None  # Will use espeak fallback
                else:
                    raise
        else:
            self.tts = None
    
    def setup_audio(self):
        """Setup audio recording and playback."""
        self.audio_config = self.config["audio"]
        
        # Initialize PyAudio
        self.audio = pyaudio.PyAudio()
        
        # Print available audio devices
        print("\nAvailable audio devices:")
        for i in range(self.audio.get_device_count()):
            info = self.audio.get_device_info_by_index(i)
            print(f"  {i}: {info['name']} ({'Input' if info['maxInputChannels'] > 0 else ''}{'Output' if info['maxOutputChannels'] > 0 else ''})")
    
    def record_audio(self, duration: Optional[float] = None) -> str:
        """Record audio and save to temporary file."""
        if duration is None:
            duration = self.audio_config["record_seconds"]
        
        print(f"Recording for {duration} seconds... (Press Ctrl+C to stop early)")
        
        # Recording parameters
        chunk = self.audio_config["chunk_size"]
        sample_rate = self.audio_config["sample_rate"]
        channels = self.audio_config["channels"]
        
        # Start recording
        stream = self.audio.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            frames_per_buffer=chunk
        )
        
        frames = []
        try:
            for i in range(0, int(sample_rate / chunk * duration)):
                data = stream.read(chunk)
                frames.append(data)
        except KeyboardInterrupt:
            print("\nRecording stopped by user")
        
        stream.stop_stream()
        stream.close()
        
        # Save to temporary file
        temp_audio_path = "temp_recording.wav"
        with wave.open(temp_audio_path, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(self.audio.get_sample_size(pyaudio.paInt16))
            wf.setframerate(sample_rate)
            wf.writeframes(b''.join(frames))
        
        print("Recording complete!")
        return temp_audio_path
    
    def speech_to_text(self, audio_path: str) -> str:
        """Convert speech to text using Whisper."""
        print("Converting speech to text...")
        
        if isinstance(self.whisper_model, WhisperModel):
            # Faster Whisper
            segments, info = self.whisper_model.transcribe(audio_path)
            text = " ".join([segment.text for segment in segments])
        else:
            # OpenAI Whisper
            result = self.whisper_model.transcribe(audio_path)
            text = result["text"]
        
        return text.strip()
    
    def generate_response(self, user_input: str) -> str:
        """Generate response using text-generation-webui."""
        print("Generating response...")
        
        # Get the working endpoint
        endpoint = self.text_gen_config.get('working_endpoint', '/v1/completions')
        url = f"{self.text_gen_config['base_url']}{endpoint}"
        
        print(f"Using endpoint: {endpoint}")
        
        # Prepare payload based on endpoint type
        if "chat" in endpoint:
            # Chat completions format
            payload = {
                "messages": [
                    {"role": "user", "content": user_input}
                ],
                "max_tokens": self.text_gen_config["parameters"]["max_tokens"],
                "temperature": self.text_gen_config["parameters"]["temperature"],
                "top_p": self.text_gen_config["parameters"]["top_p"],
                "stop": self.text_gen_config["parameters"]["stop"]
            }
        else:
            # Standard completions format
            payload = {
                "prompt": f"Human: {user_input}\nAssistant:",
                "max_tokens": self.text_gen_config["parameters"]["max_tokens"],
                "temperature": self.text_gen_config["parameters"]["temperature"],
                "top_p": self.text_gen_config["parameters"]["top_p"],
                "stop": self.text_gen_config["parameters"]["stop"]
            }
        
        try:
            response = requests.post(url, json=payload, timeout=30)
            print(f"Response status: {response.status_code}")
            
            if response.status_code == 200:
                result = response.json()
                
                # Parse response based on endpoint type
                if "chat" in endpoint:
                    # Chat completion response
                    generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    # Standard completion response
                    generated_text = result.get("choices", [{}])[0].get("text", "")
                    if not generated_text:
                        # Try legacy format
                        generated_text = result.get("results", [{}])[0].get("text", "")
                
                # Clean up the response
                generated_text = generated_text.strip()
                if generated_text.startswith("Assistant:"):
                    generated_text = generated_text[10:].strip()
                if generated_text.startswith("Human:"):
                    # Remove any leaked human input
                    lines = generated_text.split('\n')
                    generated_text = '\n'.join([line for line in lines if not line.startswith("Human:")])
                
                return generated_text if generated_text else "I'm sorry, I didn't generate a response."
            
            else:
                # Try alternative endpoints on failure
                return self._try_alternative_endpoints(user_input)
                
        except requests.exceptions.RequestException as e:
            print(f"Connection error: {e}")
            return self._try_alternative_endpoints(user_input)
        except Exception as e:
            return f"Error generating response: {e}"
    
    def _try_alternative_endpoints(self, user_input: str) -> str:
        """Try alternative endpoints if the primary one fails."""
        alternative_endpoints = ["/v1/completions", "/api/v1/generate", "/v1/chat/completions"]
        base_url = self.text_gen_config['base_url']
        
        for endpoint in alternative_endpoints:
            if endpoint == self.text_gen_config.get('working_endpoint'):
                continue  # Skip the one that just failed
                
            try:
                print(f"Trying alternative endpoint: {endpoint}")
                
                if "chat" in endpoint:
                    payload = {
                        "messages": [{"role": "user", "content": user_input}],
                        "max_tokens": 100,
                        "temperature": 0.7
                    }
                else:
                    payload = {
                        "prompt": f"Human: {user_input}\nAssistant:",
                        "max_tokens": 100,
                        "temperature": 0.7
                    }
                
                response = requests.post(f"{base_url}{endpoint}", json=payload, timeout=15)
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Update working endpoint
                    self.text_gen_config['working_endpoint'] = endpoint
                    print(f"✓ Successfully switched to {endpoint}")
                    
                    # Extract response
                    if "chat" in endpoint:
                        generated_text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                    else:
                        generated_text = result.get("choices", [{}])[0].get("text", "")
                        if not generated_text:
                            generated_text = result.get("results", [{}])[0].get("text", "")
                    
                    return generated_text.strip() if generated_text.strip() else "I understand you, but I don't have a specific response."
                    
            except Exception as e:
                print(f"Endpoint {endpoint} failed: {e}")
                continue
        
        return "I'm sorry, I cannot connect to the text generation service. Please check that text-generation-webui is running with API enabled."
    
    def text_to_speech(self, text: str, output_path: str = "temp_tts_output.wav"):
        """Convert text to speech."""
        print("Converting text to speech...")
        
        if self.tts:
            # Use Coqui TTS
            try:
                self.tts.tts_to_file(text=text, file_path=output_path)
            except Exception as e:
                print(f"Coqui TTS failed: {e}, falling back to espeak")
                self._espeak_fallback(text, output_path)
        else:
            # Use espeak fallback
            self._espeak_fallback(text, output_path)
        
        return output_path
    
    def _espeak_fallback(self, text: str, output_path: str):
        """Fallback TTS using espeak-ng."""
        import subprocess
        try:
            subprocess.run([
                "espeak-ng", "-s", "150", "-w", output_path, text
            ], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Espeak failed: {e}")
            raise
    
    def play_audio(self, audio_path: str):
        """Play audio file."""
        print("Playing audio...")
        
        try:
            # Read audio file
            data, sample_rate = sf.read(audio_path)
            
            # Play audio
            sd.play(data, sample_rate)
            sd.wait()  # Wait until audio finishes playing
            
        except Exception as e:
            print(f"Error playing audio: {e}")
    
    def run_conversation_loop(self):
        """Main conversation loop."""
        print("\n" + "="*50)
        print("Conversational AI System Started!")
        print("Press Ctrl+C to exit")
        print("="*50 + "\n")
        
        try:
            while True:
                print("\nListening... (speak now)")
                
                # Record user input
                audio_path = self.record_audio()
                
                # Convert speech to text
                user_text = self.speech_to_text(audio_path)
                if not user_text:
                    print("No speech detected, try again.")
                    continue
                
                print(f"You said: {user_text}")
                
                # Generate response
                ai_response = self.generate_response(user_text)
                print(f"AI: {ai_response}")
                
                # Convert response to speech
                tts_path = self.text_to_speech(ai_response)
                
                # Play the response
                self.play_audio(tts_path)
                
                # Clean up temporary files
                self.cleanup_temp_files()
                
        except KeyboardInterrupt:
            print("\n\nShutting down...")
        finally:
            self.cleanup()
    
    def cleanup_temp_files(self):
        """Clean up temporary audio files."""
        temp_files = ["temp_recording.wav", "temp_tts_output.wav"]
        for file in temp_files:
            if os.path.exists(file):
                try:
                    os.remove(file)
                except:
                    pass
    
    def cleanup(self):
        """Cleanup resources."""
        if hasattr(self, 'audio'):
            self.audio.terminate()
        self.cleanup_temp_files()
        print("Cleanup complete!")


def main():
    """Main function to run the conversational AI."""
    try:
        # Initialize the conversational AI system
        ai = ConversationalAI()
        
        # Run the conversation loop
        ai.run_conversation_loop()
        
    except Exception as e:
        print(f"Error initializing system: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()