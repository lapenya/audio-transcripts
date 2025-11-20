#!/usr/bin/env python3
"""
Recursively transcribe audio files in a directory using Deepgram API.
Each transcription is saved as a .txt file with the same name as the audio file.
"""

import os
import sys
import argparse
from pathlib import Path
from deepgram import DeepgramClient
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Common audio file extensions
AUDIO_EXTENSIONS = {'.mp3', '.wav', '.m4a', '.flac', '.ogg', '.opus', '.aac', '.wma', '.mp4', '.webm', '.mpeg', '.mpga'}


def is_audio_file(filepath):
    """Check if a file is an audio file based on its extension."""
    return Path(filepath).suffix.lower() in AUDIO_EXTENSIONS


def transcribe_audio(client, audio_path, model="nova-3", language="es"):
    """Transcribe an audio file using Deepgram API."""
    try:
        print(f"Transcribing: {audio_path}")
        with open(audio_path, 'rb') as audio_file:
            response = client.listen.v1.media.transcribe_file(
                request=audio_file.read(),
                model=model,
                language=language
            )
            transcript = response.results.channels[0].alternatives[0].transcript
        return transcript
    except Exception as e:
        print(f"Error transcribing {audio_path}: {str(e)}", file=sys.stderr)
        return None


def save_transcription(audio_path, transcription):
    """Save transcription to a .txt file with the same name as the audio file."""
    audio_path_obj = Path(audio_path)
    txt_path = audio_path_obj.with_suffix('.txt')
    
    try:
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(transcription)
        print(f"Saved transcription: {txt_path}")
        return True
    except Exception as e:
        print(f"Error saving transcription to {txt_path}: {str(e)}", file=sys.stderr)
        return False


def process_directory(directory, client, model="nova-3", language="es", skip_existing=True):
    """Recursively process all audio files in a directory."""
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist.", file=sys.stderr)
        return
    
    if not directory_path.is_dir():
        print(f"Error: '{directory}' is not a directory.", file=sys.stderr)
        return
    
    audio_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = Path(root) / file
            if is_audio_file(file_path):
                audio_files.append(file_path)
    
    if not audio_files:
        print(f"No audio files found in '{directory}'")
        return
    
    print(f"Found {len(audio_files)} audio file(s) to process.\n")
    
    processed = 0
    skipped = 0
    failed = 0
    
    for audio_file in audio_files:
        txt_file = audio_file.with_suffix('.txt')
        
        # Skip if transcription already exists
        if skip_existing and txt_file.exists():
            print(f"Skipping (transcription exists): {audio_file}")
            skipped += 1
            continue
        
        transcription = transcribe_audio(client, audio_file, model, language)
        
        if transcription:
            if save_transcription(audio_file, transcription):
                processed += 1
            else:
                failed += 1
        else:
            failed += 1
        
        print()  # Empty line for readability
    
    print(f"\nSummary:")
    print(f"  Processed: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description='Recursively transcribe audio files using Deepgram API'
    )
    parser.add_argument(
        'directory',
        help='Directory to process (will recursively visit all subdirectories)'
    )
    parser.add_argument(
        '--api-key',
        help='Deepgram API key (or set DEEPGRAM_API_KEY environment variable)',
        default=None
    )
    parser.add_argument(
        '--model',
        help='Deepgram model to use (default: nova-3)',
        default='nova-3'
    )
    parser.add_argument(
        '--language',
        help='Language code for transcription (default: es for Spanish). Use "auto" for automatic detection.',
        default='es'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-transcribe files even if .txt file already exists'
    )
    
    args = parser.parse_args()
    
    # Get API key from argument, environment variable, or .env file
    api_key = args.api_key or os.getenv('DEEPGRAM_API_KEY')
    
    if not api_key:
        print("Error: Deepgram API key is required.", file=sys.stderr)
        print("Set it via one of the following methods:", file=sys.stderr)
        print("  1. --api-key argument", file=sys.stderr)
        print("  2. DEEPGRAM_API_KEY environment variable", file=sys.stderr)
        print("  3. DEEPGRAM_API_KEY in .env file", file=sys.stderr)
        sys.exit(1)
    
    # Initialize Deepgram client with API key
    client = DeepgramClient(api_key=api_key)
    
    # Process directory
    process_directory(
        args.directory, 
        client, 
        model=args.model, 
        language=args.language,
        skip_existing=not args.no_skip_existing
    )


if __name__ == '__main__':
    main()

