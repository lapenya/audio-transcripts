#!/usr/bin/env python3
"""
Recursively convert all .wav files in a directory to .mp3 format.
Each .wav file will be converted to a .mp3 file with the same name in the same location.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path


def check_ffmpeg():
    """Check if ffmpeg is available."""
    try:
        result = subprocess.run(
            ['ffmpeg', '-version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def convert_wav_to_mp3(wav_path, mp3_path, bitrate='192k', logger=None):
    """Convert a WAV file to MP3 format using ffmpeg."""
    try:
        if logger:
            logger.info(f"Converting: {wav_path} -> {mp3_path}")
        
        # Use ffmpeg to convert WAV to MP3
        cmd = [
            'ffmpeg',
            '-i', str(wav_path),
            '-codec:a', 'libmp3lame',
            '-b:a', bitrate,
            '-y',  # Overwrite output file if it exists
            str(mp3_path)
        ]
        
        # Run ffmpeg (suppress output unless there's an error)
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            error_msg = f"ffmpeg error: {result.stderr}"
            if logger:
                logger.error(error_msg)
            print(f"Error: {error_msg}", file=sys.stderr)
            return False
        
        return True
    except Exception as e:
        error_msg = f"Error converting {wav_path}: {str(e)}"
        if logger:
            logger.error(error_msg, exc_info=True)
        print(error_msg, file=sys.stderr)
        return False


def process_directory(directory, bitrate='192k', skip_existing=True, logger=None):
    """Recursively process all .wav files in a directory."""
    directory_path = Path(directory)
    
    if not directory_path.exists():
        print(f"Error: Directory '{directory}' does not exist.", file=sys.stderr)
        return
    
    if not directory_path.is_dir():
        print(f"Error: '{directory}' is not a directory.", file=sys.stderr)
        return
    
    # Find all .wav files
    wav_files = []
    for root, dirs, files in os.walk(directory_path):
        for file in files:
            file_path = Path(root) / file
            if file_path.suffix.lower() == '.wav':
                wav_files.append(file_path)
    
    if not wav_files:
        print(f"No .wav files found in '{directory}'")
        return
    
    print(f"Found {len(wav_files)} .wav file(s) to convert.\n")
    
    processed = 0
    skipped = 0
    failed = 0
    
    for wav_file in wav_files:
        mp3_file = wav_file.with_suffix('.mp3')
        
        # Skip if MP3 file already exists
        if skip_existing and mp3_file.exists():
            print(f"Skipping (MP3 file exists): {wav_file}")
            skipped += 1
            continue
        
        print(f"Converting: {wav_file}")
        
        # Convert WAV to MP3
        if convert_wav_to_mp3(wav_file, mp3_file, bitrate, logger):
            processed += 1
            print(f"  ✓ Created: {mp3_file}")
        else:
            failed += 1
        
        print()  # Empty line for readability
    
    print(f"\nSummary:")
    print(f"  Converted: {processed}")
    print(f"  Skipped: {skipped}")
    print(f"  Failed: {failed}")


def main():
    parser = argparse.ArgumentParser(
        description='Recursively convert all .wav files in a directory to .mp3 format'
    )
    parser.add_argument(
        'directory',
        help='Directory to process (will recursively visit all subdirectories)'
    )
    parser.add_argument(
        '--bitrate',
        default='192k',
        help='MP3 bitrate (default: 192k). Examples: 128k, 192k, 256k, 320k'
    )
    parser.add_argument(
        '--no-skip-existing',
        action='store_true',
        help='Re-convert files even if .mp3 file already exists'
    )
    
    args = parser.parse_args()
    
    # Check if ffmpeg is available
    if not check_ffmpeg():
        print("Error: ffmpeg is not installed or not found in PATH.", file=sys.stderr)
        print("\nInstall ffmpeg:", file=sys.stderr)
        print("  macOS: brew install ffmpeg", file=sys.stderr)
        print("  Linux: sudo apt-get install ffmpeg", file=sys.stderr)
        print("  Windows: Download from https://ffmpeg.org/download.html", file=sys.stderr)
        print("\nAfter installing, make sure ffmpeg is in your PATH.", file=sys.stderr)
        sys.exit(1)
    
    # Process directory
    process_directory(
        args.directory,
        bitrate=args.bitrate,
        skip_existing=not args.no_skip_existing
    )


if __name__ == '__main__':
    main()

