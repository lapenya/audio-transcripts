# Audio Transcription Script

A Python script that recursively processes audio files in a directory and transcribes them using Deepgram's automated speech recognition API.

## Features

- Recursively visits all subdirectories
- Supports multiple audio formats (MP3, WAV, M4A, FLAC, OGG, etc.)
- Saves transcriptions as `.txt` files with the same name as the original audio file
- Skips files that already have transcriptions (optional)
- Provides progress feedback and summary statistics

## Installation

1. Install Poetry if you haven't already:
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

2. Install the project dependencies:
```bash
poetry install
```

## Configuration

Set your Deepgram API key using one of the following methods (in order of precedence):

1. **Create a `.env` file** (recommended for local development):
   ```bash
   echo "DEEPGRAM_API_KEY=your-api-key-here" > .env
   ```
   The `.env` file is automatically loaded and is already included in `.gitignore`.

2. **Set as environment variable**:
   ```bash
   export DEEPGRAM_API_KEY='your-api-key-here'
   ```

3. **Pass as command-line argument**:
   ```bash
   poetry run python transcribe_audio.py /path/to/audio/directory --api-key your-api-key-here
   ```

## Usage

Run the script with Poetry:
```bash
poetry run python transcribe_audio.py /path/to/audio/directory
```

Or activate the Poetry shell first:
```bash
poetry shell
python transcribe_audio.py /path/to/audio/directory
```

### Options

- `--no-skip-existing`: Re-transcribe files even if a `.txt` file already exists
- `--api-key`: Provide Deepgram API key as command-line argument (overrides .env file and environment variable)
- `--model`: Specify Deepgram model to use (default: `nova-3`). Other options include `nova-2`, `nova`, `base`, `enhanced`, etc.

## Supported Audio Formats

- MP3 (.mp3)
- WAV (.wav)
- M4A (.m4a)
- FLAC (.flac)
- OGG (.ogg)
- OPUS (.opus)
- AAC (.aac)
- WMA (.wma)
- MP4 (.mp4)
- WebM (.webm)
- MPEG (.mpeg, .mpga)

## Example

```bash
poetry run python transcribe_audio.py ./my-audio-files
```

This will:
1. Find all audio files in `./my-audio-files` and its subdirectories
2. Transcribe each file using Deepgram's speech recognition API
3. Save transcriptions as `.txt` files next to the original audio files

## Notes

- The script uses Deepgram's API, which requires an API key and may incur costs
- Get your API key from [Deepgram's dashboard](https://console.deepgram.com/)
- Large audio files may take some time to process
- The script will skip files that already have corresponding `.txt` files unless `--no-skip-existing` is used
- Default model is `nova-3` (Deepgram's latest general-purpose model). You can specify a different model with `--model`

