"""
This script will extract audio from a YouTube/TikTok/Instagram video, transcribe the audio, and
generate an Obsidian markdown file with the transcription and a summary of the transcription.
"""
import argparse
import json
import subprocess
import whisper

from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from pathlib import Path

template_dir = Path(__file__).parent / 'templates'
env = Environment(loader=FileSystemLoader(template_dir), autoescape=True)

def run_command(command):
    try:
        # Run the command and capture the output
        result = subprocess.run(command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True)

        # Print the output
        print("Command Output:", result.stdout)
        return result.stdout

    except subprocess.CalledProcessError as e:
        # Print the error if the command fails
        print("Command Failed:", e.stderr)
        return None


def extract_audio(url):
    command = f'yt-dlp -x {url} -o "audio.%(ext)s"'
    run_command(command)


def transcribe_audio(audio_path):
    whisper_model = whisper.load_model("base")
    result = whisper_model.transcribe(audio_path)
    return result['text']


def generate_llm_response(transcription_text):
    # Example using OpenLlama; adjust as per your setup
    llm = ChatOllama(
        model="llama3.1",
        temperature=0,
    )
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are an advanced AI with a 2,128 IQ and you are an expert in understanding "
                "any input and extracting the most important ideas from it.  You will look at the input text and "
                "summarize it in a way that is easy to understand for a human.  You will provide hashtags that "
                "can be used to categorize the text.  You will also provide a title for the text.  You will return "
                "these in well structured JSON format with the keys 'title', 'summary', and 'hashtags'.  Make sure the "
                "summary includes all of the import information from the input text.  Only provide the JSON object, no "
                "extra information.",
            ),
            ("human", "{input}"),
        ]
    )
    chain = prompt | llm
    response = chain.invoke(
        {
            "input": transcription_text,
        }
    )
    return response.content


def remove_llm_json_formatting(llm_text):
    return json.loads(llm_text.lstrip("```json").rstrip("```"))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Transcribe audio and generate markdown files.")
    parser.add_argument("url", help="URL to include in the Obsidian markdown file.")

    args = parser.parse_args()
    print("Extracting audio from video...")
    extract_audio(args.url)
    print("Transcribing audio...")
    transcription = transcribe_audio("audio.m4a")
    print("Generating LLM response...")
    llm_response = generate_llm_response(transcription)
    llm_response_dict = remove_llm_json_formatting(llm_response)
    llm_response_dict["url"] = args.url
    llm_response_dict["transcription"] = transcription
    print(json.dumps(llm_response_dict, indent=4))
    template = env.get_template('video_extracted_template.md')
    output = template.render(llm_response_dict)
    output_file = Path(f'{datetime.now().isoformat()}-output.md')
    output_file.write_text(output)
    Path("audio.m4a").unlink()
