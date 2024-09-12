import cv2
import whisper
import subprocess
import os
import numpy as np
from tqdm import tqdm
import textwrap
import streamlit as st
import tempfile
import shutil
from PIL import Image, ImageDraw, ImageFont, ImageFilter, ImageColor  # Added ImageColor
import cv2
import spacy
# Set page configuration with sidebar collapsed by default
st.set_page_config(
    page_title="Video Caption AI",
    page_icon="🎥",
    layout="centered",
    initial_sidebar_state="collapsed"  # Sidebar starts collapsed
)

def get_text_y_position(position, text_height, height):
    if position == "top":
        return text_height + 50
    elif position == "center":
        return (height - text_height) // 2
    elif position == "bottom":
        return height - text_height - 50
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2

def split_text_into_chunks(sentence, chunk_size=4):
    """Splits a sentence into chunks of a given number of words."""
    words = sentence.split()
    chunks = [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]
    return chunks

def get_active_word_index(chunk_words, current_time, word_timestamps):
    """Determine which word in the chunk is active based on current time."""
    for i, (word, start_time, end_time) in enumerate(word_timestamps):
        if start_time <= current_time <= end_time:
            return i
    return -1  # No active word if time doesn't match any


# Load spaCy English model
nlp = spacy.load("en_core_web_sm")
def add_captions(frame, sentences, current_time, width, height, font, text_color, highlight_color, position, catchy_word_color="#FFFF00", border_color="#000000", border_thickness=2, background_blur_radius=10, fade_duration=1.0):
    visible_text = ''  # This will hold the text to display
    word_timestamps = []  # To store word timing for highlighting
    highlighted_word_index = -1
    fade_in_opacity = 0  # Start with full transparency

    # Find the active sentence based on current time
    for sentence_tuple in sentences:
        if len(sentence_tuple) == 3:
            sentence, start_time, end_time = sentence_tuple
            timestamps = []
        elif len(sentence_tuple) == 4:
            sentence, start_time, end_time, timestamps = sentence_tuple
        else:
            continue

        if start_time <= current_time <= end_time:
            chunks = split_text_into_chunks(sentence)
            total_chunks = len(chunks)
            chunk_duration = (end_time - start_time) / total_chunks
            chunk_index = int((current_time - start_time) // chunk_duration)

            if chunk_index < total_chunks:
                visible_text = chunks[chunk_index]

                # Get the actual time range for the current chunk
                chunk_start_time = start_time + chunk_index * chunk_duration
                chunk_end_time = chunk_start_time + chunk_duration

                # Calculate fade in based on time
                time_since_chunk_start = current_time - chunk_start_time
                if time_since_chunk_start < fade_duration:
                    fade_in_opacity = int((time_since_chunk_start / fade_duration) * 255)
                else:
                    fade_in_opacity = 255

                # Adjust the word timestamps to the current chunk's time range
                word_timestamps = [
                    (word, ts[0] - chunk_start_time, ts[1] - chunk_start_time) for word, ts in zip(sentence.split(), timestamps)
                    if chunk_start_time <= ts[0] <= chunk_end_time
                ]

                highlighted_word_index = get_active_word_index(
                    visible_text.split(),
                    current_time - chunk_start_time,
                    word_timestamps
                )

                break

    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("RGBA")
    draw = ImageDraw.Draw(pil_image)

    text_bbox = draw.textbbox((0, 0), visible_text, font=font)
    text_width, text_height = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
    x_cursor, y_cursor = (width - text_width) // 2, get_text_y_position(position, text_height, height)

    # Create a smaller background image for the text
    background_image = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    background_draw = ImageDraw.Draw(background_image)

    # Adjust padding to make the background smaller around the text
    padding = 5  # Reduced padding
    background_bbox = (x_cursor - padding, y_cursor - padding, x_cursor + text_width + padding, y_cursor + text_height + padding)
    background_draw.rectangle(background_bbox, fill="black")

    # Apply blur to the background
    blurred_background = background_image.filter(ImageFilter.GaussianBlur(background_blur_radius))

    # Overlay the blurred background on the original image
    pil_image = Image.alpha_composite(pil_image.convert("RGBA"), blurred_background)

    # Analyze the visible text with spaCy to identify catchy words
    doc = nlp(visible_text)
    catchy_words = {token.text for token in doc if token.pos_ in {"PROPN"}}

    # Draw text with fade effect
    draw = ImageDraw.Draw(pil_image)
    words = visible_text.split()
    word_positions = []

    # Define a larger font size for catchy words
    catchy_font_size = font.size + 10  # Increase font size by 10
    catchy_font = ImageFont.truetype(font.path, catchy_font_size)

    for i, word in enumerate(words):
        # Determine the font and color for the word
        if word in catchy_words:
            word_font = catchy_font
            word_color = catchy_word_color
        else:
            word_font = font
            word_color = highlight_color if i == highlighted_word_index else text_color

        word_bbox = draw.textbbox((x_cursor, y_cursor), word + " ", font=word_font)
        word_width = word_bbox[2] - word_bbox[0]

        # Draw border with fade effect
        border_color_with_alpha = (*ImageColor.getrgb(border_color), fade_in_opacity)
        for dx in range(-border_thickness, border_thickness + 1):
            for dy in range(-border_thickness, border_thickness + 1):
                if dx != 0 or dy != 0:
                    draw.text((x_cursor + dx, y_cursor + dy), word, font=word_font, fill=border_color_with_alpha)

        # Draw word with fade effect
        text_color_with_alpha = (*ImageColor.getrgb(word_color), fade_in_opacity)
        draw.text((x_cursor, y_cursor), word, font=word_font, fill=text_color_with_alpha)

        # Update cursor position for next word
        x_cursor += word_width

    # Convert image back to OpenCV format
    frame = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
    return frame




def process_video(audio_path, video_path, output_dir, font, text_color, highlight_color, position, background_blur_radius=10):
    temp_video_path = os.path.join(output_dir, "captioned_video.mp4")
    final_output_path = os.path.join(output_dir, "final_output.mp4")

    # Process video with captions
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Error: Unable to open video file.")
        return None

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    # Extract segments
    segments_with_timestamps = [(segment['text'], segment['start'], segment['end']) for segment in whisper_model.transcribe(audio_path)['segments']]
    frame_number = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    progress_text = st.empty()

    for frame_number in tqdm(range(total_frames), desc="Processing frames"):
        ret, frame = cap.read()
        if not ret:
            break

        current_time = frame_number / fps
        frame = add_captions(frame, segments_with_timestamps, current_time, width, height, font, text_color, highlight_color, position, border_color="#000000", border_thickness=2, background_blur_radius=background_blur_radius)

        out.write(frame)

        progress_bar.progress(frame_number / total_frames)
        progress_text.text(f"Processing frame {frame_number}/{total_frames}")

    cap.release()
    out.release()

    # Merge audio with captioned video
    subprocess.run([
        'ffmpeg', '-i', temp_video_path, '-i', audio_path, 
        '-c:v', 'libx264', '-crf', '23', '-preset', 'slow',
        '-c:a', 'aac', '-b:a', '128k', '-movflags', '+faststart',
        '-shortest', final_output_path
    ], check=True)

    # Clean up
    os.remove(temp_video_path)
    return final_output_path

# Streamlit app layout
st.title("AI Video Captioning🎥")
position = st.selectbox("Choose text position:", ["top", "center", "bottom"])
text_color = st.color_picker("Choose text color:", "#FFFFFF")
highlight_color_rgb = "#ffFFff" 


fonts_dir = "fonts"
if not os.path.exists(fonts_dir):
    st.error(f"Fonts directory '{fonts_dir}' does not exist.")
else:
    font_files = [f for f in os.listdir(fonts_dir) if f.endswith('.ttf')]
    if not font_files:
        st.error("No .ttf font files found in the fonts directory.")
    else:
        font_style = st.selectbox("Choose font style:", [f[:-4] for f in font_files])
        font_size_option = st.selectbox("Choose font size:", ["Small", "Medium", "Large"])
        font_size_map = {
            "Small": 30,
            "Medium": 35,
            "Large": 40
        }
        font_size = font_size_map[font_size_option]
        sample_text = "Selected Font"
        font_path = os.path.join(fonts_dir, f"{font_style}.ttf")
        font = ImageFont.truetype(font_path, font_size)
        preview_image = Image.new("RGB", (800, 100), "black")
        draw = ImageDraw.Draw(preview_image)
        text_color_rgb = tuple(int(text_color[i:i + 2], 16) for i in (1, 3, 5))
        draw.text((10, 10), sample_text, font=font, fill=text_color_rgb)
        st.image(preview_image, caption=f"Font: {font_style}")



# Allow user to upload custom video file
uploaded_video = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_video:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video_file:
        temp_video_file.write(uploaded_video.getbuffer())
        video_path = temp_video_file.name
# Audio file uploader
uploaded_audio = st.file_uploader("Upload an audio file", type=["mp3", "wav"])

if uploaded_audio:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio_file:
        temp_audio_file.write(uploaded_audio.getbuffer())
        audio_path = temp_audio_file.name

# Button to start processing
if st.button("Start Processing"):
    if uploaded_audio and uploaded_video:
        st.write("Processing your video...")
        st.info("Transcribing audio...")
        whisper_model = whisper.load_model("base")
        result = whisper_model.transcribe(audio_path)
        if result['language'] != 'en':
            st.error("The detected language is not English. Terminating the process.")
            os.remove(audio_path)
        else:
            output_dir = tempfile.mkdtemp()
            final_output_path = process_video(
                audio_path, 
                video_path, 
                output_dir, 
                font, 
                text_color, 
                highlight_color_rgb, 
                position
            )
            if final_output_path:
                st.success("Video processing complete!")
                with open(final_output_path, "rb") as file:
                    st.download_button(label="Download Processed Video", data=file, file_name="final_output.mp4", mime="video/mp4")
                shutil.rmtree(output_dir)
    else:
        st.error("Please upload an audio file and select a video.")


import json
from hashlib import md5
import socket

# File to store the count and list of users
data_file = 'button_data.json'

# Helper function to get user's IP address (simplified for testing)
def get_user_ip():
    # Getting a unique identifier for the user
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address

# Load the button data from a file
def load_button_data():
    if os.path.exists(data_file):
        with open(data_file, 'r') as f:
            return json.load(f)
    else:
        return {'count': 0, 'users': []}

# Save the button data to a file
def save_button_data(data):
    with open(data_file, 'w') as f:
        json.dump(data, f)

# Get user IP
user_ip = md5(get_user_ip().encode()).hexdigest()

# Load existing data
data = load_button_data()

# Sidebar: Display the total number of presses
st.sidebar.write(f"user count: {data['count']}")

# Main page: Check if the user has already pressed the button
if user_ip not in data['users']:
    if st.button('Press Me!'):
        # Increment the counter and save the user's IP
        data['count'] += 1
        data['users'].append(user_ip)
        save_button_data(data)
        st.success(f"Thank you!")
else:
    st.write(f'Getting better day by day😊!')



st.markdown('<link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/5.15.3/css/all.min.css">', unsafe_allow_html=True)


def footer():

  html_temp = """
  <div style="position: fixed; bottom: 50px; width: 100%; text-align: center; font-weight: bold;">
    <p style="margin-bottom: 5px; font-size: 14px;">
      Copyright &copy; Made By <span style="color: #007bff; font-weight: bold;">AliHamzaSultan</span>
      <a href="https://www.linkedin.com/in/ali-hamza-sultan-1ba7ba267/" target="_blank" style="margin-left: 10px;"><i class="fab fa-linkedin" style="font-size: 20px;"></i></a>
      <a href="https://github.com/alihamzasultan" target="_blank"><i class="fab fa-github" style="font-size: 20px; margin-left: 10px;"></i></a>
    </p>
  </div>
  """
  st.markdown(html_temp, unsafe_allow_html=True)

footer()
