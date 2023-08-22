import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle
from tensorflow.keras.models import load_model
import numpy as np
from pydub import AudioSegment
import tempfile
import soundfile as sf
import librosa
import os


# Step 1: Load and Preprocess Data

lyrics_folder = "Lyrics"
audio_folder = "Audio"

lyrics = []
audio_features = []

# Load lyrics from text files
for filename in os.listdir(lyrics_folder):
    if filename.endswith(".txt"):
        with open(os.path.join(lyrics_folder, filename), "r", encoding="utf-8") as file:
            lyrics.append(file.read())

# Preprocess lyrics
tokenizer = Tokenizer()
tokenizer.fit_on_texts(lyrics)
total_words = len(tokenizer.word_index) + 1

input_sequences = []
for line in lyrics:
    token_list = tokenizer.texts_to_sequences([line])[0]
    for i in range(1, len(token_list)):
        n_gram_sequence = token_list[:i+1]
        input_sequences.append(n_gram_sequence)

max_sequence_len = max([len(seq) for seq in input_sequences])
input_sequences = pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre')

# Process audio files
def preprocess_audio(audio_path):
    y, sr = librosa.load(audio_path)
    mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    return mel_spectrogram

for filename in os.listdir(audio_folder):
    if filename.endswith(".mp3"):
        audio_path = os.path.join(audio_folder, filename)
        audio_feature = preprocess_audio(audio_path)
        audio_features.append(audio_feature)

# Save processed audio features
if not os.path.exists("processed_audio"):
    os.makedirs("processed_audio")

for i, audio_feature in enumerate(audio_features):
    output_path = os.path.join("processed_audio", f"audio_{i}.npy")
    np.save(output_path, audio_feature)

# Save tokenizer for lyrics
with open("lyrics_tokenizer.pkl", "wb") as tokenizer_file:
    pickle.dump(tokenizer, tokenizer_file)

# Save input sequences for lyrics generation
np.save("input_sequences.npy", input_sequences)

print("Step 1: Data loading and preprocessing completed.")

# Load preprocessed data
input_sequences = np.load("input_sequences.npy")
tokenizer = None

# Load the saved tokenizer
with open("lyrics_tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Tokenize lyrics
total_words = len(tokenizer.word_index) + 1
X, y = input_sequences[:, :-1], input_sequences[:, -1]
y = tf.keras.utils.to_categorical(y, num_classes=total_words)

# Define the RNN model
model = Sequential()
model.add(Embedding(total_words, 64, input_length=input_sequences.shape[1]-1))
model.add(LSTM(100))
model.add(Dense(total_words, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X, y, epochs=100, verbose=1)

# Save the trained model
model.save("lyrics_generation_model.h5")

print("Step 2: Model building and training completed.")

# Load the saved tokenizer
with open("lyrics_tokenizer.pkl", "rb") as tokenizer_file:
    tokenizer = pickle.load(tokenizer_file)

# Load the trained model
model = load_model("lyrics_generation_model.h5")

# Generate Lyrics

seed_text = "I'm feeling"
next_words = 50

generated_lyrics = []

for _ in range(next_words):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=input_sequences.shape[1]-1, padding='pre')
    predicted = np.argmax(model.predict(token_list), axis=-1)
    output_word = ""
    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    seed_text += " " + output_word
    generated_lyrics.append(output_word)

generated_lyrics = " ".join(generated_lyrics)

print("Generated Lyrics:")
print(generated_lyrics)


# Step 4: Generate Audio and Combine
def overlay_lyrics_on_audio(lyrics, audio_features):
    combined_audio = np.zeros_like(audio_features[0])

    # Simulate synchronization and alignment of lyrics with audio
    for i, audio_segment in enumerate(audio_features):
        start_index = i * len(audio_segment)
        end_index = (i + 1) * len(audio_segment)

        # Apply lyrics to the audio segment
        combined_audio[start_index:end_index] += audio_segment

    return combined_audio

# Overlay generated lyrics on generated audio features
generated_audio_features = [np.load("processed_audio/audio_0.npy")]  # Load your generated audio features
combined_audio = overlay_lyrics_on_audio(generated_lyrics, generated_audio_features)
print(combined_audio)

# Export the combined audio as an audio file
output_audio_path = "output_cover_song.wav"  # Adjust the output path
sf.write(output_audio_path, combined_audio, samplerate=44100)  # Adjust the samplerate if needed

print("Step 4: Audio generation and combination completed.")

# Step 5: Final Audio Output

# Convert the combined audio to an AudioSegment object
combined_audio = AudioSegment.from_numpy_array(combined_audio)

# Export the combined audio as a temporary WAV file
temp_wav_path = os.path.join(tempfile.gettempdir(), "temp_combined_audio.wav")
combined_audio.export(temp_wav_path, format="wav")

# Load the temporary WAV file as a SoundFile and resample if needed
target_sample_rate = 44100  # Adjust as needed
audio_data, _ = librosa.load(temp_wav_path, sr=target_sample_rate)

# Save the final audio output as a WAV file
final_audio_output_path = "final_cover_song.wav"  # Adjust the output path
sf.write(final_audio_output_path, audio_data, samplerate=target_sample_rate)

# Clean up the temporary WAV file
os.remove(temp_wav_path)

print("Step 5: Final audio output generated and saved.")