import os
import torchaudio
import json

def create_directory_if_not_exists(directory):
    if not os.path.existsg(directory):
        os.makedirs(directory)

def get_wav_files(directory):
    if not os.path.exists(directory):
        print(f"The directory {os.path.abspath(directory)} does not exist.")
        return []
    try:
        return [file for file in os.listdir(directory) if file.endswith('.wav')]
    except PermissionError:
        print(f"You do not have the necessary permissions to access the directory {os.path.abspath(directory)}.")
        return []

def load_and_preprocess_waveform(wav_path):
    waveform, sample_rate = torchaudio.load(wav_path)
    resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)
    waveform = waveform.mean(dim=0)
    return waveform

def create_spectrogram(waveform):
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate=16000)(waveform)
    db_spectrogram = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    return db_spectrogram.squeeze().numpy().tolist()

def save_as_json(data, json_path):
    with open(json_path, 'w') as json_file:
        json.dump(data, json_file)

def preprocess_audio(wav_dir, json_dir):
    create_directory_if_not_exists(json_dir)
    wav_files = get_wav_files(wav_dir)

    for filename in wav_files:
        wav_path = os.path.join(wav_dir, filename)
        json_path = os.path.join(json_dir, filename.replace('.wav', '.json'))

        waveform = load_and_preprocess_waveform(wav_path)
        db_spectrogram_list = create_spectrogram(waveform)
        save_as_json(db_spectrogram_list, json_path)

wav_directory = "FOR_EVAL/FOR_EVAL/TEST_FILES/BENGALI_TEST/1.wav"
json_directory = "bengali_Datamodel"

preprocess_audio(wav_directory, json_directory)


####################### run it 1 st