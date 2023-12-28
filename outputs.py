import wave
import json
# Open the .wav file
with wave.open('TEST_FILES\BENGALI_TEST', 'rb') as wav_file:
    # Do something with the .wav file
    pass

# Open the .json file
with open('bengali_Datamodel\12.json', 'r') as json_file:
    transcription = json.load(json_file)
    print(transcription)  # Print the transcription

#it will conver the wav file into json using ptorch  trained data model it will transfer  in transcriptioon  and show it in the output it want to run in certain formet
