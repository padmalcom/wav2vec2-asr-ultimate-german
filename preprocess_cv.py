import os
import csv
import string
from tqdm import tqdm
from pydub import AudioSegment
from transformers import AutoModelForSequenceClassification, TFAutoModelForSequenceClassification, AutoTokenizer, FSMTForConditionalGeneration, FSMTTokenizer, pipeline
import numpy as np
from scipy.special import softmax
import urllib.request

RAW_DATA_FILE = os.path.join('common-voice-12','validated.tsv')
TRAIN_FILE = os.path.join("common-voice-12", "train.csv")
TEST_FILE = os.path.join("common-voice-12", "test.csv")
TEST_TRAIN_RATIO = 8 # every 8th sample goes into test

## dialects
dialect_map = {
	'Amerikanisches Deutsch': 'Amerikanisch',
	'Bayerisch': 'Bayerisch',
	'Britisches Deutsch': 'Britisch',
	'Deutschland Deutsch': 'Deutsch',
	'Deutschland Deutsch,Alemanischer Akzent,Süddeutscher Akzent': 'Süddeutsch',
	'Deutschland Deutsch,Berliner Deutsch': 'Berlinerisch',
	'Deutschland Deutsch,Hochdeutsch': 'Deutsch',
	'Deutschland Deutsch,Ruhrgebiet Deutsch,West Deutsch': 'Rheinländisch',
	'Deutschland Deutsch,Süddeutsch': 'Süddeutsch',
	'Niederbayerisch': 'Bayrisch',
	'Niedersächsisches Deutsch,Deutschland Deutsch': 'Deutsch',
	'Nordrhein-Westfalen,Bundesdeutsch, Hochdeutsch,Deutschland Deutsch': 'Deutsch',
	'Ostbelgien,Belgien,Belgisches Deutsch': 'Belgisch',
	'Schweizerdeutsch': 'Schweizerdeutsch',
	'Süddeutsch': 'Süddeutsch',
	'Österreichisches Deutsch': 'Österreichisch',
	'': 'Deutsch'
}
	

### emotion
EMOTION_MODEL_NAME = "padmalcom/wav2vec2-large-emotion-detection-german"
emotions = {'anger':0, 'boredom':1, 'disgust':2, 'fear':3, 'happiness':4, 'sadness':5, 'neutral':6}
audio_classifier = pipeline(task="audio-classification", model=EMOTION_MODEL_NAME)

def emotion(audio_file):
	preds = audio_classifier(audio_file)
	max_score = 0
	max_label = 6
	max_label_text = 'neutral'
	for p in preds:
		if p['score'] > max_score and p['score'] > 0.25:
			max_score = p['score']
			max_label = emotions[p["label"]]
			max_label_text = p["label"]
			print("There is an emotional file:", max_label_text)
	return max_label_text
	
### translation
TRANSLATION_MODEL_NAME = "facebook/wmt19-de-en"
translation_tokenizer = FSMTTokenizer.from_pretrained(TRANSLATION_MODEL_NAME)
translation_model = FSMTForConditionalGeneration.from_pretrained(TRANSLATION_MODEL_NAME)

def translate(text):
	input_ids = translation_tokenizer.encode(text, return_tensors="pt")
	outputs = translation_model.generate(input_ids)
	return translation_tokenizer.decode(outputs[0], skip_special_tokens=True)

### preparation
def prepare_data():
	labels = {}
	with open(RAW_DATA_FILE) as f:
		row_count = sum(1 for line in f)
		print("There are", row_count, "rows in the dataset.")
	
	with open(RAW_DATA_FILE, 'r', encoding="utf8") as f:
		tsv = csv.DictReader(f, delimiter="\t")
		
		if not os.path.exists(os.path.join('common-voice-12', "wavs")):
			os.mkdir(os.path.join('common-voice-12', "wavs"))
			
		i = 0
		faulty_lines = 0
		train_file_header_written = False
		test_file_header_written = False
		test_count = 0
		train_count = 0
		with open(TRAIN_FILE, 'w', newline='', encoding="utf8") as train_f:
			with open(TEST_FILE, 'w', newline='', encoding="utf8") as test_f:
				try:
					for line in tqdm(tsv, total=row_count):
						formatted_sample = {}
						formatted_sample['speaker'] = line['client_id']
						formatted_sample['file'] = line['path']
						formatted_sample['sentence'] = line['sentence'].translate(str.maketrans('', '', string.punctuation))
						formatted_sample['age'] = line['age']
						formatted_sample['gender'] = line['gender']
						formatted_sample['language'] = line['locale']
						formatted_sample['accent'] = dialect_map[line['accents'].strip()]
						formatted_sample['speaker'] = line['client_id']
						
						if (formatted_sample['sentence'] == None or formatted_sample['sentence'] == 'nan' or line['client_id'] == None or line['path'] == None or line['sentence'] == None or line['age'] == None or line['gender'] == None or line['locale'] == None or line['client_id'].strip() == '' or line['path'].strip() == '' or line['sentence'].strip() == '' or line['age'].strip() == '' or line['gender'].strip() == '' or line['locale'].strip() == '' or formatted_sample['sentence'].strip() == ''):
							#print("Faulty line: ", line)
							faulty_lines += 1
							continue
						
						# english text
						if not line['locale'] == 'en':
							formatted_sample['eng_sentence'] = translate(formatted_sample['sentence'])
						else:
							formatted_sample['eng_sentence'] = formatted_sample['sentence']
						
						mp3FullPath = os.path.join('common-voice-12', "clips", line['path'])
						filename, _ = os.path.splitext(os.path.basename(mp3FullPath))
						sound = AudioSegment.from_mp3(mp3FullPath)
						if sound.duration_seconds > 0:
								sound = sound.set_frame_rate(16000)
								sound = sound.set_channels(1)
								wav_path = os.path.join('common-voice-12', "wavs", filename + ".wav")
								sound.export(wav_path, format="wav")
								formatted_sample['file'] = filename + ".wav"
								
								# emotion classification
								formatted_sample['emotion'] = emotion(wav_path)

								if i % TEST_TRAIN_RATIO == 0:
									if not test_file_header_written:
										test_w = csv.DictWriter(test_f, formatted_sample.keys())
										test_w.writeheader()
										test_file_header_written = True
									test_w.writerow(formatted_sample)
									test_count += 1
								else:
									if not train_file_header_written:
										train_w = csv.DictWriter(train_f, formatted_sample.keys())
										train_w.writeheader()
										train_file_header_written = True
									train_w.writerow(formatted_sample)
									train_count += 1
								i += 1
				except KeyboardInterrupt:
					print("Keyboard interrupt called. Exiting...")
				
				#random.shuffle(data)
				print("Found", i, "samples.", train_count, "in train and", test_count, "in test.", faulty_lines, "lines were faulty.")
		
if __name__ == '__main__':
	prepare_data()