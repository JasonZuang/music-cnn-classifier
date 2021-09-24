import os
import librosa
import math
import json

SAMPLE_RATE = 22040
DURATION = 30
SPT = SAMPLE_RATE * DURATION
DATAPATH = "Data/test_clips/"
JSONPATH = "Data/data_test.json"

def save_mfcc(data_path, json_path, n_mfcc = 13, n_fft = 2048, hop_length = 512, num_seg = 5):

	data = {
			"mapping":[],
			"mfcc":[],
			"labels":[]

			}

	nsps = int(SPT/num_seg)
	expected_mfcc_per_segment = math.ceil(nsps / hop_length)

	for i,(dirpath,dirnames, filenames) in  enumerate(os.walk(data_path)):
		print(filenames,dirnames)
		#gets genre from foldername and adds into mapping
		if dirpath is not data_path:
			dirpath_comp = dirpath.split("/")
			label = dirpath_comp[-1]
			data['mapping'].append(label)
			print("\n Processing", label)

	for f in filenames:
		#load audio file
		print(f)
		if "._" not in f:

			filepath = os.path.join(dirpath,f)
			signal, sr = librosa.load(filepath,sr = SAMPLE_RATE)

			#process segments extracting mfcc and storing data
			for s in range(num_seg):
				start_sample = nsps * s
				finish_sample = start_sample + nsps

				mfcc = librosa.feature.mfcc(signal[start_sample:finish_sample], 
				sr = SAMPLE_RATE,
				n_fft = n_fft,
				n_mfcc = n_mfcc,
				hop_length = hop_length)
				mfcc = mfcc.T


				#store mfcc for segment if it has expected length
				if len(mfcc) == expected_mfcc_per_segment:
					data["mfcc"].append(mfcc.tolist())
					data['labels'].append(i-1)
					print(filepath,"segment: ",s)

	with open(json_path,"w") as fp:
		json.dump(data,fp,indent=4)


if __name__ == "__main__":
	save_mfcc(DATAPATH,JSONPATH)