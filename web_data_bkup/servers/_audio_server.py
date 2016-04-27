from bs4 import BeautifulSoup
from pydub import AudioSegment
from IPython import embed
import subprocess
import urllib2
import random
import Pyro4
import re
import os

class AudioServer():

	def __init__(self):
		
		self.rootFolder = 'http://192.168.1.205/audio/audio/'
		self.wavFilesParentDir = '/home/ubuntu/wav/'
		self.mp3FilesParendDir = '/home/ubuntu/mp3/'

		self.iterItems = self.__iter__()
		
	def iter_next(self):
		return self.iterItems.next()


	def get_subDir_list(self):

		resp = urllib2.urlopen(self.rootFolder)
		html = resp.read()
		soup = BeautifulSoup(html, 'html.parser')
		subDirs = [i['href'] for i in soup.select('a') if i['href']!='/audio/']
		
		return sorted(subDirs)

	def mp3_crawler(self, subdir):

		url = self.rootFolder + subdir
		resp = urllib2.urlopen(url)
		html = resp.read()
		soup = BeautifulSoup(html, 'html.parser')

		mp3Files = [i['href'] for i in soup.select('a') if i['href'].endswith(".mp3")]
		
		titles = self.get_titles_dict(url)

		for mp3 in mp3Files:

			title_key = mp3.split(".")[0]
			mp3file = urllib2.urlopen(url+mp3)
			sound = AudioSegment.from_file(mp3file)

			mp3Dir = self.mp3FilesParendDir+subdir
			if not os.path.exists(mp3Dir):
			 		os.makedirs(mp3Dir)

			mp3Title = title_key
			if titles is not None and titles.has_key(title_key):
				mp3Title = titles[title_key]

			if not os.path.exists(mp3Dir+mp3Title):
				sound.export(mp3Dir+mp3Title, format='mp3')

		return mp3Files

	def get_titles_dict(self, url):

		try:
			resp = urllib2.urlopen(url+'available.txt')
		except:
			try:
				resp = urllib2.urlopen(url+'data.txt')
			except:
				return None
		
		regex = re.compile('[^a-zA-Z_+\s]')
		
		titles_dict = {}

		for line in resp.readlines():
			key, val = re.split('\t+', line.rstrip())
			titles_dict[key] = regex.sub('', val.replace (" ", "_")).lower()
		
		return titles_dict

	def convert_mp3_to_wav(self):

		subDirsList = self.get_subDir_list()
		count = 0
		
		for subdir in subDirsList:

			url = self.rootFolder + subdir
			mp3List = self.mp3_crawler(subdir)
			titles = self.get_titles_dict(url)

			for mp3 in mp3List:

				title_key = mp3.split(".")[0]
				outputDir = self.wavFilesParentDir+subdir

				if not os.path.exists(outputDir):
			 		os.makedirs(outputDir)

			 	wavTitle = title_key
				if titles is not None and titles.has_key(title_key):
					wavTitle = titles[title_key]

				file_path = outputDir+wavTitle+".wav"
				if not os.path.exists(file_path):
					# set bit rate to 64 kbps and  sample rate to 8000 Hz
					sound = "ffmpeg -i " + url+mp3 + " -ac 1 -ab 64000 -ar 8000 " + file_path
					subprocess.call(sound, shell=True)
				
				count +=1
		
		print 'wave files count: ', count

	def get_all_wav_files(self, withDir=False):
		
		parentDir = self.wavFilesParentDir
		subDirsList = os.listdir(parentDir)
		
		wavFilesList = []

		for subdir in subDirsList:
			for wavfile in os.listdir(parentDir+subdir):
				wavDir = parentDir + subdir + '/'
				if withDir:
					wavFilesList.append({"wav":wavfile, "dir":wavDir})
				else:
					wavFilesList.append(wavfile)

		return wavFilesList
	
	def get_wav_files_by_keys(self, keys):

		all_wav_files = self.get_all_wav_files(True)
		key = "wav"
		filterdList = filter(lambda x: key in x and x[key] in keys, all_wav_files)

		if type(keys) is list:
			wav_dict = {}
			for obj in filterdList:
				wav_dict[obj["wav"]] = obj["dir"]+obj["wav"]
			return wav_dict
		else:
			return filterdList[0]["dir"]+filterdList[0]["wav"]

	def __iter__(self):

		wavFiles = self.get_all_wav_files(True)

		while True:
			wavfile = random.choice(wavFiles)
			wavPath = wavfile["dir"]+wavfile["wav"]
			yield {"audio": wavPath}


#audioServer = AudioServer()

daemon = Pyro4.Daemon()               	# make a Pyro daemon
ns = Pyro4.locateNS()                  	# find the name server
uri = daemon.register(AudioServer)   	# register the greeting maker as a Pyro object
ns.register("audioserver", uri)   		# register the object with a name in the name server

print("Ready...")
daemon.requestLoop() 

# server = AudioServer()
# x = server.__iter__()
# print x.next()

#python -m Pyro4.naming