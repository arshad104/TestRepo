import subprocess
import psutil
from IPython import embed
from time import sleep
import sys
import Pyro4

class ServerManager():

	def __init__(self):

		self.running_process = {}

	def start_all_servers(self):

		#p1 = subprocess.Popen(["python", "api_server.py"], stdout=subprocess.PIPE)
		p2 = subprocess.Popen(["python", "websocket.py"], stdout=subprocess.PIPE)
		p3 = subprocess.Popen(["python", "../web/manage.py", "runserver"], stdout=subprocess.PIPE)

		self.running_process["socket_server"] = p2
		self.running_process["django_server"] = p3
		#self.running_process["api_server"] = p1

		# lines_iterator = iter(p1.stdout.readline, b"")
		# for line in lines_iterator:
		# 	print "process1: ", line 

	def restart_all(self):
		pass

	def start_process(self):
		
		p1 = subprocess.Popen(["python", "api_server.py"], stdout=subprocess.PIPE)
		sleep(5)
		self.running_process["api_server"] = p1

		# lines_iterator = iter(p1.stdout.readline, b"")
		# for line in lines_iterator:
		# 	print "process1: ", line 

	def stop_process(self):

		try:
			proc = self.running_process["api_server"]
		except Exception, e:
			print "process already dead!"
			return
		
		proc.terminate()
		sleep(2)
		proc.kill()
		
		self.running_process.pop("api_server")

	def restart_process(self):

		self.stop_process()
		sleep(0.1)
		self.start_process()


sm = ServerManager()
sm.start_all_servers()

daemon = Pyro4.Daemon()               	# make a Pyro daemon
ns = Pyro4.locateNS()                  	# find the name server
uri = daemon.register(sm)   						# register the greeting maker as a Pyro object
ns.register("server_manager", uri)   		# register the object with a name in the name server

print("Processes Started...")
daemon.requestLoop() 
