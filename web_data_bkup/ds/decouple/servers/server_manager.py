from subprocess import Popen, PIPE, STDOUT
import psutil
from IPython import embed
from time import sleep
import sys
import Pyro4
import os

class ServerManager():

	def __init__(self):

		self.running_process = {}
		self.log_file = open("../error_log.txt", "w")

	def start_all_servers(self):

		p1 = Popen(["python", "graph_spec.py"], stderr=self.log_file)
		p2 = Popen(["python", "websocket.py"], stderr=self.log_file)
		p3 = Popen(["python", "../web/manage.py", "runserver"])
		#stderr=open("../django_warnings.txt", "w")

		self.running_process["graph_spec_server"] = p1
		self.running_process["socket_server"] = p2
		self.running_process["django_server"] = p3

	def restart_all(self):
		pass

	def start_process(self):
		
		proc = Popen(["python", "api_server.py"], stderr=self.log_file)
		sleep(5)
		self.running_process["api_server"] = proc

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
		sleep(1)
		self.start_process()


sm = ServerManager()
sm.start_all_servers()

daemon = Pyro4.Daemon()               	# make a Pyro daemon
ns = Pyro4.locateNS()                  	# find the name server
uri = daemon.register(sm)   						# register the greeting maker as a Pyro object
ns.register("server_manager", uri)   		# register the object with a name in the name server

print("Server manager is running...")
daemon.requestLoop() 
