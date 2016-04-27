from IPython import embed
import Pyro4
import os

import sys
sys.path.append('../web')
sys.path.append('../old_awok_data_science')

from graph import MasterGraph, Graph
from data_loader import MnistServer
from word_weight_mapper import WordWeightMapper

os.environ['CUDARRAY_BACKEND'] = 'cuda'
import cudarray as ca

class ServerAPI():

	def __init__(self):

		self.session_id = None
		self.master_graphs = {}
		self.observation_server = MnistServer({})
		self.word_weight_mapper = WordWeightMapper()

	def create_master_graphs(self, key):

		master_graph = MasterGraph({'iterations':50000000,'time_frame':2,'batch_size':1,'log_grad':False,'log_activation':False,'log_error':True, 'session_id':key})
		master_graph.observation_server = self.observation_server
		master_graph.word_weight_mapper = self.word_weight_mapper

		self.master_graphs[key] = master_graph

	def create_graph(self, mold):

		if mold['session_id'] not in self.master_graphs:
			self.create_master_graphs(mold['session_id'])

		self.session_id = mold["session_id"]

		Graph({'name':mold['name'],'type':mold['type'],'master_graph':self.master_graphs[self.session_id],'batch_size':1}).build_graph({'batch_size':1,'master_graph':self.master_graphs[self.session_id],'name':mold['name'],'mold':mold})

	def add_node_to_graph(self, graph_name, node):

		if node['class'] == "TargetNode" or node['class'] == "DataNode":

			_iter = self.master_graphs[self.session_id].observation_server.__iter__()

			observation = _iter.next()

			y_vec = observation['digit-label']
			x = observation['image']
			x.shape = (256,784)
			x = ca.array(x) / 255.0
			y_vec  = ca.array(y_vec)
			
			if node['class'] == "TargetNode":
				node['opts']['y'] = y_vec

			elif node['class'] == "DataNode":
				node['opts']['key'] = x

		self.master_graphs[self.session_id].graph[graph_name].add_node(node)

	def add_child_to_node(self, graph_name, relations):

		self.master_graphs[self.session_id].graph[graph_name].add_children(relations)		

	def create_backend_graphs(self, session_id, graphs):

		for graph in graphs:
			graph_object = { 'name': graph['name'], 'type': graph['type'], 'session_id': session_id, 'size': graph['size'] }
			self.create_graph(graph_object)

			for node in graph['nodes']:
				self.add_node_to_graph( graph['name'], node )

		for graph in graphs:
			relations = graph['relations']
			for relation in relations:
				for child in relation['children']:
					mold = {'parent': relation['parent'], 'children':[child]}
					self.add_child_to_node(graph['name'], mold)

	def start_graph(self, session_id, graphs):

		self.create_backend_graphs(session_id, graphs)

		_iter = self.master_graphs[self.session_id].observation_server.__iter__()

		i = 0

		total_iterations = self.master_graphs[self.session_id].iterations

		while i < total_iterations:
			
			self.master_graphs[self.session_id].reset_except_origin()

			observation = _iter.next()

			y_vec = observation['digit-label']
			x = observation['image']
			x.shape = (256,784)
			x = ca.array(x) / 255.0
			y_vec  = ca.array(y_vec)

			for graph in graphs:
				if graph['name'].split('_')[-1] != 'origin':
					g_name = graph['name']
					mold = {'name':g_name,'type': graph['type'], 'session_id':self.session_id,'size':graph['size']}
					self.create_graph(mold)

					for node in graph['nodes']:
						if node['class'] == "TargetNode":
							node['opts']['y'] = y_vec

						elif node['class'] == "DataNode":
							node['opts']['key'] = x
						self.add_node_to_graph(g_name, node)

					relations = graph['relations']
					for relation in relations:
						for child in relation['children']:
							mold = {'parent': relation['parent'], 'children':[child]}
							self.add_child_to_node(g_name, mold)
			
			self.master_graphs[self.session_id].forward()
			self.master_graphs[self.session_id].backward()
			self.master_graphs[self.session_id].update_weights()
			self.master_graphs[self.session_id].print_error(i)

			i += 1

	def get_session_id(self):
		return self.session_id

	def show_relations(self):

		if self.master_graphs:
			for g in self.master_graphs[self.session_id].graph:
				print 'Graph: --> ', g
				print '========================================='
				for n in self.master_graphs[self.session_id].graph[g].nodes:
					print 'Node: --> ', n
					parents = self.master_graphs[self.session_id].graph[g].nodes[n].parents_rel
					childs = self.master_graphs[self.session_id].graph[g].nodes[n].children_rel
					print 'Parent Nodes: --> ', parents
					print 'Child Nodes: --> ', childs
					print '-----------------------------------------'


apiserver = ServerAPI()

daemon = Pyro4.Daemon()               	# make a Pyro daemon
ns = Pyro4.locateNS()                  	# find the name server
uri = daemon.register(apiserver)   	# register the greeting maker as a Pyro object
ns.register("apiserver", uri)   		# register the object with a name in the name server

print("Ready...")
daemon.requestLoop() 
