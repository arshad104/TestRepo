from IPython import embed
import Pyro4
import os
import cProfile, pstats, io
import gc

from copy import deepcopy
import sys
sys.path.append('../web')
sys.path.append('../awok_data_science')

from core.engine.graph import MasterGraph, Graph
from core.engine.word_weight_mapper import WordWeightMapper
from data_iterators.data_loader import MnistServer

os.environ['CUDARRAY_BACKEND'] = 'cuda'
os.environ['CUDNN_ENABLED'] = '1'

import cudarray as ca

class ServerAPI():

	def __init__(self):

		self.session_id = None
		self.master_graphs = {}
		self.observation_server = MnistServer({})
		self.word_weight_mapper = WordWeightMapper()
		self.graph_specs_server = Pyro4.Proxy("PYRONAME:spec_server")

	def create_master_graphs(self, key):

		master_graph = MasterGraph({'iterations':500000,'time_frame':2,'batch_size':1,'log_grad':False,'log_activation':False,'log_error':True, 'session_id':key})
		master_graph.observation_server = self.observation_server
		master_graph.word_weight_mapper = self.word_weight_mapper

		self.master_graphs[key] = master_graph

	def create_graph(self, mold):

		if mold['session_id'] not in self.master_graphs:
			self.create_master_graphs(mold['session_id'])

		self.session_id = mold["session_id"]

		Graph({'name':mold['name'],'type':mold['type'],'master_graph':self.master_graphs[self.session_id],'batch_size':1}).build_graph({'batch_size':1,'master_graph':self.master_graphs[self.session_id],'name':mold['name'],'mold':mold})

	def add_node(self, graph_name, node):

		self.master_graphs[self.session_id].graph[graph_name].add_node(node)

	def add_child(self, graph_name, relations):

		self.master_graphs[self.session_id].graph[graph_name].add_children(relations)		

	def create_backend_graphs(self, session_id, graphs_list):

		for graph_dict in graphs_list:
			self.create_graph({ 'name': graph_dict['name'], 'type': graph_dict['type'], 'session_id': session_id, 'size': graph_dict['size'] })

			for node in graph_dict['nodes']:
				if 'filter_shape' in node['opts']:
					node['opts']['filter_shape'] = tuple(node['opts']['filter_shape'])
				self.add_node( graph_dict['name'], node )
				
		for graph_dict in graphs_list:
			relations = graph_dict['relations']
			for relation in relations:
				for child in relation['children']:
					mold = {'parent': relation['parent'], 'children':[child]}
					self.add_child(graph_dict['name'], mold)

	def start_graph(self, session_id, graphs_list):
		
		self.create_backend_graphs(session_id, graphs_list)

		_iter = self.master_graphs[self.session_id].observation_server.__iter__()
		
		total_iterations = self.master_graphs[self.session_id].iterations

		i = 0

		while i < total_iterations:

			self.master_graphs[self.session_id].reset_except_origin()
			
			observation = _iter.next()
			y_vec = observation['digit-label']
			y_vec  = ca.array(y_vec)
			x = observation['image']
			
			for graph_dict in graphs_list:
				if graph_dict['name'].split('_')[-1] != 'origin':
					g_name = graph_dict['name']
					self.create_graph({'name':g_name,'type': graph_dict['type'], 'session_id':self.session_id,'size':graph_dict['size']})

					for node in graph_dict['nodes']:
						if node['class'] == "TargetNode":
							node['opts']['y'] = y_vec
						elif node['class'] == "DataNode":
							if 'image_shape' in node['opts']:
								x.shape =	(1,1,node['opts']['image_shape'][0],node['opts']['image_shape'][1])
								node['opts']['image_shape'] = x.shape
							else:
								x.shape = (1,x.size)
							x = ca.array(x) / 255.0
							node['opts']['key'] = x

						self.add_node(g_name, node)

						node['opts'].pop('graph')
					for node in graph_dict['nodes']:
						if node['class'] == "TargetNode":
							node['opts'].pop('y')
						elif node['class'] == "DataNode":
							if 'image_shape' in node['opts']:
								node['opts']['image_shape'] = [node['opts']['image_shape'][2],node['opts']['image_shape'][3]]
							node['opts'].pop('key')
					
					relations = graph_dict['relations']
					for relation in relations:
						for child in relation['children']:
							rel_mold = {'parent': relation['parent'], 'children':[child]}
							self.add_child(g_name, rel_mold)
			
			self.master_graphs[self.session_id].forward()
			self.master_graphs[self.session_id].backward()
			self.master_graphs[self.session_id].update_weights()
			self.master_graphs[self.session_id].print_error(i)
			self.master_graphs[self.session_id].reset_grads()

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

# pr = cProfile.Profile()
# pr.enable()
# apiserver = ServerAPI()
# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.dump_stats('program.prof')

apiserver = ServerAPI()

daemon = Pyro4.Daemon()               	# make a Pyro daemon
ns = Pyro4.locateNS()                  	# find the name server
uri = daemon.register(apiserver)   	# register the greeting maker as a Pyro object
ns.register("apiserver", uri)   		# register the object with a name in the name server

print("API server is running...")
daemon.requestLoop() 
