import Pyro4
import sys

from IPython import embed
import cProfile, pstats, io

sys.path.append('../web')

from data_tree.models import MongoDB
from awok_data import settings

class GraphSpecifications():

	def __init__(self):

		self.mongo = MongoDB()

		self.running = False
		self.activeCanvas = None
		self.new_model = {}
		self.canvas_specs = {}
		self.modified_canvas = {}
		self.external_relations = {}

		self.server_manager = Pyro4.Proxy("PYRONAME:server_manager")
		self.api_server = Pyro4.async(Pyro4.Proxy("PYRONAME:apiserver"))

	def set_active_canvas(self, canvas_name):
		self.activeCanvas = canvas_name

	def get_active_canvas(self):
		return self.activeCanvas, self.running

	def start_process(self, session_id):
		self.server_manager.start_process()
		self.running = True

	def stop_process(self, session_id):
		self.server_manager.stop_process()
		self.running = False

	def restart_process(self, session_id):
		self.server_manager.restart_process()

	def get_model_names(self):
		models = self.mongo.get_all_models()
		names_list = [model["model"] for model in models]
		return names_list

	def create_graphs(self, session_id):
		specs_dictionary = self.canvas_specs[session_id]
		self.api_server.start_graph( session_id, specs_dictionary )

	def set_canvas_specs(self, session_id, canvas_specs):
		self.canvas_specs[session_id] = canvas_specs
		self.modified_canvas[session_id] = { "modified_graphs": set(), "new_graphs": [], 'delete_graphs': [] }
		self.external_relations[session_id] = { "add_relations": [], "delete_relations": [] }

	def get_canvas_specs(self, session_id):
		specs_dictionary = {}
		if session_id in self.canvas_specs:
			specs_dictionary = self.canvas_specs[session_id]
		return specs_dictionary

	def save_canvas_specs(self, session_id, canvas_data):
		if session_id in self.canvas_specs:
			self.canvas_specs.pop(session_id)
		self.new_model[session_id] = canvas_data
		self.set_canvas_specs( session_id, { 'canvas_graphs': [], 'canvas_relations': [] } )

	def save_graph_specs(self, session_id, graph_constr):
		self.canvas_specs[session_id]['canvas_graphs'].append(graph_constr)
		self.modified_canvas[session_id]['new_graphs'].append(graph_constr['name'])

	def delete_graph_specs(self, session_id, graph_name):
		if graph_name in self.modified_canvas[session_id]['modified_graphs']:
			self.modified_canvas[session_id]['modified_graphs'].remove(graph_name)
		if graph_name in self.modified_canvas[session_id]['new_graphs']:
			self.modified_canvas[session_id]['new_graphs'].remove(graph_name)
		else:
			self.modified_canvas[session_id]['delete_graphs'].append(graph_name)

		for graph in self.canvas_specs[session_id]['canvas_graphs']:
			if graph['name'] == graph_name:
				self.canvas_specs[session_id]['canvas_graphs'].remove(graph)

		for relation in self.canvas_specs[session_id]['canvas_relations']:
			if graph_name in relation['parent'] or graph_name in relation['child']:
				if relation in self.external_relations[session_id]['add_relations']:
					self.external_relations[session_id]['add_relations'].remove(relation)
				else:
					self.external_relations[session_id]['delete_relations'].append( relation )
				self.canvas_specs[session_id]['canvas_relations'].remove(relation)

	def get_node_specs(self, session_id, graph_name, node_name):
		mold = {}
		for graph in self.canvas_specs[session_id]['canvas_graphs']:
			if graph['name'] == graph_name:
				for node in graph['nodes']:
					if node['name'] == node_name:
						mold = node
		return mold

	def save_node_specs(self, session_id, graph_name, node_mold):
		if graph_name not in self.modified_canvas[session_id]['new_graphs']:
			self.modified_canvas[session_id]['modified_graphs'].add(graph_name)
		for graph in self.canvas_specs[session_id]['canvas_graphs']:
			if graph['name'] == graph_name:
				graph['nodes'].append(node_mold)
				break

	def update_node_specs(self, session_id, graph_name, node_specs):
		if graph_name not in self.modified_canvas[session_id]['new_graphs']:
			self.modified_canvas[session_id]['modified_graphs'].add(graph_name)
		if session_id in self.canvas_specs:
			for graph in self.canvas_specs[session_id]['canvas_graphs']:
				if graph['name'] == graph_name:
					for node in graph['nodes']:
						if node['name'] == node_specs['name']:
							node['opts'] = node_specs['opts']
							break

	def delete_node_specs(self, session_id, graph_name, node_name):
		if graph_name not in self.modified_canvas[session_id]['new_graphs']:
			self.modified_canvas[session_id]['modified_graphs'].add(graph_name)
		for graph in self.canvas_specs[session_id]['canvas_graphs']:
			if graph['name'] == graph_name:
				for node in graph['nodes']:
					if node['name'] == node_name:
						graph['nodes'].remove(node)
						break
	
	def save_child_relations(self, session_id, graph_name, child_relation):
		if graph_name not in self.modified_canvas[session_id]['new_graphs']:
			self.modified_canvas[session_id]['modified_graphs'].add(graph_name)
		for graph in self.canvas_specs[session_id]['canvas_graphs']:
			if graph['name'] == graph_name:
				isExists = False;
				for relation in graph['relations']:
					if child_relation['parent'] == relation['parent']:
						isExists = True
						children_rel = relation
						break
				if isExists:	
						children_rel['children'].append(child_relation['children'][0])
				else:
					graph['relations'].append(child_relation)
				break

	def delete_parent_relations(self, session_id, graph_name, child_node, parent_nodes):
		if graph_name not in self.modified_canvas[session_id]['new_graphs']:
			self.modified_canvas[session_id]['modified_graphs'].add(graph_name)
		for graph in self.canvas_specs[session_id]['canvas_graphs']:
			if graph['name'] == graph_name:
				for relation in graph['relations']:
					for parent_node in parent_nodes:
						if relation['parent'] == parent_node:
							relation['children'].remove(child_node)
							if len(relation['children']) < 1:
								graph['relations'].remove(relation)
							break

	def delete_child_relations(self, session_id, graph_name, parent_node):
		for graph in self.canvas_specs[session_id]['canvas_graphs']:
			if graph['name'] == graph_name:
				for relation in graph['relations']:
					if relation['parent'] == parent_node:
						graph['relations'].remove(relation)
						break

	def save_external_relations(self, session_id, relation):
		self.canvas_specs[session_id]['canvas_relations'].append(relation)
		self.external_relations[session_id]['add_relations'].append(relation)

	def delete_external_relations(self, session_id, node_relation):
		for i in range(len(self.canvas_specs[session_id]['canvas_relations'])):
			for relation in self.canvas_specs[session_id]['canvas_relations']:
				if relation['parent'] == node_relation:
					temp_relation = relation
				elif relation['child'] == node_relation:
					temp_relation = relation
				if temp_relation in self.external_relations[session_id]['add_relations']:
					self.external_relations[session_id]['add_relations'].remove(temp_relation)
				else:
					self.external_relations[session_id]['delete_relations'].append(temp_relation)
				self.canvas_specs[session_id]['canvas_relations'].remove(temp_relation)

	def get_canvas_nodes_and_connections(self, session_id):
		
		nodes, connections = [], []
		if self.activeCanvas is not None and session_id in self.canvas_specs:
			graphs = self.canvas_specs[session_id]['canvas_graphs']
			for graph in graphs:
				graph_name = graph["name"]
				nodes.append( { "key": graph_name, "isGroup":True } )
				for node in graph["nodes"]:
					nodes.append( { "key": node["name"], "group": graph_name, "size": "50 50", "color": self.getNodeColor(node['class']) } )
				for relation in graph["relations"]:
					link_from = relation["parent"]
					for child in relation["children"]:
						link_to = child
						connections.append( { "from": link_from, "to": link_to } )

			for canvas_rel in self.canvas_specs[session_id]['canvas_relations']:
				connections.append( { "from": canvas_rel["parent"][1], "to": canvas_rel["child"][1] } )
		
		return { 'nodes': nodes, 'connections': connections }

	def get_model_nodes_and_connections(self, session_id, username):
		
		nodes, connections = [], []
		if self.activeCanvas is not None:
			model = self.mongo.get_model( { "model": self.activeCanvas } )
			graphs = self.mongo.get_graphs(model["graphs"])
			self.set_canvas_specs(session_id, { "canvas_graphs": graphs, "canvas_relations": model["relations"] })

			for graph in graphs:
				graph_name = graph["name"]
				nodes.append( { "key": graph_name, "isGroup":True } )
				for node in graph["nodes"]:
					nodes.append( { "key": node["name"], "group": graph_name, "size": "50 50", "color": self.getNodeColor(node['class']) } )
				for relation in graph["relations"]:
					link_from = relation["parent"]
					for child in relation["children"]:
						link_to = child
						connections.append( { "from": link_from, "to": link_to } )

			for canvas_rel in model['relations']:
				connections.append( { "from": canvas_rel["parent"][1], "to": canvas_rel["child"][1] } )

		return { 'nodes': nodes, 'connections': connections }

	def make_copy_of_model(self, json_object):

		graphs = self.canvas_specs[json_object['session_id']]['canvas_graphs']

		self.isModified = False

		copied_graphs = []
		names_dict = {}

		for graph in graphs:
			name = graph['name']
			new_g_name = name + '_' + json_object['model']
			splitedName = name.split('_')
			if splitedName[-1] == 'origin':
				new_g_name = '_'.join(splitedName[:-1]+[json_object['model']]+[splitedName[-1]])

			graph['name'] = new_g_name
			graph['username'] = json_object['username']
			graph['session_id'] = json_object['session_id']
			
			names_dict[name] = new_g_name

			copied_graphs.append(graph)
		
		for new_graph in copied_graphs:
			relations = new_graph['relations']
			for relation in relations:
				for child in relation['children']:
					if type(child) is list:
							child[0] = names_dict[child[0]]
			
		#self.new_model[json_object['session_id']] = json_object

		if json_object['session_id'] in self.model_specs:
			del self.model_specs[json_object['session_id']][:]

		self.model_specs[json_object['session_id']] = copied_graphs

		return json_object

	def save_changes(self, session_id):
		embed()
		if session_id in self.new_model:
			for graph in self.canvas_specs[session_id]['canvas_graphs']:
				graph_name = graph['name']
				self.new_model[session_id]['graphs'].append(graph_name)
				if graph_name in self.modified_canvas[session_id]['new_graphs']:
					self.mongo.insert_graph(graph)
				elif graph_name in self.modified_canvas[session_id]['modified_graphs']:
					print 'The graph is modified!', graph_name
			for canvas_rel in self.canvas_specs[session_id]['canvas_relations']:
				self.new_model[session_id]['relations'].append(canvas_rel)
			model_specs = self.new_model[session_id]
			self.mongo.insert_model(model_specs)
			self.new_model.pop(session_id)
		else:
			for graph in self.canvas_specs[session_id]['canvas_graphs']:
				graph_name = graph['name']
				if graph_name in self.modified_canvas[session_id]['new_graphs']:
					self.mongo.insert_graph(graph)
					self.mongo.add_graph_to_model( { 'model': self.activeCanvas, 'session_id': session_id }, graph_name )
				elif graph_name in self.modified_canvas[session_id]['delete_graphs']:
					self.mongo.remove_graph_from_model( { 'model': self.activeCanvas, 'session_id': session_id }, graph_name )
				elif graph_name in self.modified_canvas[session_id]['modified_graphs']:
					print 'The graph "{name}" is modified!'.format(name = graph_name)

			delete_canvas_rel = self.external_relations[session_id]['delete_relations']
			add_canvas_rel = self.external_relations[session_id]['add_relations']
			for del_relation in delete_canvas_rel:
				self.mongo.remove_external_relations(session_id, self.activeCanvas, del_relation)
			for add_relation in add_canvas_rel:
				self.mongo.add_external_relations(session_id, self.activeCanvas, add_relation)

		del self.modified_canvas[session_id]['new_graphs'][:]
		del self.modified_canvas[session_id]['delete_graphs'][:]
		del self.external_relations[session_id]['add_relations'][:]
		del self.external_relations[session_id]['delete_relations'][:]
		while self.modified_canvas[session_id]['modified_graphs']:
			self.modified_canvas[session_id]['modified_graphs'].pop()

	def getNodeColor(self, nodeClass):
		return settings.NODE_COLORS[nodeClass]

# pr = cProfile.Profile()
# pr.enable()
# spec_server = GraphSpecifications()
# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
# ps.dump_stats('program.prof')

spec_server = GraphSpecifications()

daemon = Pyro4.Daemon()
ns = Pyro4.locateNS()
uri = daemon.register(spec_server)
ns.register("spec_server", uri) 

print("Graph Specs is running...")
daemon.requestLoop() 