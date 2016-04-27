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

		self.selected_model = None
		self.session_id = None
		self.running = False
		self.isModified = True
		self.allowEdit = False

		self.new_model = {}
		self.model_specs = {}

		self.deleted_graphs = {}
		self.deleted_nodes = {}
		self.deleted_relations = {}

		self.server_manager = Pyro4.Proxy("PYRONAME:server_manager")
		self.api_server = Pyro4.async(Pyro4.Proxy("PYRONAME:apiserver"))

	def set_model_specs(self, session_id, specs):

		self.model_specs[session_id] = specs

	def get_model_specs(self, session_id):

		graphs = []

		if session_id in self.model_specs:
			graphs = self.model_specs[session_id]

		return graphs
	
	def set_selected_model(self, model):
		self.selected_model = model

	def get_selected_model(self):
		return self.selected_model, self.running

	def start_process(self, session_id):
		self.server_manager.start_process()
		self.running = True

	def stop_process(self, session_id):
		self.server_manager.stop_process()
		self.running = False

	def restart_process(self, session_id):
		self.server_manager.restart_process()

	def create_graphs(self, session_id):
		self.api_server.start_graph(session_id, self.model_specs[session_id])

	def save_model_specs(self, json_object):

		session_id = json_object['session_id']
		if session_id in self.model_specs:
			del self.model_specs[session_id][:]

		self.isModified = False
		self.allowEdit = True

		self.new_model[json_object['session_id']] = json_object
		self.model_specs[json_object['session_id']] = []

	def get_model_names(self):

		models = self.mongo.get_all_models()
		names_list = [model["model"] for model in models]

		return names_list

	def save_graph_specs(self, json_object):

		if self.isModified:
			json_object['isNewGraph'] = True

		self.model_specs[json_object['session_id']].append(json_object)

	def delete_graph_specs(self, session_id, graph_name):

		if session_id not in self.deleted_graphs:
			self.deleted_graphs[session_id] = []

		if session_id not in self.deleted_relations:
			self.deleted_relations[session_id] = []

		if self.isModified:
			self.deleted_graphs[session_id].append( { 'model': self.selected_model, 'name': graph_name, 'session_id': session_id } )

		for graph in self.model_specs[session_id]:
			for relation in graph['relations']:
				for i in range(len(relation['children'])):
					for child in relation['children']:
						if type(child) is list and child[0] == graph_name:
							relation['children'].remove(child)
							if self.isModified:
								self.deleted_relations[session_id].append( { 'name': graph['name'], 'session_id': session_id, 'node': relation['parent'], 'child': child } )
							if len(relation['children']) == 0:
								graph['relations'].remove(relation)

			if graph['name'] == graph_name and graph['session_id'] == session_id:
				self.model_specs[session_id].remove(graph)

	def get_node_specs(self, graph_object, node_name):

		mold = {}
		for graph in self.model_specs[graph_object['session_id']]:
			if graph['name'] == graph_object['name']:
				for node in graph['nodes']:
					if node['name'] == node_name:
						mold = node

		return mold

	def save_node_specs(self, graph_object, json_object):
		
		for graph in self.model_specs[graph_object['session_id']]:
			if graph['name'] == graph_object['name']:
				if self.isModified:
					json_object['isNewNode'] = True
					graph['isModified'] = True
				graph['nodes'].append(json_object)
				break

	def modify_node_specs(self, graph_object, node_specs):

		if graph_object['session_id'] in self.model_specs:
			for graph in self.model_specs[graph_object['session_id']]:
				if graph['name'] == graph_object['name']:
					for node in graph['nodes']:
						if node['name'] == node_specs['name']:
							if self.isModified:
								node['isModified'] = True
								graph['isModified'] = True
							node['opts'] = node_specs['opts']
							break

	def delete_node_specs(self, session_id, parent_graph, node_name, parent_relations):

		if session_id not in self.deleted_nodes:
			self.deleted_nodes[session_id] = []

		if self.isModified:
			self.deleted_nodes[session_id].append( { 'name': parent_graph, 'session_id': session_id, 'node': node_name } )
		
		for p_graph, p_nodes in parent_relations.iteritems():
			if parent_graph == p_graph:
				for n_name in p_nodes:
					self.delete_child_specs(session_id, p_graph, n_name, node_name)
			else:
				for n_name in p_nodes:
					self.delete_child_specs(session_id, p_graph, n_name, [parent_graph, node_name])

		for graph in self.model_specs[session_id]:
			if graph['name'] == parent_graph:
				for node in graph['nodes']:
					if node['name'] == node_name:
						graph['nodes'].remove(node)
						break

				for rel in graph['relations']:
					if rel['parent'] == node_name:
						graph['relations'].remove(rel)
						break

	def save_child_specs(self, graph_object, json_object):

		for graph in self.model_specs[graph_object['session_id']]:
			if graph['name'] == graph_object['name']:
				isExists = False;
				if self.isModified:
					graph['isModified'] = True

				for relation in graph['relations']:
					if json_object['parent'] == relation['parent']:
						if self.isModified:
							relation['isModified'] = True
						isExists = True
						children_rel = relation
						break

				if isExists:	
					if len(json_object['children']) == 1:
						children_rel['children'].append(json_object['children'][0])
					else:
						children_rel['children'].append(json_object['children'])
				else:
					if self.isModified:
						json_object['isModified'] = True
					graph['relations'].append(json_object)
				break

	def delete_child_specs(self, session_id, graph_name, node, child):

		if session_id not in self.deleted_relations:
			self.deleted_relations[session_id] = []

		if self.isModified:
			self.deleted_relations[session_id].append( { 'name': graph_name, 'session_id': session_id, 'node': node, 'child': child } )

		for graph in self.model_specs[session_id]:
			if graph['name'] == graph_name:
				for relation in graph['relations']:
					if node == relation['parent']:
						all_childrens = relation['children']
						for children in all_childrens:
							if children == child:
								all_childrens.remove(children)
								if len(all_childrens) < 1:
									graph['relations'].remove(relation)
								break
	
	def get_nodes_and_connections(self, session_id):
		
		nodes, connections = [], []
		if self.selected_model is not None and session_id in self.model_specs:
			graphs = self.model_specs[session_id]

			for graph in graphs:
				graph_name = graph["name"]
				nodes.append( { "key": graph_name, "isGroup":True } )
				for node in graph["nodes"]:
					nodes.append( { "key": node["name"], "group": graph_name, "size": "50 50", "color": self.getNodeColor(node['class']) } )
				for relation in graph["relations"]:
					link_from = relation["parent"]
					for child in relation["children"]:
						link_to = child
						if type(child) is list:
							link_to = child[1]

						connections.append( { "from": link_from, "to": link_to } )
		
		return { 'nodes': nodes, 'connections': connections , 'isRunning': self.running, 'allowEdit': self.allowEdit }

	def get_model_nodes_and_connections(self, session_id, username):
		
		nodes, connections = [], []
		if self.selected_model is not None:
			model = self.mongo.get_model( { "model": self.selected_model } )

			graphs = self.mongo.get_graphs( model['graphs'], model['session_id'] )

			self.set_model_specs(session_id, graphs)

			for graph in graphs:
				graph_name = graph["name"]
				nodes.append( { "key": graph_name, "isGroup":True } )
				for node in graph["nodes"]:
					nodes.append( { "key": node["name"], "group": graph_name, "size": "50 50", "color": self.getNodeColor(node['class']) } )
				for relation in graph["relations"]:
					link_from = relation["parent"]
					for child in relation["children"]:
						link_to = child
						if type(child) is list:
							link_to = child[1]

						connections.append( { "from": link_from, "to": link_to } )

			if model['username'] == username:
				self.allowEdit = True
			else:
				self.allowEdit = False

		return { 'nodes': nodes, 'connections': connections , 'isRunning': self.running, 'allowEdit': self.allowEdit }

	def make_copy_of_model(self, json_object):

		graphs = self.model_specs[json_object['session_id']]

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
			
		self.new_model[json_object['session_id']] = json_object

		if json_object['session_id'] in self.model_specs:
			del self.model_specs[json_object['session_id']][:]

		self.model_specs[json_object['session_id']] = copied_graphs

		return json_object

	def save_changes(self, session_id):
		
		if not self.isModified:
			for graph in self.model_specs[session_id]:
				graph_name = graph['name']
				self.new_model[session_id]['graphs'].append(graph_name)
				self.mongo.insert_graph(graph)
			self.mongo.insert_model(self.new_model[session_id])
			self.new_model.pop(session_id)
		else:
			if session_id in self.deleted_graphs:
				for d_graph in self.deleted_graphs[session_id]:
					g_name = d_graph.pop('name')
					self.mongo.remove_graph_from_model(d_graph, g_name)
					self.mongo.delete_graph( { 'name': g_name, 'session_id': d_graph['session_id'] } )
				del self.deleted_graphs[session_id][:]

			if session_id in self.deleted_nodes:
				for d_node in self.deleted_nodes[session_id]:
					n_name = d_node.pop('node')
					self.mongo.delete_node( d_node, n_name )
				del self.deleted_nodes[session_id][:]

			if session_id in self.deleted_relations:
				for d_relation in self.deleted_relations[session_id]:
					p_node = d_relation.pop('node')
					d_child = d_relation.pop('child')
					self.mongo.delete_relations( d_relation, p_node, d_child )
				del self.deleted_relations[session_id][:]

			for graph in self.model_specs[session_id]:
				graph_name = graph['name']
				graph_object = {'name': graph_name, 'session_id': session_id } 
				if 'isNewGraph' in graph:
					graph.pop('isNewGraph')
					self.mongo.insert_graph(graph)
					self.mongo.add_graph_to_model( { 'model': self.selected_model, 'session_id': session_id }, graph_name )
				elif 'isModified' in graph:
					graph.pop('isModified')
					for node in graph['nodes']:
						if 'isNewNode' in node:
							node.pop('isNewNode')
							self.mongo.insert_node( graph_object, node )
						elif 'isModified' in node:
							node.pop('isModified')
							self.mongo.update_node( graph_object, node )
					for relation in graph['relations']:
						if 'isModified' in relation:
							relation.pop('isModified')
							if len(relation['children']) == 0:
								self.mongo.delete_parent_rel( graph_object, relation['parent'] )
							else:
								print relation
								self.mongo.insert_child( graph_object, relation )

		self.isModified = True

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