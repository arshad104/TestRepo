import Pyro4
import sys

from IPython import embed

sys.path.append('../web')

from data_tree.models import MongoDB

class GraphSpecifications():

	def __init__(self):

		self.mongo = MongoDB()

		self.selected_model = None
		self.session_id = None
		self.running = False
		self.model_specs = {}
		self.modified = {}

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
		print "Creating Graphs..."
		self.api_server.start_graph(session_id, self.model_specs[session_id])

	def save_model_specs(self, json_object):

		session_id = json_object['session_id']
		if session_id in self.model_specs:
			del self.model_specs[session_id][:]

		self.mongo.insert_model(json_object)

	def get_model_names(self):

		models = self.mongo.get_all_models()
		names_list = [model["model"] for model in models]

		return names_list

	def save_graph_specs(self, json_object):

		self.mongo.insert_graph(json_object)
		self.mongo.add_graph_to_model( { 'model': self.selected_model, 'session_id': json_object['session_id'] }, json_object['name'] )

		self.model_specs[json_object['session_id']].append(json_object)

	def delete_graph_specs(self, session_id, graph_name):

		model_object = { 'model': self.selected_model, 'session_id': session_id }
		graph_object = { 'name': graph_name, 'session_id': session_id }

		self.mongo.remove_graph_from_model(model_object, graph)
		self.mongo.delete_graph(graph_object)

		for graph in self.model_specs[session_id]:
			if graph['name'] == graph_name and graph['session_id'] == session_id:
				self.model_specs[session_id].pop(graph)
				break

	def get_node_specs(self, graph_object, node_name):

		data = self.mongo.get_node(graph_object, node_name)

		try:
			mold = data['nodes'][0]
		except:
			return {}

		return mold

	def save_node_specs(self, graph_object, json_object):
		
		self.mongo.insert_node(graph_object, json_object)

		for graph in self.model_specs[graph_object['session_id']]:
			if graph['name'] == graph_object['name']:
				graph['nodes'].append(json_object)
				break

	def modify_node_specs(self, graph_object, node_specs):

		if graph_object['session_id'] in self.model_specs:
			for graph in self.model_specs[graph_object['session_id']]:
				if graph['name'] == graph_object['name']:
					for node in graph['nodes']:
						if node['name'] == node_specs['name']:
							node['opts'] = node_specs['opts']
							self.mongo.update_node(graph_object, node_specs)
							break

	def delete_node_specs(self, session_id, parent_graph, node_name, parent_relations):

		for p_graph, p_nodes in parent_relations.iteritems():
			if parent_graph == p_graph:
				for n_name in p_nodes:
					self.delete_child_specs(session_id, p_graph, n_name, node_name)
			else:
				for n_name in p_nodes:
					self.delete_child_specs(session_id, p_graph, n_name, [parent_graph, node_name])
		self.mongo.delete_node( { 'name': parent_graph, 'session_id': session_id }, node_name )

		for graph in self.model_specs[session_id]:
			if graph['name'] == parent_graph:
				for node in graph['nodes']:
					if node['name'] == node_name:
						graph['nodes'].remove(node)
						break

	def save_child_specs(self, graph_object, json_object):

		self.mongo.insert_child(graph_object, json_object)

		for graph in self.model_specs[graph_object['session_id']]:
			if graph['name'] == graph_object['name']:
				isExists = False;
				for relation in graph['relations']:
					if json_object['parent'] == relation['parent']:
						isExists = True
						children_rel = relation
						break

				if isExists:	
					if len(json_object['children']) == 1:
						children_rel['children'].append(json_object['children'][0])
					else:
						children_rel['children'].append(json_object['children'])
				else:
					graph['relations'].append(json_object)
				break

	def delete_child_specs(self, session_id, graph_name, node, child):

		self.mongo.delete_relations( { 'name': graph_name, 'session_id': session_id }, node, child )

		for graph in self.model_specs[session_id]:
			if graph['name'] == graph_name:
				for relation in graph['relations']:
					if node == relation['parent']:
						all_childrens = relation['children']
						for children in all_childrens:
							if children == child:
								all_childrens.remove(children)
								break
	
	def get_nodes_and_connections(self, session_id):
		
		nodes, connections = [], []

		if self.selected_model is not None:

			model = self.mongo.get_model( { "session_id": session_id, "model": self.selected_model } )

			graphs = self.mongo.get_graphs( model['graphs'], session_id )

			self.set_model_specs(session_id, graphs)

			for graph in graphs:
				graph_name = graph["name"]
				nodes.append( { "key": graph_name, "isGroup":True } )
				for node in graph["nodes"]:
					nodes.append( { "key": node["name"], "group": graph_name, "size": "60 60" } )
				for relation in graph["relations"]:
					link_from = relation["parent"]
					for child in relation["children"]:
						link_to = child
						if type(child) is list:
							link_to = child[1]

						connections.append( { "from": link_from, "to": link_to } )
		
		return { 'nodes': nodes, 'connections': connections , 'isRunning': self.running }

	def make_copy_of_model(self, json_object):

		graphs = self.model_specs[json_object['session_id']]

		copied_graphs = []
		copied_graph_names = []
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

			copied_graph_names.append(new_g_name)

			copied_graphs.append(graph)
		
		for new_graph in copied_graphs:
			relations = new_graph['relations']
			for relation in relations:
				for child in relation['children']:
					if type(child) is list:
							child[0] = names_dict[child[0]]

			self.mongo.insert_graph(new_graph)

		json_object['graphs'] = copied_graph_names
		self.mongo.insert_model(json_object)

		if json_object['session_id'] in self.model_specs:
			del self.model_specs[json_object['session_id']][:]

		self.model_specs[json_object['session_id']] = copied_graphs
		self.set_selected_model(json_object['model'])

		return json_object



spec_server = GraphSpecifications()

daemon = Pyro4.Daemon()
ns = Pyro4.locateNS()
uri = daemon.register(spec_server)
ns.register("spec_server", uri) 

print("Ready...")
daemon.requestLoop() 