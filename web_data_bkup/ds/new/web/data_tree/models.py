from pymongo import MongoClient
from IPython import embed

MONGO_DB_IP = '127.0.0.1'
MONGO_DB_PORT = 27017

DEFAULT_DB = "test2_db"
GRAPH_COLLECTION = 'test2_graphs'
MODEL_COLLECTION = 'test2_models'

class MongoDB(object):

	def __init__(self):

		self.database = DEFAULT_DB
		self.graph_collection = GRAPH_COLLECTION
		self.model_collection = MODEL_COLLECTION
		self.conn = MongoClient(MONGO_DB_IP,MONGO_DB_PORT)
		self.db = self.conn[self.database]
		self.graph_coll = self.db[self.graph_collection]
		self.model_coll = self.db[self.model_collection]

	def insert_model(self, model_data):
		
		self.model_coll.insert(model_data)

		print "Model saved successfully!"

		return model_data

	def add_graph_to_model(self, model_data, graph_name):

		self.model_coll.update( model_data, { "$addToSet": { "graphs": graph_name } } )

		print 'Model updated successfully.'
		
		return model_data

	def remove_graph_from_model(self, model_object, graph_name):

		self.model_coll.update( model_object, { "$pull": { "graphs": graph_name } } )

		print 'Model updated successfully.'
		
		return graph_name

	def add_external_relations(self, session_id, model_name, relation):
		self.model_coll.update( { 'model': model_name, 'session_id': session_id }, { '$addToSet': {'relations': relation } } )

	def remove_external_relations(self, session_id, model_name, relation):
		self.model_coll.update( { 'model': model_name, 'session_id': session_id }, { '$pull': {'relations': relation } } )

	def get_all_models(self):
		
		cursor = self.model_coll.find({}, { '_id':0, 'session_id':0, 'graphs':0 })
		
		models = [model for model in cursor]

		return models

	def get_model(self, obj):
		
		obj = self.model_coll.find_one( obj, { '_id':0 } )

		return obj

	def insert_graph(self, graph_data):
		
		self.graph_coll.insert_one(graph_data)
		
		print 'Graph inserted successfully.'
		
		return graph_data

	def delete_graph(self, graph_data):
		
		self.graph_coll.remove(graph_data)
		
		print 'Graph deleted successfully.'
		
		return graph_data

	def get_graph(self, obj):
		
		cursor = self.graph_coll.find( obj, { '_id':0 } )
		
		data = [graph for graph in cursor]

		return data

	def get_graphs(self, list_obj):
		
		cursor = self.graph_coll.find( { "name": { "$in": list_obj } }, { '_id':0 } )
		
		data = [graph for graph in cursor]

		return data

	def insert_node(self, graph_data, node_data):
		
		self.graph_coll.update( graph_data, { "$addToSet": { "nodes": node_data } } )

		print 'Node inserted successfully.'
		
		return node_data

	def update_node(self, graph_data, node_data):

		graph_data['nodes.name'] = node_data['name']

		self.graph_coll.update( graph_data, { "$set": { "nodes.$.opts": node_data['opts'] } } )

		print 'Node updated successfully.'
		
		return node_data

	def delete_node(self, graph_data, node_name):

		self.delete_parent_rel( graph_data, node_name )

		self.graph_coll.update( graph_data, { "$pull": { "nodes": { "name": node_name } } } )

		print 'Node deleted successfully.'
		
		return node_name

	def get_node(self, graph_data, node_name):

		obj = self.graph_coll.find_one( graph_data, {"nodes": { "$elemMatch": { "name": node_name } }, '_id':0} )

		return obj

	def insert_child(self, graph_data, relation):

		graph_data['relations.parent'] = relation["parent"]

		try:
			self.graph_coll.update( graph_data, { '$set': {'relations.$.children': relation['children'] } }, True )
		except:
			graph_data.pop('relations.parent')
			self.graph_coll.update( graph_data, { '$addToSet': {'relations': relation } }, True )

	def update_child(self, graph_data, relation):

		graph_data['relations.parent'] = relation["parent"]

		self.graph_coll.update( graph_data, { '$set': {'relations.$.children': relation['children'] } }, True )

		print "Updated Node relation"

	def delete_relations(self, graph_data, node, child):
		
		graph_data['relations.parent'] = node
		
		self.graph_coll.update( graph_data, { "$pull": { "relations.$.children": child } } )

		print 'Child relation deleted successfully.'
		
		return child

	def delete_parent_rel(self, graph_data, node_name):

		self.graph_coll.update( graph_data, { "$pull": {"relations": {"parent": node_name } } } )

		print 'Parent relation deleted successfully.'
		
		return node_name