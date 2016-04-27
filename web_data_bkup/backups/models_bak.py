from pymongo import MongoClient
import simplejson

import logging

LOG = logging.getLogger("django")

MONGO_DB_IP = '127.0.0.1'
MONGO_DB_PORT = 27017

DEFAULT_DB = "test_db"

class MongoDB(object):

	def __init__(self):

		self.database=DEFAULT_DB

		self.db_name = self.database
		self.conn = MongoClient(MONGO_DB_IP,MONGO_DB_PORT)
		self.db = self.conn[self.database]

	def save_graph_data(self, graph_data):
		
		try:
			self.coll = self.conn[self.database]['graphs']
		except Exception as Err:
			LOG.exception(Err)
		self.coll.insert_one(graph_data)
		print 'Graph inserted successfully.'
		
		return

	def delete_graph_data(self, graph_data):
		
		self.delete_node_data({'graph':graph_data['name'],'session_id':graph_data['session_id']})

		try:
			self.coll = self.conn[self.database]['graphs']
		except Exception as Err:
			LOG.exception(Err)
		
		self.coll.remove(graph_data)
		print 'Graph deleted successfully.'
		
		return

	def save_node_data(self, node_data):
		
		try:
			self.coll = self.conn[self.database]['nodes']
		except Exception as Err:
			LOG.exception(Err)
		
		self.coll.insert_one(node_data)
		print 'Node inserted successfully.'
		
		return

	def delete_node_data(self, node_data, deleting_graph=False):
		
		if deleting_graph:
			self.delete_node_relations(node_data)
		else:
			self.delete_node_relations({'session_id':node_data['session_id'], 'parent':node_data['name']})

		try:
			self.coll = self.conn[self.database]['nodes']
		except Exception as Err:
			LOG.exception(Err)
		
		self.coll.delete_many(node_data)
		print 'Node inserted successfully.'
		
		return

	def save_node_relations(self, relations_data):
		
		try:
			self.coll = self.conn[self.database]['relations']
		except Exception as Err:
			LOG.exception(Err)
		
		self.coll.insert_one(relations_data)
		print 'Relation inserted successfully.'
		
		return

	def update_node_relations(self, relations_data):
		
		try:
			self.coll = self.conn[self.database]['relations']
		except Exception as Err:
			LOG.exception(Err)

		child = relations_data['children'][0]
		if len(relations_data['children']) > 1:
			child = relations_data['children']

		self.coll.update_one(
			{ "session_id": relations_data['session_id'], "parent": relations_data['parent'] },
			{ "$addToSet": { "children": child } }
		)

		print 'Relation Updated successfully.'
		
		return

	def delete_node_relations(self, relations_data, child={}):
		
		try:
			self.coll = self.conn[self.database]['relations']
		except Exception as Err:
			LOG.exception(Err)
		
		if not child:
			self.coll.delete_many(relations_data)
		else:
			self.coll.update_one(relations_data, {'$pop':child})
		
		print 'Relation deleted successfully.'
		
		return

	def get_graphs_by_key(self, session_id):
		
		try:
			self.coll = self.conn[self.database]['graphs']
		except Exception as Err:
			LOG.exception(Err)
		
		objects = self.coll.find({"session_id":session_id},{ '_id':0})
		
		graphs = []

		for graph in objects:
			graphs.append(graph)
		
		return graphs

	def get_nodes_for_graph(self, session_id, graph_name):
		
		try:
			self.coll = self.conn[self.database]['nodes']
		except Exception as Err:
			LOG.exception(Err)

		query = {'session_id':session_id,'graph':graph_name}
		exclude = { '_id':0, 'session_id':0, 'graph':0}

		objects = self.coll.find(query, exclude)
		
		nodes = []

		for node in objects:
			nodes.append(node)
		
		return nodes

	def get_nodes_relations(self, session_id):
		
		try:
			self.coll = self.conn[self.database]['relations']
		except Exception as Err:
			LOG.exception(Err)

		query = {'session_id':session_id}
		exclude = { '_id':0, 'session_id':0}

		objects = self.coll.find(query, exclude)
		
		relations = []

		for rel in objects:
			relations.append(rel)
		
		return relations

try:
	mongodb = MongoDB()
except Exception,e:
	LOG.exception(e)
