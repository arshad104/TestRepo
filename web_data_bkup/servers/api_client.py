import Pyro4

api_server = Pyro4.Proxy("PYRONAME:apiserver")

api_server.create_graph({'name':"test",'type': 'empty','size':100})
api_server.add_node_to_graph('test', {'class':"WeightNode",'name':"Node1",'opts':{}})
api_server.add_node_to_graph('test', {'class':"LossNode",'name':"Node2",'opts':{}})
api_server.add_child_to_node('test', {'parent':"Node1",'children':["Node2"]})
api_server.add_child_to_node('test', {'parent':"Node2",'children':["Node1"]})

api_server.show_relations()