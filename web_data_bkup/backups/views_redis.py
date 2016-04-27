from django.shortcuts import render
from django.http import HttpResponse
from django.views.generic import View
from IPython import embed

import simplejson as json
import sys
import pickle
import datetime, time

from awok_data.settings import redis_server , r

sys.path.append("..")  #Path to parent directory

from graph import MasterGraph, Graph

class GraphAPI(View):

	def get(self, request):

		context = {"nodes":[], "connections":[]}

		return HttpResponse(json.dumps(context), content_type='application/json')

	def post(self, request, *args, **kwargs):

		graph_name = request.POST.get('graphname')

		master_graph = MasterGraph({'batch_size':1})
		mold = {'name':graph_name,'type': 'empty','size':100}
		Graph({'name':graph_name,'type':mold['type'],'master_graph':master_graph,'batch_size':1}).build_graph({'batch_size':1,'master_graph':master_graph,'name':mold['name'],'mold':mold})
		
		set_master_object(graph_name, master_graph)
		zadd(graph_name)

		return HttpResponse(json.dumps({"graph":graph_name}))

class NodesAPI(View):

	def post(self, request, *args, **kwargs):

		graph_name = request.POST.get('graphname')
		node_class = request.POST.get('nodeclass')
		node_name = request.POST.get('nodename')

		master_graph = get_master_object(graph_name)
		master_graph.graph[graph_name].add_node({'class':node_class,'name':node_name,'opts':{}})
		set_master_object(graph_name, master_graph)

		return HttpResponse(json.dumps({"node":node_name,"group":graph_name}))

class LinkesAPI(View):

	def post(self, request, *args, **kwargs):

		parent_node = request.POST.get('parentnode')
		child_node = request.POST.get('childnode')
		parent_graph = request.POST.get('parentgraph')
		child_graph = request.POST.get('childgraph')

		children = []

		if parent_graph == child_graph:
			children.append(child_node)
		else:
			children.append([child_graph, child_node])

		mold = {'parent':parent_node,'children':children}

		master_graph = get_master_object(parent_graph)
		master_graph.graph[parent_graph].add_children(mold)
		set_master_object(parent_graph, master_graph)

		return HttpResponse(json.dumps({"from":parent_node,"to":child_node}))


def set_master_object(key, obj):

	pickled_object = pickle.dumps(obj)
	redis_server.set(key, pickled_object)

def get_master_object(key):

	pickled_object = redis_server.get(key)
	obj = pickle.loads(pickled_object)
	return obj

def zadd(key):

	# get current timestamp
	dt = datetime.datetime.now()
	# make time epoch
	t = time.mktime(dt.timetuple())

	r.zadd('all_graph_keys', key, t)

	#print r.zrange('all_graph_keys',0,-1)




# class Attributes(View):
		
# 	def get(self, request):

# 		nodeClass = request.GET.get('nodeclass')

# 		nodes =  ["text0","text00","text000"]
# 		if nodeClass == "class1":
# 			nodes =  ["text10","text11","text12","text13"]
# 		elif nodeClass == "class2":
# 			nodes =  ["text20","text21","text22", "text23","text24"]

# 		context = {"attributes":nodes}

# 		html = render(request, 'config.html', context)
		
# 		return html

def index(request):
	return render(request, 'tree.html')