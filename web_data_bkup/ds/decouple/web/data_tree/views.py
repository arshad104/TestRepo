 #!/usr/bin/env python -W ignore::DeprecationWarning

from django.contrib.auth.decorators import login_required
from django.template.defaulttags import register
from django.views.generic import View
from django.shortcuts import render
from django.http import QueryDict
from IPython import embed

import simplejson as json
import Pyro4

from base import *
from awok_data import settings

spec_server = Pyro4.Proxy("PYRONAME:spec_server")

class ModelAPI(APIView):

	def get(self, request, *args, **kwargs):

		model_name = request.GET.get('modelname')

		session_id = request.session.session_key
		username = request.user.username
		spec_server.set_selected_model(model_name)

		json_object = spec_server.get_model_nodes_and_connections(session_id, username)
		
		return self.render_response( json_object )
	
	def post(self, request, *args, **kwargs):

		model_name = request.POST.get('modelname')

		username = request.user.username
		session_id = request.session.session_key

		model_data = { "model": model_name, "username": username, 'session_id': session_id, 'graphs': [] }

		spec_server.set_selected_model(model_name)

		spec_server.save_model_specs(model_data)

		return self.render_response({'model':model_name})

	def put(self, request, *args, **kwargs):

		q = QueryDict(request.body)

		model_name = q.get('modelname')

		username = request.user.username
		session_id = request.session.session_key

		model_data = {"model":model_name, "username":username, 'session_id':session_id}

		spec_server.make_copy_of_model(model_data)

		return self.render_response(model_data)

	def delete(self, request, *args, **kwargs):

		q = QueryDict(request.body)

		graph_name = q.get("modelname")

		spec_server.delete_model_specs(graph_name)

		return self.render_response({"graph":graph_name})

class GraphAPI(APIView):

	def get(self, request, *args, **kwargs):

		session_id = request.session.session_key

		json_object = spec_server.get_nodes_and_connections(session_id)

		selected_model, isRunning = spec_server.get_selected_model()

		json_object["selected_model"] = selected_model
		json_object["isRunning"] = isRunning

		return self.render_response( json_object )

	def post(self, request, *args, **kwargs):

		graph_name = request.POST.get('graphname')

		session_id = request.session.session_key
		username = request.user.username

		json_object = { 'name': graph_name, 'type': 'empty', 'session_id': session_id, 'username': username, 'size':100, 'nodes': [], 'relations': [] }

		spec_server.save_graph_specs(json_object)
		
		return self.render_response({"graph":graph_name})

	def delete(self, request, *args, **kwargs):

		q = QueryDict(request.body)

		graph_name = q.get("graphname")

		session_id = request.session.session_key

		spec_server.delete_graph_specs(session_id, graph_name)

		return self.render_response({"graph":graph_name})

class NodesAPI(APIView):

	def get(self, request):

		graph_name = request.GET.get('graphname')
		node_name = request.GET.get('nodename')
		model_name = request.GET.get('modelname')
		session_id = request.session.session_key

		json_data = { 'name':graph_name, 'session_id': session_id }

		mold = spec_server.get_node_specs(json_data, node_name)
		shapes = ['image_shape', 'filter_shape']
		for shape in shapes:
			if shape in mold['opts']:
				mold['opts'][shape] = map(str, mold['opts'][shape])

		mold['graphname'] = graph_name
		
		if mold['class'] == "VectorsToMatrixNode":
			return self.render_response({"html":False})

		html = render(request, 'edit_config.html', mold)
		
		return html

	def post(self, request, *args, **kwargs):

		q = request.POST

		graph_name = q.get('graphname')
		node_class = q.get('nodeclass')
		node_name = q.get('nodename')

		options = get_node_opts(q)
		session_id = request.session.session_key
		node = { 'class': node_class, 'name': node_name, 'opts': options }
		to_graph = { 'name': graph_name, 'session_id': session_id }
		spec_server.save_node_specs(to_graph, node)

		return self.render_response({"node":node,"group":graph_name})

	def put(self, request, *args, **kwargs):

		q = QueryDict(request.body)

		graph_name = q.get('graphname')
		node_name = q.get('nodename')
		node_class = q.get('nodeclass')

		options = get_node_opts(q)

		node = {'class':node_class,'name':node_name,'opts':options}

		username = request.user.username
		session_id = request.session.session_key
		
		spec_server.modify_node_specs( { 'name': graph_name, 'session_id': session_id }, node )

		return self.render_response( { 'node': node } )

	def delete(self, request, *args, **kwargs):

		q = QueryDict(request.body)

		graph_name = q.get('graphname')
		node_name = q.get('nodename')
		parents = q.get('parents')
		parent_rel = json.loads(parents)

		session_id = request.session.session_key

		spec_server.delete_node_specs(session_id, graph_name, node_name, parent_rel)

		return self.render_response({"node":node_name})

class LinkesAPI(APIView):

	def post(self, request, *args, **kwargs):

		parent_node = request.POST.get('parentnode')
		child_node = request.POST.get('childnode')
		parent_graph = request.POST.get('parentgraph')
		child_graph = request.POST.get('childgraph')

		session_id = request.session.session_key

		children = []

		if parent_graph == child_graph:
			children.append(child_node)
		else:
			children.append([child_graph, child_node])

		to_graph = {"name": parent_graph, "session_id": session_id}

		relation = {'parent':parent_node,'children':children}

		spec_server.save_child_specs(to_graph, relation)

		return self.render_response({"from":parent_node,"to":child_node})

	def delete(self, request, *args, **kwargs):

		q = QueryDict(request.body)

		parent_node = q.get('parentnode')
		child_node = q.get('childnode')
		parent_graph = q.get('parentgraph')
		child_graph = q.get('childgraph')

		session_id = request.session.session_key

		child = child_node
		if parent_graph != child_graph:
			child = [child_graph, child_node]

		spec_server.delete_child_specs(session_id, parent_graph, parent_node, child)

		return self.render_response({"child":"child_node"})

class Attributes(APIView):
		
	def get(self, request):

		nodeClass = request.GET.get('nodeclass')

		context = {"nodeClass":nodeClass}

		html = render(request, 'config.html', context)
		
		return html

	def post(self, request):

		session_id = request.session.session_key

		spec_server.save_changes(session_id)
		
		return self.render_response({'modified': True})

class ProcessAPI(APIView):

	def get(self, request, *args, **kwargs):

		username = request.user.username
		session_id = request.session.session_key

		start = request.GET.get('start')
		loop = request.GET.get('loop')

		if loop == 'true':
			spec_server.start_loop(session_id)
		elif start == 'true':
			spec_server.start_process(session_id)
			allowEdit = False
			spec_server.create_graphs(session_id)
		elif start == 'false':
			allowEdit = True
			spec_server.stop_process(session_id)

		return self.render_response( { 'allowEdit': allowEdit } )

@login_required
def home(request):

	models = spec_server.get_model_names()
	selected_model, isRunning = spec_server.get_selected_model()

	return render( request, 'tree.html', { 'models': models, 'selected_model': selected_model, "isRunning": isRunning } )

def output_console(request):
	return render(request, 'errors.html')

@register.filter
def get_item(dictionary, key):

	return dictionary.get(key)

@register.filter
def joinby(value, arg):
	return arg.join(value)

@register.filter
def isDict(dictio, scalar):
	return type(scalar) is dict

def get_node_opts(q):

	options = {}

	for option in settings.NODE_OPTIONS:
		if option['name'] == 'init_scaler':
			if q.get('init_scale_type') or q.get('init_scale_value'):
				options['init_scaler'] = {'type': q.get('init_scale_type'), 'scale': float(q.get('init_scale_value'))}
		else:
			data = q.get(option['name'])
			if data is not None and data != "":
				if option['type'] == 'number':
					data = float(data)
				elif option['type'] == 'boolean':
					if data == 'True':
						data = True
					else:
						data = False
				elif option['name'] == 'a_func':
					data = {'name': data}
					if q.get('a_func_param'):
						data['params'] = {data['name']:float(q.get('a_func_param'))}
				elif option['name'] in ['filter_shape', 'image_shape']:
					data = map(int, data.split('x'))

				options[option['name']] = data

	return options