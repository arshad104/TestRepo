from random import shuffle
from IPython import embed
import nn_functions as nn_func
import datetime
import math
from neural_nodes import *
import cProfile, pstats, io
import numpy as np
import matplotlib.pyplot as plt
import ipdb
import influx_pusher as influx
import os

class MasterGraph():
  def __init__(self,kwargs):
    self.iterations = self.arg_exists(kwargs,'iterations')
    self.log_error = self.arg_exists(kwargs,'log_error')
    self.log_grad = self.arg_exists(kwargs,'log_grad')
    self.trainee_graph = self.arg_exists(kwargs,'trainee_graph')
    self.log_activation = self.arg_exists(kwargs,'log_activation')
    self.error_history = []
    self.classification_history = []
    self.index_array = []
    self.batch_size = 1
    self.graph = {}
    self.seq_clock = 0

  def arg_exists(self,dictio,arg):
    if arg in dictio: 
      return dictio[arg]
    else:
      return None  

  def add_graph(self,graph):
    self.graph[graph.name] = graph
    graph.master_graph = self

  def remove_graph(self,graph):
    self.graph.pop(graph.name)  

  def reset_except_origin(self):
    for _tuple in self.graph.items():
      _tuple[1].destroy_graph()
           
  def add_children(self,relation):
    self.graph[relation['parent']].add_graph_children(relation['children'])    
  
  def reset_parent_path(self,node):
    candidates = node.parents()
    new_candidates = []
                
    while candidates != []:
      for node in candidates:
        for n in node.parents():
          new_candidates.append(n)
        node.activation_status = False

      candidates = new_candidates
      new_candidates = []


  def has_cycle(self):

    not_visited_set = set()
    partialy_visited_set = set()
    visited_set = set()

    for g_name, g in self.graph.iteritems():
      for n_name, node in g.nodes.iteritems():
        not_visited_set.add(node)

    while len(not_visited_set) > 0:
      current = next(iter(not_visited_set))
      if self.dfs(current, not_visited_set, partialy_visited_set, visited_set) == True:
        #print "hey here", current.name
        #self.cycle.append(current)
        #embed()
        return True

    return False  

  def dfs(self, current, not_visited_set, partialy_visited_set, visited_set):
    self.move_vertex(current, not_visited_set, partialy_visited_set)
    for child in current.children_rel:
      neighbor = current.evaluate_link(child)
      if neighbor in visited_set:
        continue
      if neighbor in partialy_visited_set:
        print 'The Graph is cyclic.'
        print "---> ", neighbor.graph.name +'-'+neighbor.name
        return True
      if self.dfs(neighbor, not_visited_set, partialy_visited_set, visited_set) == True:
        print "---> ", neighbor.graph.name +'-'+neighbor.name
        #self.cycle.insert(0, neighbor)
        return True

    self.move_vertex(current, partialy_visited_set, visited_set)
    return False

  def move_vertex(self, vertex, source_set, destination_set):
    source_set.remove(vertex)
    destination_set.add(vertex)

  def get_cycle(self, node):
    for parent in node.parents_rel:
      parent_node = node.evaluate_link(parent)


  def top_sort(self):
    print self.has_cycle()
    embed()
    return
    parent_list = []
    sorted_list = []
    not_visited_nodes = set()

    counter = 0
    for g_name, g in self.graph.iteritems():
      for n_name, node in g.nodes.iteritems():
        if node.parents_rel == []:
          parent_list.append(node)
        counter+=1
        node.temp_parent_rel = list(node.parents_rel)
    #ipdb.set_trace()
    while parent_list != []:
      # print 'parent-node: ', parent_list[0].name
      # print 'parents_list: ', [p.name for p in parent_list]
      # print 'sorted_list: ', [s.name for s in sorted_list]
      parent_node = parent_list.pop(0)
      sorted_list.append(parent_node)
      for child in parent_node.children_rel:
        child_node = parent_node.evaluate_link(child)
        try:
          child_node.temp_parent_rel.remove([parent_node.graph.name,parent_node.name])
          if child_node in not_visited_nodes:
            not_visited_nodes.remove(child_node)
        except:
          embed()  
        if child_node.temp_parent_rel == []:
          parent_list.append(child_node)
        else:
          child_node.temp_parent_rel
          print "parents of "+child_node.name+" not visited yet"
          not_visited_nodes.add(child_node)
          #embed()
    embed()
    return sorted_list
        

  def forward(self):
    candidates = self.top_sort()
    for node in list(reversed(candidates)):
      node.forward()
    candidates = []    
    
  def backward(self):
    candidates = self.top_sort()
    for node in candidates:
      node.backward()
    candidates = []

  def debug_grads(self,loss_graph,loss_node):          
    for g_name, g in self.graph.iteritems():      
      for name, node in g.nodes.iteritems():
        if node.debug == 'gradient':
          node.debug_grad(loss_graph,loss_node)     
        
  def update_weights(self):          
    for g_name, g in self.graph.iteritems():      
      for name, node in g.nodes.iteritems():
        if node.weight_update_status == True:
          node.update_weights()

  def take_snapshots(self):          
    for g_name, g in self.graph.iteritems():      
      for name, node in g.nodes.iteritems():
        if node.weight_update_status == True:
          node.take_snapshot()

  def restore_snapshots(self):          
    for g_name, g in self.graph.iteritems():      
      for name, node in g.nodes.iteritems():
        if node.weight_update_status == True:
          node.restore_snapshot()

  def add_noise(self):
    for g_name, g in self.graph.iteritems():      
      for name, node in g.nodes.iteritems():
        if node.weight_update_status == True:
          node.add_noise()                         

  def collect_reward(self):
    reward = 0
    counter = 0        
    for g_name, g in self.graph.iteritems():      
      for name, node in g.nodes.iteritems():
        if node.__class__.__name__ == 'LossNode':
          reward += node.mean_error
          counter += 1.0

    return reward / counter      

  def persist_weights(self,i):
    if (i % self.weight_persist_freq == 0) and (i != 0):      
      for g_name, g in self.graph.iteritems():      
        for name, node in g.nodes.iteritems():
          if node.weight_update_status == True:
            node.save_weights()

  def reset_grads(self,and_memory = False):
    for g_name, g in self.graph.iteritems():      
      for name, node in g.nodes.iteritems():
        node.gradient_status = False
        if and_memory:
          node.grad = []
  
  def print_error(self,i,node_name = '',session_id='none'):
    for g_name, g in self.graph.iteritems():      
      for name, node in g.nodes.iteritems():
       
        if node.__class__.__name__ == 'LossNode' or node.type == 'loss':
          
          # err = node.error[0]
          err = np.array(node.a)[0]
          # if err[0] < 0.0001:
          #   embed()
          self.error_history.append( err )

          # print str(err) + " - " + node.graph.name
          temp_ary = []
     
          if g.print_classifier:
            target = node.graph.nodes['target']
            prediction = node.graph.master_graph.graph[node.graph.mold['input_node'][0]].nodes[node.graph.mold['input_node'][1]]

            equal_mean = np.mean(ca.argmax(target.a,axis=1) == ca.argmax(prediction.a,axis=1))
            self.classification_history.append(equal_mean ) 

          if g.print_classifier:
           
            ave_classification_history = self.running_average(1000,self.classification_history)
            ave_classification_history_100 = self.running_average(100,self.classification_history)
            
          if g.print_classifier:

            print "Running Average Classification (1000) => {ave_classification_history}".format(ave_classification_history = ave_classification_history)
            print "Running Average Classification (100) => {ave_classification_history_100}".format(ave_classification_history_100 = ave_classification_history_100)
            influx.push("classification_rate",[['process_id',os.getpid()],['session_id',session_id]],float(equal_mean))
    
          if self.log_error:

            influx.push("error_rate",[['process_id',os.getpid()],['session_id',session_id]],float(err))


    ave_error_history_100 = self.running_average(100,self.error_history)      
    ave_error_history_1000 = self.running_average(1000,self.error_history)
    min_error_history_1000 = min(self.error_history)
    
    # print "Running Average Error (1) => {ave_error_history}".format(ave_error_history = ratio)  
    print "Running Average Error (100) => {ave_error_history_100}".format(ave_error_history_100 = ave_error_history_100)
    print "Running Average Error (1000) => {ave_error_history_1000}".format(ave_error_history_1000 = ave_error_history_1000)
    print "Running Minimum Error (1000) => {min_error_history_1000}".format(min_error_history_1000 = min_error_history_1000)
    
    print "Batch number => {i}".format(i = i)
    print "------------------------------------"    
      
  def running_average(self,scale,ary):
    return float(sum(ary[-scale:])) / float(min([scale,len(ary)]))    
      
  def check_sync(self,_type):
    sync_list = []
    for g_name, g in self.graph.iteritems():
      for name, node in g.nodes.iteritems():
        if _type == 'forward':
          sync_list.append(node.activation_status)
        elif _type == 'backward':
          sync_list.append(node.gradient_status)   
    return sum(sync_list) == len(sync_list)



class Graph:
  def __init__(self,kwargs):
    self.name = self.arg_exists(kwargs,'name')
    self.y = self.arg_exists(kwargs,'y')
    self.print_classifier = False
    self.friends = self.arg_exists(kwargs,'friends')
    self.batch_size = self.arg_exists(kwargs,'batch_size')
    self.master_graph = self.arg_exists(kwargs,'master_graph')
    self.graph_index = self.arg_exists(kwargs,'graph_index')
    self.type = self.arg_exists(kwargs,'type')
    self.master_graph.add_graph(self)
    self.dt = self.arg_exists(kwargs,'data_object')
    self.nodes = {}
    self.children_rel = []
    self.parent_rel = []
    self.delta = []
 
  def add_node(self,node_constructor):
    node_constructor['opts']['graph'] = self
    constructor = globals()[node_constructor['class']]
    instance = constructor(node_constructor['opts'])
    instance.name = node_constructor['name']
    self.nodes[node_constructor['name']] = instance
  
  def add_children(self,relation):
    self.nodes[relation['parent']].add_children(relation['children'])

  def destroy_graph(self):
    if self.name.split('_')[-1] != 'origin':
      for _tuple in self.nodes.items():
        _tuple[1].destroy()
      self.master_graph.graph.pop(self.name,None)

  def children(self):
    children = set([])
    for node_name , node_object in self.nodes.iteritems():
      for child in node_object.children_rel:
        if child.__class__.__name__ == 'list':
          children.add(child[0])
    children = list(children)
    
    children = [self.master_graph.graph[child] for child in children]

    return children      

  def parents(self):
    parents = set([])
    for node_name , node_object in self.nodes.iteritems():
      for parent in node_object.parents_rel:
        if parent.__class__.__name__ == 'list':
          parents.add(parent[0])
    parents = list(parents)
    
    parents = [self.master_graph.graph[parent] for parent in parents]

    return parents      

  def evaluate_link(self,relation):
    return self.graph.master_graph.graph[reference] 
       
  def arg_exists(self,dictio,arg):
    if arg in dictio: 
      return dictio[arg]
    else:
      return None

  def add_bias_sub_nets(self) : 
    node_names = self.nodes.keys()
    for n_name in node_names:
      node = self.nodes[n_name]
      if node.add_bias:
        parents_cache = list(node.parents_rel)
        children_cache = list(node.children_rel)
        name = self.name + "-" + n_name
        bias_node_name = self.mold['type'] + "-" + n_name
        input_node = [self.name,n_name]
        fan_out = node.fan_out
        dropout = node.dropout
        debug = node.debug
        node.destroy(destroy_dangling=False)
        self.add_node({'class':'DotProductNode','name':n_name,'opts':{'dropout':dropout,'debug':debug}}) 
        mold = {'name':bias_node_name,'type':'bias_unit','input_node':input_node,'size':fan_out }
       
        g = Graph({'name':name,'master_graph':self.master_graph,'batch_size':1})
        g.build_graph({'batch_size':self.batch_size,'master_graph':self.master_graph,'name':mold['name'],'mold':mold})  

        for parent in parents_cache:
       
          self.master_graph.graph[parent[0]].add_children({'parent':parent[1],'children':[[g.name,'output']]})

        for child in children_cache:
          if child.__class__.__name__ == 'list':
            self.add_children({'parent':n_name,'children':[[child[0],child[1]]]})
          else:
            self.add_children({'parent':n_name,'children':[child]})

  # http://arxiv.org/pdf/1502.03167v3.pdf
  def add_batch_normalization_sub_nets(self) : 
    node_names = self.nodes.keys()
    for n_name in node_names:
      node = self.nodes[n_name]
      if node.add_bias:
        parents_cache = list(node.parents_rel)
        children_cache = list(node.children_rel)
        name = self.name + "-" + n_name
        bias_node_name = self.mold['type'] + "-" + n_name
        input_node = [self.name,n_name]
        fan_out = node.fan_out
        dropout = node.dropout
        debug = node.debug
        node.destroy(destroy_dangling=False)
        self.add_node({'class':'DotProductNode','name':n_name,'opts':{'dropout':dropout,'debug':debug}}) 
        mold = {'name':bias_node_name,'type':'batch_normalize','input_node':input_node,'size':fan_out }
       
        g = Graph({'name':name,'master_graph':self.master_graph,'batch_size':1})
        g.build_graph({'batch_size':self.batch_size,'master_graph':self.master_graph,'name':mold['name'],'mold':mold})  

        for parent in parents_cache:
          self.master_graph.graph[parent[0]].add_children({'parent':parent[1],'children':[[g.name,'output']]})

        for child in children_cache:
          if child.__class__.__name__ == 'list':
            self.add_children({'parent':n_name,'children':[[child[0],child[1]]]})
          else:
            self.add_children({'parent':n_name,'children':[child]})               

  def build_graph(self,kwargs):    
    
    if kwargs['mold']['type'] == 'empty':
      pass

    elif kwargs['mold']['type'] == 'bias_unit_origin':
      pass

    elif kwargs['mold']['type'] == 'bias_unit':
      name = kwargs['mold']['name']
      self.mold = kwargs['mold']
      input_node = kwargs['mold']['input_node']
      size = kwargs['mold']['size']
      self.add_node({'class':'VectorAddNode','name':'output','opts':{}})
      if kwargs['mold']['name'] in self.master_graph.graph['bias_unit_origin'].nodes:
        self.add_children({'parent':'output','children':[input_node,['bias_unit_origin',name]]})
      else:
        self.master_graph.graph['bias_unit_origin'].add_node({'class':'OnesNode','name':name,'opts':{'dropout':False,'fan_in':1,'fan_out':size,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':0.0001}}})
        self.add_children({'parent':'output','children':[input_node,['bias_unit_origin',name]]})

    elif kwargs['mold']['type'] == 'batch_normalize_origin':
      pass    

    elif kwargs['mold']['type'] == 'batch_normalize':
      name = kwargs['mold']['name']
      input_node = kwargs['mold']['input_node']
      size = kwargs['mold']['size']
      self.add_node({'class':'VectorAddNode','name':'output','opts':{}})
      self.add_node({'class':'VectorAddNode','name':'mean_add','opts':{}})
      self.add_node({'class':'HadamardNode','name':'hada_1','opts':{'fan_in':size}})
      self.add_node({'class':'HadamardNode','name':'hada_2','opts':{'fan_in':size}})
      self.mold = kwargs['mold']

      beta_name = kwargs['mold']['name'] + "-" + "beta"
      if beta_name in self.master_graph.graph['batch_normalize_origin'].nodes:
        self.add_children({'parent':'output','children':['hada_1',['batch_normalize_origin',beta_name]]})
      else:
        self.master_graph.graph['batch_normalize_origin'].add_node({'class':'OnesNode','name':beta_name,'opts':{'dropout':False,'fan_in':1,'fan_out':size,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':0.1}}})
        self.add_children({'parent':'output','children':['hada_1',['batch_normalize_origin',beta_name]]})                  
      
      gama_name = kwargs['mold']['name'] + "-" + "gama"
      if gama_name in self.master_graph.graph['batch_normalize_origin'].nodes:
        self.add_children({'parent':'hada_1','children':[['batch_normalize_origin',gama_name],'hada_2']})
      else:
        self.master_graph.graph['batch_normalize_origin'].add_node({'class':'OnesNode','name':gama_name,'opts':{'dropout':False,'fan_in':1,'fan_out':size,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':10000.1}}})
        self.add_children({'parent':'hada_1','children':[['batch_normalize_origin',gama_name],'hada_2']})
      
      variance_name = kwargs['mold']['name'] + "-" + "variance"
      if variance_name in self.master_graph.graph['batch_normalize_origin'].nodes:
        self.add_children({'parent':'hada_2','children':[['batch_normalize_origin',variance_name],'mean_add']})
        self.master_graph.graph['batch_normalize_origin'].add_children({'parent':variance_name,'children':[input_node]})
      else:
        self.master_graph.graph['batch_normalize_origin'].add_node({'class':'EmpericalVarianceNode','name':variance_name,'opts':{'skip_grad':True}})
        self.add_children({'parent':'hada_2','children':[['batch_normalize_origin',variance_name],'mean_add']})
        self.master_graph.graph['batch_normalize_origin'].add_children({'parent':variance_name,'children':[input_node]})

      mean_name = kwargs['mold']['name'] + "-" + "mean"
      if mean_name in self.master_graph.graph['batch_normalize_origin'].nodes:
        self.add_children({'parent':'mean_add','children':[['batch_normalize_origin',mean_name],input_node]})
        self.master_graph.graph['batch_normalize_origin'].add_children({'parent':mean_name,'children':[input_node]})
      else:
        self.master_graph.graph['batch_normalize_origin'].add_node({'class':'EmpericalMeanNode','name':mean_name,'opts':{'skip_grad':True}})
        self.add_children({'parent':'mean_add','children':[['batch_normalize_origin',mean_name],input_node]})
        self.master_graph.graph['batch_normalize_origin'].add_children({'parent':mean_name,'children':[input_node]})
      # embed()




    elif kwargs['mold']['type'] == 'vector_to_matrix_adapter_origin':
      self.mold = kwargs['mold']
      input_node = kwargs['mold']['input_node']
      in_size = kwargs['mold']['in_size']
      out_size = kwargs['mold']['out_size']
      
      self.add_node({'class':'WeightNode','name':'adapter','opts':{'fan_in':in_size,'fan_out':out_size,'weight_update_status': True,'init_scaler':{'type':'random','scale':0.1}}})
      self.add_node({'class':'WeightNode','name':'outer_vec','opts':{'fan_in':1,'fan_out':out_size,'weight_update_status': True,'init_scaler':{'type':'random','scale':0.1}}})
      self.add_node({'class':'WeightNode','name':'non_span','opts':{'fan_in':out_size,'fan_out':out_size,'weight_update_status': True,'init_scaler':{'type':'random','scale':0.1}}})

    elif kwargs['mold']['type'] == 'vector_to_matrix_adapter':
      self.mold = kwargs['mold']
      input_node = kwargs['mold']['input_node']
      size = kwargs['mold']['size']
      self.print_classifier = True


      self.add_node({'class':'FunctionNode','name':'adapter_func','opts':{'a_func':{'name':'tanh'}}})
      self.add_node({'class':'FunctionNode','name':'outer_func','opts':{'a_func':{'name':'tanh'}}})
      self.add_node({'class':'FunctionNode','name':'output','opts':{'a_func':{'name':'tanh'}}})
      self.add_node({'class':'TransposeNode','name':'transpose','opts':{}})
      
      self.add_node({'class':'DotProductNode','name':'adapter_dot','opts':{}})
      self.add_node({'class':'DotProductNode','name':'outer_dot','opts':{}})
      self.add_node({'class':'DotProductNode','name':'output_dot','opts':{}})
    

      self.add_children({'parent':'output','children':['output_dot']})
      self.add_children({'parent':'output_dot','children':['outer_func',['vector_to_matrix_adapter_origin','non_span']]})
      self.add_children({'parent':'outer_func','children':['outer_dot']})
      self.add_children({'parent':'outer_dot','children':['transpose',['vector_to_matrix_adapter_origin','outer_vec']]})
      self.add_children({'parent':'transpose','children':['adapter_func']})
      self.add_children({'parent':'adapter_func','children':['adapter_dot']})
      self.add_children({'parent':'adapter_dot','children':[input_node,['vector_to_matrix_adapter_origin','adapter']]})  

    elif kwargs['mold']['type'] == 'gibbs_actor_critic_loss':
      self.mold = kwargs['mold']
      input_node = kwargs['mold']['input_node']
      size = kwargs['mold']['size']
      self.print_classifier = True

      self.add_node({'class':'FunctionNode','name':'half_x_square','opts':{'a_func':{'name':'half_x_square'}}})
      
      self.add_node({'class':'HadamardNode','name':'hada_1','opts':{'fan_in':size}})
    
      self.add_node({'class':'VectorAddNode','name':'add','opts':{}})
      self.add_node({'class':'WeightNode','name':'1-column','opts':{'fan_in':size,'fan_out':1,'weight_update_status': False,'init_scaler':{'type':'ones','scale':1.0/10.0}}})
      self.add_node({'class':'WeightNode','name':'min-vec','opts':{'fan_in':1,'fan_out':size,'weight_update_status': False,'init_scaler':{'type':'ones','scale':-1.0}}})
      self.add_node({'class':'DotProductNode','name':'loss-dot','opts':{}})
      self.add_node({'class':'TargetNode','name': 'target','opts':{'fan_in':1,'fan_out':1,'y':kwargs['mold']['y']}})
      self.add_node({'class':'VectorAddNode','name':'loss','opts':{'type':'loss'}})

      self.add_children({'parent':'loss','children':['loss-dot']})
      self.add_children({'parent':'loss-dot','children':['1-column','half_x_square']})
      self.add_children({'parent':'half_x_square','children':['add']})
      self.add_children({'parent':'add','children':['hada_1',input_node]})
      self.add_children({'parent':'hada_1','children':['target','min-vec']})  

    elif kwargs['mold']['type'] == 'attention':
      self.mold = kwargs['mold']
      _children = kwargs['mold']['_children']
      
      self.add_node({'class':'VectorsToMatrixNode','name':'vecs','opts':{}})
      self.add_children({'parent':'vecs','children':_children})
      
    elif kwargs['mold']['type'] == 'max_log_prob_loss':
      self.mold = kwargs['mold']
      input_node = kwargs['mold']['input_node']
      seq_len = kwargs['mold']['seq_len']
      size = kwargs['mold']['size']
      graph_index = kwargs['mold']['graph_index']
      y = kwargs['mold']['y']
      self.print_classifier = False
      # (-1.0 / float(seq_len))
      if graph_index == 0:
        self.add_node({'class':'VectorAddNode','name':'loss','opts':{'type':'loss'}})

      self.add_node({'class':'FunctionNode','name':'log','opts':{'a_func':{'name':'log_plus_1'}}})
      self.add_node({'class':'HadamardNode','name':'hada','opts':{'fan_in':size}})
      self.add_node({'class':'TargetNode','name': 'target','opts':{'fan_in':1,'fan_out':1,'y':y}})
      self.add_node({'class':'WeightNode','name':'1-column','opts':{'fan_in':size,'fan_out':1,'weight_update_status': False,'init_scaler':{'type':'ones','scale':(-1.0 / float(seq_len))  }}})
      self.add_node({'class':'DotProductNode','name':'loss-dot','opts':{}})

      if graph_index == 0:
        self.add_children({'parent':'loss','children':['loss-dot']})
      self.add_children({'parent':'loss-dot','children':['1-column','hada']})
      self.add_children({'parent':'hada','children':['target','log']})
      self.add_children({'parent':'log','children':[input_node]})
      
    elif kwargs['mold']['type'] == 'square_error_loss':
      self.mold = kwargs['mold']
      input_node = kwargs['mold']['input_node']
      size = kwargs['mold']['size']
      self.print_classifier = True

      self.add_node({'class':'FunctionNode','name':'half_x_square','opts':{'a_func':{'name':'half_x_square'}}})
      
      self.add_node({'class':'HadamardNode','name':'hada_1','opts':{'fan_in':size}})
    
      self.add_node({'class':'VectorAddNode','name':'add','opts':{}})
      self.add_node({'class':'WeightNode','name':'1-column','opts':{'fan_in':size,'fan_out':1,'weight_update_status': False,'init_scaler':{'type':'ones','scale':1.0/10.0}}})
      self.add_node({'class':'WeightNode','name':'min-vec','opts':{'fan_in':1,'fan_out':size,'weight_update_status': False,'init_scaler':{'type':'ones','scale':-1.0}}})
      self.add_node({'class':'DotProductNode','name':'loss-dot','opts':{}})
      self.add_node({'class':'TargetNode','name': 'target','opts':{'fan_in':1,'fan_out':1,'y':kwargs['mold']['y']}})
      self.add_node({'class':'VectorAddNode','name':'loss','opts':{'type':'loss'}})

      self.add_children({'parent':'loss','children':['loss-dot']})
      self.add_children({'parent':'loss-dot','children':['1-column','half_x_square']})
      self.add_children({'parent':'half_x_square','children':['add']})
      self.add_children({'parent':'add','children':['hada_1',input_node]})
      self.add_children({'parent':'hada_1','children':['target','min-vec']})
    
    elif kwargs['mold']['type'] == 'cross_entropy_loss':
      self.mold = kwargs['mold']
      input_node = kwargs['mold']['input_node']
      size = kwargs['mold']['size']
      self.print_classifier = True

      self.add_node({'class':'FunctionNode','name':'log_1','opts':{'a_func':{'name':'log'}}})
      self.add_node({'class':'FunctionNode','name':'log_2','opts':{'a_func':{'name':'log'}}})
      self.add_node({'class':'FunctionNode','name':'1_minus_1','opts':{'a_func':{'name':'1_minus'}}})
      self.add_node({'class':'FunctionNode','name':'1_minus_2','opts':{'a_func':{'name':'1_minus'}}})
      self.add_node({'class':'HadamardNode','name':'hada_1','opts':{'fan_in':size}})
      self.add_node({'class':'HadamardNode','name':'hada_2','opts':{'fan_in':size}})
      self.add_node({'class':'VectorAddNode','name':'add','opts':{}})
      self.add_node({'class':'WeightNode','name':'1-column','opts':{'fan_in':size,'fan_out':1,'weight_update_status': False,'init_scaler':{'type':'ones','scale':-1.0/10.0}}})
      self.add_node({'class':'DotProductNode','name':'loss-dot','opts':{}})
      self.add_node({'class':'TargetNode','name': 'target','opts':{'fan_in':1,'fan_out':1,'y':kwargs['mold']['y']}})
      self.add_node({'class':'VectorAddNode','name':'loss','opts':{'type':'loss'}})

      self.add_children({'parent':'loss','children':['loss-dot']})
      self.add_children({'parent':'loss-dot','children':['1-column','add']})
      self.add_children({'parent':'add','children':['hada_1','hada_2']})
      self.add_children({'parent':'hada_1','children':['target','log_1']})
      self.add_children({'parent':'log_1','children':[input_node]})
      self.add_children({'parent':'hada_2','children':['1_minus_1','log_2']})
      self.add_children({'parent':'1_minus_1','children':['target']})
      self.add_children({'parent':'log_2','children':['1_minus_2']})
      self.add_children({'parent':'1_minus_2','children':[input_node]})

    elif kwargs['mold']['type'] == 'ff_origin':
      self.add_node({'class':'WeightNode','name':'n5','opts':{'fan_in':300,'fan_out':10,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.0,'alpha':0.01,'init_scaler':{'type':'random','scale':0.1}}})
      self.add_node({'class':'WeightNode','name':'n10','opts':{'dropout':False,'fan_in':784,'fan_out':600,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.2,'alpha':0.01,'init_scaler':{'type':'random','scale':0.1}}}) 
      self.add_node({'class':'WeightNode','name':'n8','opts':{'dropout':False,'fan_in':600,'fan_out':300,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.2,'alpha':0.01,'init_scaler':{'type':'random','scale':0.1}}})
      # self.add_node({'class':'WeightNode','name':'bias','opts':{'dropout':False,'fan_in':1,'fan_out':10,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.2,'alpha':0.005,'init_scaler':0.1}})        

    elif kwargs['mold']['type'] == 'ff':
      origin_graph = 'ff_origin'

      self.add_node({'class':'FunctionNode','name':'n3','opts':{'a_func':{'name':'softmax'},'fan_in':2400,'fan_out':2400}})
      # self.add_node({'class':'VectorAddNode','name':'bias-add','opts':{}})
      self.add_node({'class':'DotProductNode','name':'n4','opts':{'fan_in':100,'fan_out':100}})
      self.add_node({'class':'DotProductNode','name':'n9.1','opts':{'fan_in':100,'fan_out':100}})      
      self.add_node({'class':'DataNode','name':'n12','opts':{'key':kwargs['mold']['key'],'skip_grad': True,'a_func':{'name':'scalar','params':{'scalar':1.0}}}})
      self.add_node({'class':'FunctionNode','name':'n6','opts':{'dropout':False,'a_func':{'name':'relu'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'n9','opts':{'dropout':False,'a_func':{'name':'relu'},'fan_in':100,'fan_out':100}})
      # self.add_node({'class':'VectorAddNode','name':'n4.a-sum','opts':{'fan_in':100,'fan_out':100}})
      self.add_node({'class':'DotProductNode','name':'n7','opts':{'fan_in':100,'fan_out':100}})

      
      self.add_children({'parent':'n3','children':['n4']})
      # self.add_children({'parent':'bias-add','children':[[origin_graph,'bias'],'n4']})
      self.add_children({'parent':'n4','children':['n6',[origin_graph,'n5']]})

      self.add_children({'parent':'n6','children':['n7']})
     
      self.add_children({'parent':'n7','children':['n9',[origin_graph,'n8']]})
      self.add_children({'parent':'n9','children':['n9.1']})
      self.add_children({'parent':'n9.1','children':['n12',[origin_graph,'n10']]})

    elif kwargs['mold']['type'] == 'ff_circ_origin':
      self.add_node({'class':'WeightNode','name':'n5','opts':{'fan_in':784,'fan_out':10,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.0,'alpha':0.005,'init_scaler':10.1}})

      self.add_node({'class':'OnesNode','name':'n10','opts':{'dropout':False,'fan_in':1,'fan_out':784,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.2,'alpha':0.005,'init_scaler':5.1}}) 
     
      self.add_node({'class':'OnesNode','name':'n8','opts':{'dropout':False,'fan_in':1,'fan_out':784,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.2,'alpha':0.005,'init_scaler':5.1}})    

    elif kwargs['mold']['type'] == 'ff_circ':
      self.add_node({'class':'LossNode','name': 'n1','opts':{'a_func':{'name':'classifier'},'fan_in':1,'fan_out':1}})
      self.add_node({'class':'TargetNode','name': 'n2','opts':{'fan_in':1,'fan_out':1,'y':kwargs['mold']['y']}})
      self.add_node({'class':'FunctionNode','name':'n3','opts':{'a_func':{'name':'softmax'},'fan_in':2400,'fan_out':2400}})
      self.add_node({'class':'CircularConvNode','name':'n4','opts':{'fan_in':100,'fan_out':100}})
      
      self.add_node({'class':'CircularConvNode','name':'n9.1','opts':{'fan_in':100,'fan_out':100}})      
      self.add_node({'class':'DataNode','name':'n12','opts':{'key':kwargs['mold']['key'],'skip_grad': True,'a_func':{'name':'scalar','params':{'scalar':1.0}}}})
      self.add_node({'class':'FunctionNode','name':'n6','opts':{'dropout':False,'a_func':{'name':'sigmoid'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'n9','opts':{'dropout':False,'a_func':{'name':'sigmoid'},'fan_in':100,'fan_out':100}})
      # self.add_node({'class':'VectorAddNode','name':'n4.a-sum','opts':{'fan_in':100,'fan_out':100}})
      self.add_node({'class':'DotProductNode','name':'n7','opts':{'fan_in':100,'fan_out':100}})

      self.add_children({'parent':'n1','children':['n2','n3']})
      self.add_children({'parent':'n3','children':['n4']})
      
      self.add_children({'parent':'n4','children':['n6',['ff_origin','n5']]})

      self.add_children({'parent':'n6','children':['n7']})
     
      self.add_children({'parent':'n7','children':['n9',['ff_origin','n8']]})
      self.add_children({'parent':'n9','children':['n9.1']})
      self.add_children({'parent':'n9.1','children':['n12',['ff_origin','n10']]})  

    elif kwargs['mold']['type'] == 'conv_origin':

      self.add_node({'class':'FilterNode','name':'f-1','opts':{'n_channels':1,'n_filters':16,'filter_shape':(7,7),'fan_in':1,'fan_out':1,'weight_update_status': True,'weight_decay':0.001,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':1.1}}})
      self.add_node({'class':'FilterNode','name':'f-2','opts':{'n_channels':16,'n_filters':32,'filter_shape':(7,7),'fan_in':1,'fan_out':1,'weight_update_status': True,'weight_decay':0.001,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':1.1}}})
      self.add_node({'class':'FilterNode','name':'f-3','opts':{'n_channels':32,'n_filters':512,'filter_shape':(64,22),'fan_in':1,'fan_out':1,'weight_update_status': True,'weight_decay':0.001,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':1.1}}})
     
    elif kwargs['mold']['type'] == 'conv':
      self.add_node({'class':'DataNode','name':'n12','opts':{'key':kwargs['mold']['key'],'img_shape':kwargs['mold']['img_shape'],'skip_grad': True,'a_func':{'name':'scalar','params':{'scalar':0.0000001}}}})
      
      self.add_node({'class':'ConvolutionNode','name':'conv-1','opts':{}})
      self.add_node({'class':'ConvolutionNode','name':'conv-2','opts':{}})
      self.add_node({'class':'ConvolutionNode','name':'conv-3','opts':{'no_padding':True}})
     
      self.add_node({'class':'FunctionNode','name':'func-1','opts':{'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'func-2','opts':{'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'func-3','opts':{'reform':True,'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
      
      self.add_node({'class':'PoolNode','name':'pool-1','opts':{'fan_in':100,'fan_out':100}})
      self.add_node({'class':'PoolNode','name':'pool-2','opts':{'fan_in':100,'fan_out':100}})
      
      self.add_children({'parent':'func-3','children':['conv-3']})
      self.add_children({'parent':'conv-3','children':['pool-2',['conv_origin','f-3']]})
      self.add_children({'parent':'pool-2','children':['func-2']})
      self.add_children({'parent':'func-2','children':['conv-2']})
      self.add_children({'parent':'conv-2','children':['pool-1',['conv_origin','f-2']]})
      self.add_children({'parent':'pool-1','children':['func-1']})
      self.add_children({'parent':'func-1','children':['conv-1']})
      self.add_children({'parent':'conv-1','children':['n12',['conv_origin','f-1']]})
     
    elif kwargs['mold']['type'] == 'lstm_origin':
      x_dim = 2400
      mem_cells = 1024  
      
      if kwargs['mold']['loss_gate']:
        self.add_node({'class':'WeightNode','name':'n6','opts':{'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.0,'alpha':0.005,'init_scaler':1.1}})
        self.add_node({'class':'OnesNode','name':'n4-ones','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.0,'alpha':0.005,'init_scaler':0.001}})
        self.add_node({'class':'WeightNode','name':'n6.a','opts':{'fan_in':mem_cells,'fan_out':2400,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.0,'alpha':0.005,'init_scaler':1.1}})
        self.add_node({'class':'OnesNode','name':'n4.a-ones','opts':{'fan_in':1,'fan_out':2400,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.0,'alpha':0.005,'init_scaler':0.001}}) 
      
      self.add_node({'class':'OnesNode','name':'n5','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':1.1}})

      self.add_node({'class':'OnesNode','name':'n18','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':1.1}})

      self.add_node({'class':'OnesNode','name':'n301','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':0.001}})
     

      self.add_node({'class':'FilterNode','name':'n27','opts':{'n_channels':1,'n_filters':64,'filter_shape':(1,128),'fan_in':x_dim,'fan_out':mem_cells,'weight_update_status': False,'weight_decay':0.001,'momentum':0.2,'alpha':5.5,'init_scaler':2.1}})
      self.add_node({'class':'FilterNode','name':'n27.a','opts':{'n_channels':64,'n_filters':128,'filter_shape':(1,64),'fan_in':x_dim,'fan_out':mem_cells,'weight_update_status': False,'weight_decay':0.001,'momentum':0.2,'alpha':5.5,'init_scaler':2.1}})
      self.add_node({'class':'FilterNode','name':'n27.b','opts':{'n_channels':128,'n_filters':300,'filter_shape':(1,512),'fan_in':x_dim,'fan_out':mem_cells,'weight_update_status': False,'weight_decay':0.001,'momentum':0.2,'alpha':5.5,'init_scaler':2.1}})
      
      self.add_node({'class':'WeightNode','name':'n24.3','opts':{'dropout':False,'fan_in':300,'fan_out':mem_cells,'weight_update_status': False,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':4.1}})
      
      
      self.add_node({'class':'WeightNode','name':'n28','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
      self.add_node({'class':'WeightNode','name':'n29','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})

      self.add_node({'class':'WeightNode','name':'n21-weight','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
      self.add_node({'class':'WeightNode','name':'n22-weight','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
      self.add_node({'class':'WeightNode','name':'n31-weight','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
      self.add_node({'class':'WeightNode','name':'n7-weight','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})

      self.add_node({'class':'OnesNode','name':'n21-ones','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.001,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
      self.add_node({'class':'OnesNode','name':'n22-ones','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.001,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
      self.add_node({'class':'OnesNode','name':'n31-ones','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.001,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
      self.add_node({'class':'OnesNode','name':'n7-ones','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.001,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
    
    elif kwargs['mold']['type'] == 'lstm':

      graph_index = kwargs['mold']['graph_index']
      output_gate = kwargs['mold']['output_gate']

      if graph_index == 0:
        previous_graph = 'lstm_origin'
      else:
        previous_graph = "lstm" + "_" + str(graph_index-1)

      mem_cells = 1024  

      if kwargs['mold']['loss_gate']:
        self.add_node({'class':'LossNode','name': 'n1','opts':{'a_func':{'name':'square-error'},'fan_in':1,'fan_out':1}})
        self.add_node({'class':'TargetNode','name': 'n2','opts':{'fan_in':1,'fan_out':1,'y':kwargs['mold']['y']}})
        self.add_node({'class':'FunctionNode','name':'n3','opts':{'a_func':{'name':'linear'},'fan_in':2400,'fan_out':2400}})
        self.add_node({'class':'FunctionNode','name':'n3.a','opts':{'dropout':False,'a_func':{'name':'sin'},'fan_in':2400,'fan_out':2400}})    
        self.add_node({'class':'VectorAddNode','name':'n4-sum','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
        self.add_node({'class':'DotProductNode','name':'n4','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
        self.add_node({'class':'VectorAddNode','name':'n4.a-sum','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
        self.add_node({'class':'DotProductNode','name':'n4.a','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
        


      self.add_node({'class':'DataNode','name':'n12','opts':{'key':kwargs['mold']['key'],'img_shape':(76,39),'skip_grad': True,'a_func':{'name':'scalar','params':{'scalar':0.0000001}}}})
      
      self.add_node({'class':'FunctionNode','name':'n201','opts':{'a_func':{'name':'negative'},'fan_in':mem_cells,'fan_out':mem_cells}})
      
      self.add_node({'class':'HadamardNode','name':'n5','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      
      self.add_node({'class':'FunctionNode','name':'n7','opts':{'a_func':{'name':'sigmoid'},'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'DotProductNode','name':'n7-dot','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
       
      self.add_node({'class':'FunctionNode','name':'n8','opts':{'a_func':{'name':'tanh'},'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'VectorAddNode','name':'n18','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'VectorAddNode','name':'n21-add','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'VectorAddNode','name':'n22-add','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'VectorAddNode','name':'n31-add','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'VectorAddNode','name':'n7-add','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'HadamardNode','name':'n19','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'FunctionNode','name':'n21','opts':{'a_func':{'name':'sigmoid'},'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'DotProductNode','name':'n21-dot','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'FunctionNode','name':'n22','opts':{'a_func':{'name':'sigmoid'},'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'DotProductNode','name':'n22-dot','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'FunctionNode','name':'n31','opts':{'a_func':{'name':'sigmoid'},'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'DotProductNode','name':'n31-dot','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      
      self.add_node({'class':'VectorAddNode','name':'n23','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'ConvolutionNode','name':'n24','opts':{}})
      self.add_node({'class':'ConvolutionNode','name':'n24.a','opts':{}})
      self.add_node({'class':'ConvolutionNode','name':'n24.b','opts':{'no_padding':True}})

      self.add_node({'class':'FunctionNode','name':'n24.5','opts':{'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'n24.1','opts':{'dropout':False,'a_func':{'name':'tanh'},'img_shape':(7,7),'fan_in':100,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'n24.1.a','opts':{'dropout':False,'img_shape':(14,14),'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'n24.1.b','opts':{'reform':True,'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
      # self.add_node({'class':'VectorAddNode','name':'n24.4-sum','opts':{'fan_in':100,'fan_out':100}})
      self.add_node({'class':'DotProductNode','name':'n24.4','opts':{'fan_in':100,'fan_out':100}})
      
      self.add_node({'class':'PoolNode','name':'n24.2','opts':{'fan_in':100,'fan_out':100}})
      self.add_node({'class':'PoolNode','name':'n24.2.a','opts':{'fan_in':100,'fan_out':100}})
      self.add_node({'class':'PoolNode','name':'n24.2','opts':{'a_func':{'name':'relu'},'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'PoolNode','name':'n24.2.a','opts':{'a_func':{'name':'relu'},'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'DotProductNode','name':'n25','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      
      self.add_node({'class':'DotProductNode','name':'n26','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
    
      self.add_node({'class':'HadamardNode','name':'n20','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
     
      if kwargs['mold']['loss_gate']:
        self.add_children({'parent':'n1','children':['n2','n3']})
        self.add_children({'parent':'n3','children':['n4.a-sum']})
        self.add_children({'parent':'n4.a-sum','children':['n4.a','n25.5',['lstm_origin','n4.a-ones']]})
        self.add_children({'parent':'n4.a','children':['n3.a',['lstm_origin','n6.a']]})
        self.add_children({'parent':'n3.a','children':['n4-sum']})
        self.add_children({'parent':'n4-sum','children':['n4',['lstm_origin','n4-ones']]})
        self.add_children({'parent':'n4','children':['n5',['lstm_origin','n6']]})

      self.add_children({'parent':'n5','children':['n7','n8']})

      self.add_children({'parent':'n8','children':['n18']})

      self.add_children({'parent':'n18','children':['n19','n201']})

      self.add_children({'parent':'n201','children':['n20']})  

      self.add_children({'parent':'n19','children':['n21','n22']})

      self.add_children({'parent':'n20','children':['n31',[previous_graph,'n18']]})
      
      self.add_children({'parent':'n7','children':['n7-add']})
      self.add_children({'parent':'n21','children':['n21-add']})
      self.add_children({'parent':'n22','children':['n22-add']})
      self.add_children({'parent':'n31','children':['n31-add']})

      self.add_children({'parent':'n7-add','children':['n7-dot',['lstm_origin','n7-ones']]})
      self.add_children({'parent':'n21-add','children':['n21-dot',['lstm_origin','n21-ones']]})
      self.add_children({'parent':'n22-add','children':['n22-dot',['lstm_origin','n22-ones']]})
      self.add_children({'parent':'n31-add','children':['n31-dot',['lstm_origin','n31-ones']]})

      self.add_children({'parent':'n21-dot','children':['n23',['lstm_origin','n21-weight']]})
      self.add_children({'parent':'n22-dot','children':['n23',['lstm_origin','n22-weight']]})
      self.add_children({'parent':'n31-dot','children':['n23',['lstm_origin','n31-weight']]})
      self.add_children({'parent':'n7-dot','children':['n23',['lstm_origin','n7-weight']]})

      self.add_children({'parent':'n23','children':['n24.5','n25','n26',['lstm_origin','n301']]})
     

      self.add_children({'parent':'n24.5','children':['n24.4']})
      # self.add_children({'parent':'n24.4-sum','children':['n24.4',['conv_origin','n24.4-ones']]})
      self.add_children({'parent':'n24.4','children':['n24.1.b',['lstm_origin','n24.3']]})
      self.add_children({'parent':'n24.1.b','children':['n24.b']})
      self.add_children({'parent':'n24.b','children':['n24.1',['lstm_origin','n27.b']]})
      self.add_children({'parent':'n24.1','children':['n24.2']})
      self.add_children({'parent':'n24.2','children':['n24']})
      self.add_children({'parent':'n24','children':['n24.1.a',['lstm_origin','n27.a']]})
      self.add_children({'parent':'n24.1.a','children':['n24.2.a']})
      self.add_children({'parent':'n24.2.a','children':['n24.a']})
      self.add_children({'parent':'n24.a','children':['n12',['lstm_origin','n27']]})

     
      self.add_children({'parent':'n25','children':[[previous_graph,'n18'],['lstm_origin','n28']]})      
      self.add_children({'parent':'n26','children':[[previous_graph,'n5'],['lstm_origin','n29']]})
  
    elif kwargs['mold']['type'] == 'lstm_word_origin':
      x_dim = 100
      mem_cells = 1024  
      # this node will be empty becuase it has no children. When this happens it should be initialized with a senesible value
      self.add_node({'class':'OnesNode','name':'n5','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':1.1}})

      self.add_node({'class':'OnesNode','name':'n18','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':1.1}})

      # self.add_node({'class':'OnesNode','name':'n401','opts':{'dynamic_sizing':True,'fan_in':1,'fan_out':2,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'init_scaler':0.01}})
      self.add_node({'class':'OnesNode','name':'n301','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':0.1}})
      self.add_node({'class':'OnesNode','name':'n2301','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':0.1}})
      self.add_node({'class':'OnesNode','name':'n3701','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':0.1}})
      self.add_node({'class':'OnesNode','name':'n8001','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':0.1}})
      self.add_node({'class':'OnesNode','name':'n4.6.1','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':2.1}})
      # self.add_node({'class':'OnesNode','name':'n801','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train'}})
      #replace with a lookup
      # self.add_node({'class':'OnesNode','name':'final-bias-weights','opts':{'fan_in':1,'fan_out':2400,'weight_update_status': False,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train'}})
      

      self.add_node({'class':'WeightNode','name':'n6','opts':{'word_weight_mapper':True,'fan_in':mem_cells,'fan_out':2400,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.0,'alpha':0.005,'init_scaler':1.1}})

      self.add_node({'class':'WeightNode','name':'n13','opts':{'dropout':False,'fan_in':x_dim,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.9,'alpha':0.005,'init_scaler':3.1}})

      self.add_node({'class':'WeightNode','name':'n14','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.9,'alpha':0.005,'init_scaler':3.1}})

      self.add_node({'class':'WeightNode','name':'n16','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.9,'alpha':0.005,'init_scaler':3.1}})

      self.add_node({'class':'WeightNode','name':'n27','opts':{'dropout':False,'fan_in':x_dim,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.9,'alpha':0.005,'init_scaler':3.1}})

      self.add_node({'class':'WeightNode','name':'n28','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.9,'alpha':0.005,'init_scaler':3.1}})

      self.add_node({'class':'WeightNode','name':'n29','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.9,'alpha':0.005,'init_scaler':3.1}})

      self.add_node({'class':'WeightNode','name':'n35','opts':{'dropout':False,'fan_in':x_dim,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.9,'alpha':0.005,'init_scaler':3.1}})

      self.add_node({'class':'WeightNode','name':'n36','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.9,'alpha':0.005,'init_scaler':3.1}})

      self.add_node({'class':'WeightNode','name':'n41','opts':{'dropout':False,'fan_in':x_dim,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.9,'alpha':0.005,'init_scaler':3.1}})

      self.add_node({'class':'WeightNode','name':'n42','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.9,'alpha':0.005,'init_scaler':3.1}})

      self.add_node({'class':'WeightNode','name':'n43','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.9,'alpha':0.005,'init_scaler':3.1}})

    elif kwargs['mold']['type'] == 'lstm_word':

      link_graph = kwargs['mold']['link_graph']

      graph_index = kwargs['mold']['graph_index']
      if graph_index == 0:
        previous_graph = "lstm_word_origin"

      else:
        previous_graph = "lstm_word" + "_" + str(graph_index-1)
       
      

      mem_cells = 1024   

      self.add_node({'class':'DataNode','name':'n12','opts':{'key':kwargs['mold']['key'],'skip_grad': False,'weight_update_status': True,'weight_decay':0.01,'momentum':0.5,'alpha':0.005}})

      self.add_node({'class':'LossNode','name': 'n1','opts':{'a_func':{'name':'square-error'},'fan_in':1,'fan_out':1}})

      self.add_node({'class':'TargetNode','name': 'n2','opts':{'fan_in':1,'fan_out':1,'y':kwargs['mold']['y']}})
      
      self.add_node({'class':'FunctionNode','name':'n3','opts':{'dropout':False,'a_func':{'name':'softmax'},'fan_in':2400,'fan_out':2400}})

     
      # self.add_node({'class':'VectorAddNode','name':'n501','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'VectorAddNode','name':'n4.6','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells}})
      # self.add_node({'class':'FunctionNode','name':'n4.0','opts':{'a_func':{'name':'tanh'},'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'DotProductNode','name':'n4.1','opts':{}})
      self.add_node({'class':'AttentionNode','name':'n4.2','opts':{'dropout':False,'a_func':{'name':'softmax'},'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'DotProductNode','name':'n4.3','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':100}})
      self.add_node({'class':'VectorsToMatrixNode','name':'n4.4','opts':{}})
      self.add_node({'class':'VectorsToMatrixNode','name':'n4.4.b','opts':{}})
      self.add_node({'class':'FunctionNode','name':'n201','opts':{'a_func':{'name':'negative'},'fan_in':mem_cells,'fan_out':mem_cells}})
      
      self.add_node({'class':'DotProductNode','name':'n4','opts':{'fan_in':mem_cells,'fan_out':100}})

      self.add_node({'class':'HadamardNode','name':'n5','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      #candidates options creates children relationships based on graph regular expression and node name
      # self.add_node({'class':'MaxSelectorNode','name':'n5-max','opts':{'candidates':['lstm_word','n5']},'method':'dotproduct'})

      self.add_node({'class':'FunctionNode','name':'n8','opts':{'a_func':{'name':'tanh'},'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'VectorAddNode','name':'n18','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'HadamardNode','name':'n19','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'FunctionNode','name':'n21','opts':{'a_func':{'name':'sigmoid'},'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'VectorAddNode','name':'n23','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'DotProductNode','name':'n24','opts':{'fan_in':100,'fan_out':mem_cells}})
      

      self.add_node({'class':'DotProductNode','name':'n25','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
    

      self.add_node({'class':'DotProductNode','name':'n26','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
    

      self.add_node({'class':'FunctionNode','name':'n22','opts':{'a_func':{'name':'tanh'},'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'VectorAddNode','name':'n30','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'DotProductNode','name':'n33','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      

      self.add_node({'class':'DotProductNode','name':'n34','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})


      self.add_node({'class':'HadamardNode','name':'n20','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'FunctionNode','name':'n31','opts':{'a_func':{'name':'sigmoid'},'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'VectorAddNode','name':'n37','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'DotProductNode','name':'n38','opts':{'fan_in':100,'fan_out':mem_cells}})
  

      self.add_node({'class':'DotProductNode','name':'n39','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})


      self.add_node({'class':'DotProductNode','name':'n40','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})


      self.add_node({'class':'FunctionNode','name':'n7','opts':{'a_func':{'name':'sigmoid'},'fan_in':mem_cells,'fan_out':mem_cells}})
      
      self.add_node({'class':'VectorAddNode','name':'n80','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'DotProductNode','name':'n9','opts':{'fan_in':100,'fan_out':mem_cells}})
   

      self.add_node({'class':'DotProductNode','name':'n10','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})


      self.add_node({'class':'DotProductNode','name':'n11','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})


      self.add_children({'parent':'n1','children':['n2','n3']})

      # self.add_children({'parent':'n3','children':['n4']})
      self.add_children({'parent':'n3','children':['n4']})
    

      self.add_children({'parent':'n4','children':['n4.6',['lstm_word_origin','n6']]})
      self.add_children({'parent':'n4.6','children':['n4.1',['lstm_word_origin','n4.6.1']]})
      # self.add_children({'parent':'n4.0','children':['n4.3']})
      self.add_children({'parent':'n4.1','children':['n4.2','n4.4.b']})
      self.add_children({'parent':'n4.2','children':['n4.3']})
      self.add_children({'parent':'n4.3','children':['n5','n4.4']})

      temp_child = []
      temp_child_1 = []
      for graph_name, graph in self.master_graph.graph.iteritems():
        if graph.type == 'audio-frame':
          temp_child.append([graph_name,'n5'])
          temp_child_1.append([graph_name,'n24.5'])
     
      self.add_children({'parent':'n4.4','children':temp_child})
      self.add_children({'parent':'n4.4.b','children':temp_child_1})
     
      self.add_children({'parent':'n5','children':['n7','n8']})

      self.add_children({'parent':'n8','children':['n18']})

      self.add_children({'parent':'n18','children':['n19','n201']})

      self.add_children({'parent':'n201','children':['n20']})  

      self.add_children({'parent':'n19','children':['n21','n22']})

      self.add_children({'parent':'n21','children':['n23']})

      self.add_children({'parent':'n23','children':['n24','n25','n26',['lstm_word_origin','n2301']]})

      self.add_children({'parent':'n24','children':['n12',['lstm_word_origin','n27']]})

      self.add_children({'parent':'n25','children':[[previous_graph,'n18'],['lstm_word_origin','n28']]})      
      
      if graph_index == 0:
        self.add_children({'parent':'n26','children':[[previous_graph,'n5'],['lstm_word_origin','n29']]})   
        self.add_children({'parent':'n34','children':[[previous_graph,'n5'],['lstm_word_origin','n36']]})
        self.add_children({'parent':'n40','children':[[previous_graph,'n5'],['lstm_word_origin','n43']]})
        self.add_children({'parent':'n11','children':[[previous_graph,'n5'],['lstm_word_origin','n16']]})  
      else:
        self.add_children({'parent':'n26','children':[[previous_graph,'n4.6'],['lstm_word_origin','n29']]})   
        self.add_children({'parent':'n34','children':[[previous_graph,'n4.6'],['lstm_word_origin','n36']]})
        self.add_children({'parent':'n40','children':[[previous_graph,'n4.6'],['lstm_word_origin','n43']]})
        self.add_children({'parent':'n11','children':[[previous_graph,'n4.6'],['lstm_word_origin','n16']]})  

     

      self.add_children({'parent':'n22','children':['n30']})

      self.add_children({'parent':'n30','children':['n33','n34',['lstm_word_origin','n301']]})

      self.add_children({'parent':'n33','children':['n12',['lstm_word_origin','n35']]})


      self.add_children({'parent':'n20','children':['n31',[previous_graph,'n18']]})

      self.add_children({'parent':'n31','children':['n37']})

      self.add_children({'parent':'n37','children':['n38','n39','n40',['lstm_word_origin','n3701']]})

      self.add_children({'parent':'n38','children':['n12',['lstm_word_origin','n41']]})

      self.add_children({'parent':'n39','children':[[previous_graph,'n18'],['lstm_word_origin','n42']]})


      self.add_children({'parent':'n7','children':['n80']})

      self.add_children({'parent':'n80','children':['n9','n10','n11',['lstm_word_origin','n8001']]})

      self.add_children({'parent':'n9','children':['n12',['lstm_word_origin','n13']]})

      self.add_children({'parent':'n10','children':[[previous_graph,'n18'],['lstm_word_origin','n14']]})

    elif kwargs['mold']['type'] == 'gru_origin':
      x_dim = 2400
      mem_cells = 512  
      
      self.add_node({'class':'OnesNode','name':'n5','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':{'type':'random','scale':0.1}}})
      self.add_node({'class':'WeightNode','name':'n6.4.1','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':0.1}}})
      self.add_node({'class':'WeightNode','name':'n6.5.1','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':0.1}}})
      self.add_node({'class':'WeightNode','name':'n7.3.1','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':10.1}}})
      self.add_node({'class':'WeightNode','name':'n7.4.1','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':10.1}}})
      self.add_node({'class':'WeightNode','name':'n7.4.2.2.1.1','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':1.1}}})
      self.add_node({'class':'WeightNode','name':'n7.4.2.2.2.1','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':1.1}}})
     
    elif kwargs['mold']['type'] == 'gru':

      graph_index = kwargs['mold']['graph_index']
      # graph_level = kwargs['mold']['graph_level']
      mem_cells =  kwargs['mold']['size'] 
      link_node = kwargs['mold']['link_node']
      self.mold = kwargs['mold']
      
      if graph_index == 0:
        previous_graph = "gru_origin"
      else:
        previous_graph = "gru" + "_" + str(graph_index-1)

      self.add_node({'class':'FunctionNode','name':'n50','opts':{'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'VectorAddNode','name':'n5','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      
      self.add_node({'class':'HadamardNode','name':'n6','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'FunctionNode','name':'n6.1','opts':{'dropout':False,'a_func':{'name':'1_minus'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'n6.2','opts':{'dropout':False,'a_func':{'name':'sigmoid'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'VectorAddNode','name':'n6.3','opts':{}})
      self.add_node({'class':'DotProductNode','name':'n6.4','opts':{'add_bias':True,'fan_out':512}})
      self.add_node({'class':'DotProductNode','name':'n6.5','opts':{'add_bias':True,'fan_out':512}})

      self.add_node({'class':'HadamardNode','name':'n7','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'FunctionNode','name':'n7.1','opts':{'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'VectorAddNode','name':'n7.2','opts':{}})
      self.add_node({'class':'DotProductNode','name':'n7.3','opts':{'add_bias':True,'fan_out':512}})
      self.add_node({'class':'DotProductNode','name':'n7.4','opts':{'add_bias':True,'fan_out':512}})
      self.add_node({'class':'HadamardNode','name':'n7.4.2','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'FunctionNode','name':'n7.4.2.1','opts':{'dropout':False,'a_func':{'name':'sigmoid'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'VectorAddNode','name':'n7.4.2.2','opts':{}})
      self.add_node({'class':'DotProductNode','name':'n7.4.2.2.2','opts':{'add_bias':True,'fan_out':512}})
      self.add_node({'class':'DotProductNode','name':'n7.4.2.2.1','opts':{'add_bias':True,'fan_out':512}})

      self.add_children({'parent':'n50','children':['n5']})
      self.add_children({'parent':'n5','children':['n6','n7']})

      self.add_children({'parent':'n6','children':['n6.1',[previous_graph,'n5']]})
      self.add_children({'parent':'n6.1','children':['n6.2']})
      self.add_children({'parent':'n6.2','children':['n6.3']})
      self.add_children({'parent':'n6.3','children':['n6.4','n6.5']})
      self.add_children({'parent':'n6.4','children':[link_node,['gru_origin','n6.4.1']]})
      self.add_children({'parent':'n6.5','children':[['gru_origin','n6.5.1'],[previous_graph,'n5']]})
      
      self.add_children({'parent':'n7','children':['n7.1','n6.2']})
      self.add_children({'parent':'n7.1','children':['n7.2']})
      self.add_children({'parent':'n7.2','children':['n7.3','n7.4']})
      self.add_children({'parent':'n7.3','children':[link_node,['gru_origin','n7.3.1']]})
      self.add_children({'parent':'n7.4','children':['n7.4.2',['gru_origin','n7.4.1']]})
      self.add_children({'parent':'n7.4.2','children':[[previous_graph,'n5'],'n7.4.2.1']})
      self.add_children({'parent':'n7.4.2.1','children':['n7.4.2.2']})
      self.add_children({'parent':'n7.4.2.2','children':['n7.4.2.2.1','n7.4.2.2.2']})
      self.add_children({'parent':'n7.4.2.2.2','children':[['gru_origin','n7.4.2.2.2.1'],[previous_graph,'n5']]})
      self.add_children({'parent':'n7.4.2.2.1','children':[['gru_origin','n7.4.2.2.1.1'],link_node]})

      self.add_batch_normalization_sub_nets()
      
    elif kwargs['mold']['type'] == 'gru_word_origin':
      x_dim = 512
      mem_cells = 512  
     
      self.add_node({'class':'WeightNode','name':'n6','opts':{'word_weight_mapper':True,'fan_in':mem_cells,'fan_out':2400,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.0,'alpha':0.005,'init_scaler':{'type':'random','scale':0.1}}})     
      self.add_node({'class':'OnesNode','name':'n5','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':1.1,'mode':'train','init_scaler':{'type':'random','scale':0.1}}})
      self.add_node({'class':'WeightNode','name':'n6.4.1','opts':{'dropout':False,'fan_in':x_dim,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':1.1}}})
      self.add_node({'class':'WeightNode','name':'n6.5.1','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':1.1}}})
      self.add_node({'class':'WeightNode','name':'n7.3.1','opts':{'dropout':False,'fan_in':x_dim,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':10.1}}})
      self.add_node({'class':'WeightNode','name':'n7.4.1','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':10.1}}})
      self.add_node({'class':'WeightNode','name':'n7.4.2.2.1.1','opts':{'dropout':False,'fan_in':x_dim,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':1.1}}})
      self.add_node({'class':'WeightNode','name':'n7.4.2.2.2.1','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':{'type':'random','scale':1.1}}})
      
    elif kwargs['mold']['type'] == 'gru_word':
      self.mold = kwargs['mold']
      mem_cells = kwargs['mold']['size']  
      graph_index = kwargs['mold']['graph_index']
      data_object = kwargs['mold']['data_object']
      attention_block = kwargs['mold']['link_node']
      key = kwargs['mold']['key']
      
      if graph_index == 0:
        previous_graph = "gru_word_origin"
      else:
        previous_graph = "gru_word" + "_" + str(graph_index-1)
 
      self.add_node({'class':'FunctionNode','name':'n3','opts':{'a_func':{'name':'softmax'},'fan_in':2400,'fan_out':2400}})
      self.add_node({'class':'FunctionNode','name':'attention_scaler_func','opts':{'a_func':{'name':'tanh'},'fan_in':2400,'fan_out':2400}})
      self.add_node({'class':'DotProductNode','name':'top-dot','opts':{'debug':'activation','dropout':False,'add_bias':True,'fan_out':29}})      
      self.add_node({'class':'DataNode','name':'data','opts':{'key':key,'data_object':data_object,'weight_update_status':True,'a_func':{'name':'scalar','params':{'scalar':1.0}}}})
      self.add_node({'class':'VectorAddNode','name':'n5','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'HadamardNode','name':'n6','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'FunctionNode','name':'n6.1','opts':{'dropout':False,'a_func':{'name':'1_minus'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'n6.2','opts':{'dropout':False,'a_func':{'name':'sigmoid'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'VectorAddNode','name':'n6.3','opts':{}})
      self.add_node({'class':'DotProductNode','name':'n6.4','opts':{'add_bias':True,'fan_out':512}})
      self.add_node({'class':'DotProductNode','name':'n6.5','opts':{'add_bias':True,'fan_out':512}})

      self.add_node({'class':'HadamardNode','name':'n7','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'FunctionNode','name':'n7.1','opts':{'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'VectorAddNode','name':'n7.2','opts':{}})
      self.add_node({'class':'DotProductNode','name':'n7.3','opts':{'add_bias':True,'fan_out':512}})
      self.add_node({'class':'DotProductNode','name':'n7.4','opts':{'add_bias':True,'fan_out':512}})
      self.add_node({'class':'HadamardNode','name':'n7.4.2','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'FunctionNode','name':'n7.4.2.1','opts':{'dropout':False,'a_func':{'name':'sigmoid'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'VectorAddNode','name':'n7.4.2.2','opts':{}})
      self.add_node({'class':'DotProductNode','name':'n7.4.2.2.1','opts':{'add_bias':True,'fan_out':512}})
      self.add_node({'class':'DotProductNode','name':'n7.4.2.2.2','opts':{'add_bias':True,'fan_out':512}})

      # self.add_node({'class':'CosineSimNode','name':'attention_sim','opts':{}})
      self.add_node({'class':'DotProductNode','name':'attention_sim','opts':{}})
      self.add_node({'class':'DotProductNode','name':'attention','opts':{}})
      self.add_node({'class':'TransposeNode','name':'transpose','opts':{}})
      self.add_node({'class':'FunctionNode','name':'attention_func','opts':{'a_func':{'name':'softmax'}}})

      self.add_children({'parent':'n5','children':['n6','n7']})

      self.add_children({'parent':'n6','children':['n6.1',[previous_graph,'n5']]})
      self.add_children({'parent':'n6.1','children':['n6.2']})
      self.add_children({'parent':'n6.2','children':['n6.3']})
      self.add_children({'parent':'n6.3','children':['n6.4','n6.5']})
      
      self.add_children({'parent':'n6.4','children':['data',['gru_word_origin','n6.4.1']]})
      self.add_children({'parent':'n6.5','children':[['gru_word_origin','n6.5.1'],[previous_graph,'n5']]})

      self.add_children({'parent':'n7','children':['n7.1','n6.2']})
      self.add_children({'parent':'n7.1','children':['n7.2']})
      self.add_children({'parent':'n7.2','children':['n7.3','n7.4']})
      self.add_children({'parent':'n7.3','children':['data',['gru_word_origin','n7.3.1']]})
      self.add_children({'parent':'n7.4','children':['n7.4.2',['gru_word_origin','n7.4.1']]})
      self.add_children({'parent':'n7.4.2','children':[[previous_graph,'n5'],'n7.4.2.1']})
      self.add_children({'parent':'n7.4.2.1','children':['n7.4.2.2']})
      self.add_children({'parent':'n7.4.2.2','children':['n7.4.2.2.1','n7.4.2.2.2']})
      self.add_children({'parent':'n7.4.2.2.2','children':[['gru_word_origin','n7.4.2.2.2.1'],[previous_graph,'n5']]})
      self.add_children({'parent':'n7.4.2.2.1','children':[['gru_word_origin','n7.4.2.2.1.1'],'data']})
     
      self.add_children({'parent':'n3','children':['top-dot']})
      # self.add_children({'parent':'char-bias-add','children':['top-dot',['gru_word_origin','char-bias']]})
      self.add_children({'parent':'top-dot','children':['attention_scaler_func',['gru_word_origin','n6']]})
      self.add_children({'parent':'attention_scaler_func','children':['attention']})
      self.add_children({'parent':'attention','children':['transpose',attention_block]})
      self.add_children({'parent':'transpose','children':['attention_func']})
      self.add_children({'parent':'attention_func','children':['attention_sim']})
      self.add_children({'parent':'attention_sim','children':['n5',attention_block]})

      self.add_batch_normalization_sub_nets()       

    elif kwargs['mold']['type'] == 'row_conv_origin':
      mem_cells = 1024  
     
      # self.add_node({'class':'OnesNode','name':'gate-left','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':1.1}})
      # self.add_node({'class':'OnesNode','name':'gate-right','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':1.1}})


      # self.add_node({'class':'OnesNode','name':'ones-left','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':0.0}})
      # self.add_node({'class':'OnesNode','name':'ones-right','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':0.0}})
      # self.add_node({'class':'OnesNode','name':'ones-up','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':0.0}})
    
      
      self.add_node({'class':'WeightNode','name':'w-left','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
      self.add_node({'class':'WeightNode','name':'w-up','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
      self.add_node({'class':'WeightNode','name':'w-right','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
      self.add_node({'class':'WeightNode','name':'w-output','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':1.1}})
       
    elif kwargs['mold']['type'] == 'row_conv':

      mem_cells = 1024

      if kwargs['mold']['graph_index'] == 0:
        left_graph = 'row_conv_origin'
      else:  
        left_graph = kwargs['mold']['bottom_graph'] + '_' + str(kwargs['mold']['graph_index'] - 1)

      if kwargs['mold']['last_graph']:  
        right_graph = 'row_conv_origin'
      else:
        right_graph = kwargs['mold']['bottom_graph'] + '_' + str(kwargs['mold']['graph_index'] +1)
      
      under_graph = kwargs['mold']['bottom_graph'] + '_' + str(kwargs['mold']['graph_index'])

      if kwargs['mold']['graph_index'] != 0 and kwargs['mold']['graph_level'] != 1  :
        self.add_node({'class':'DotProductNode','name':'dot-left','opts':{'fan_in':mem_cells,'fan_out':100}})
        self.add_node({'class':'FunctionNode','name':'func-left','opts':{'dropout':False,'a_func':{'name':'sigmoid'},'fan_in':100,'fan_out':100}})
        # self.add_node({'class':'VectorAddNode','name':'add-left','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
        self.add_node({'class':'HadamardNode','name':'gate-left','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      if kwargs['mold']['last_graph'] != True and kwargs['mold']['graph_level'] != 1:
        self.add_node({'class':'DotProductNode','name':'dot-right','opts':{'fan_in':mem_cells,'fan_out':100}})
        self.add_node({'class':'FunctionNode','name':'func-right','opts':{'dropout':False,'a_func':{'name':'sigmoid'},'fan_in':100,'fan_out':100}})
        # self.add_node({'class':'VectorAddNode','name':'add-right','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
        self.add_node({'class':'HadamardNode','name':'gate-right','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      

      self.add_node({'class':'HadamardNode','name':'gate-up','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'DotProductNode','name':'dot-up','opts':{'fan_in':mem_cells,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'func-up','opts':{'dropout':False,'a_func':{'name':'sigmoid'},'fan_in':100,'fan_out':100}})
      # self.add_node({'class':'VectorAddNode','name':'add-up','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      
      self.add_node({'class':'DotProductNode','name':'dot-output','opts':{'fan_in':mem_cells,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'func-output','opts':{'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'VectorAddNode','name':'add-combiner','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      
      if kwargs['mold']['graph_index'] != 0 and kwargs['mold']['graph_level'] != 1:
        self.add_children({'parent':'gate-left','children':['func-left','func-output']})
        self.add_children({'parent':'func-left','children':['dot-left']})
        # self.add_children({'parent':'add-left','children':['dot-left',['row_conv_origin','ones-left']]})
        self.add_children({'parent':'dot-left','children':[['row_conv_origin','w-left'],'add-combiner']})
       

      if kwargs['mold']['last_graph'] != True and kwargs['mold']['graph_level'] != 1:
        self.add_children({'parent':'gate-right','children':['func-right','func-output']})
        self.add_children({'parent':'func-right','children':['dot-right']})
        # self.add_children({'parent':'add-right','children':['dot-right',['row_conv_origin','ones-right']]})
        self.add_children({'parent':'dot-right','children':[['row_conv_origin','w-right'],'add-combiner']})
       

      self.add_children({'parent':'gate-up','children':['func-up','func-output']})
      self.add_children({'parent':'func-up','children':['dot-up']})
      self.add_children({'parent':'func-output','children':['dot-output']})

      g_i = kwargs['mold']['graph_index']
      if kwargs['mold']['graph_level'] == 0:
        if g_i == 0:
          self.add_children({'parent':'add-combiner','children':[  [kwargs['mold']['bottom_graph'],'slicer-'+str(g_i+1)] , [kwargs['mold']['bottom_graph'],'slicer-'+str(g_i)] ]})
        elif kwargs['mold']['last_graph']:
          self.add_children({'parent':'add-combiner','children':[  [kwargs['mold']['bottom_graph'],'slicer-'+str(g_i)] ,  [kwargs['mold']['bottom_graph'],'slicer-'+str(g_i-1)]  ]})
        else:
          self.add_children({'parent':'add-combiner','children':[  [kwargs['mold']['bottom_graph'],'slicer-'+str(g_i)] ,  [kwargs['mold']['bottom_graph'],'slicer-'+str(g_i+1)] , [kwargs['mold']['bottom_graph'],'slicer-'+str(g_i-1)]  ]})     
        # self.add_children({'parent':'add-combiner','children':[  [kwargs['mold']['bottom_graph'],'slicer-'+str(g_i)] ]})     
      else:
        if g_i == 0:
          self.add_children({'parent':'add-combiner','children':[  [right_graph,'gate-left'] , [under_graph,'gate-up' ]]})
        elif kwargs['mold']['last_graph']:
          self.add_children({'parent':'add-combiner','children':[  [under_graph,'gate-up'] ,  [left_graph,'gate-right']  ]})
        else:
          self.add_children({'parent':'add-combiner','children':[  [under_graph,'gate-up'] ,  [right_graph,'gate-left'] , [left_graph,'gate-right']  ]})
          # self.add_children({'parent':'add-combiner','children':[  [under_graph,'gate-up']   ]})
        
 
      # self.add_children({'parent':'add-up','children':['dot-up',['row_conv_origin','ones-up']]})
      self.add_children({'parent':'dot-up','children':[['row_conv_origin','w-up'],'add-combiner']})
      self.add_children({'parent':'dot-output','children':[['row_conv_origin','w-output'],'add-combiner']})
  
    elif kwargs['mold']['type'] == 'row_conv_classifier_origin':
      mem_cells = 1024
      self.add_node({'class':'OnesNode','name':'gate-left','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':0.0}})
      self.add_node({'class':'OnesNode','name':'gate-right','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'mode':'train','init_scaler':0.0}})
      
      self.add_node({'class':'WeightNode','name':'w-classify','opts':{'word_weight_mapper':True,'fan_in':mem_cells,'fan_out':2400,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.0,'alpha':0.005,'init_scaler':1.1}})
      # self.add_node({'class':'OnesNode','name':'ones-classify','opts':{'dynamic_sizing':True,'fan_in':1,'fan_out':1,'weight_update_status': True,'weight_decay':0.0,'momentum':0.1,'alpha':0.005,'init_scaler':0.0}})
  
    elif kwargs['mold']['type'] == 'row_conv_classifier':
      mem_cells = 1024

      if kwargs['mold']['graph_index'] == 0:
        left_graph = 'row_conv_classifier_origin'
      else:  
        left_graph = kwargs['mold']['bottom_graph'] + '_' + str(kwargs['mold']['graph_index'] - 1)

      if kwargs['mold']['last_graph']:  
        right_graph = 'row_conv_classifier_origin'
      else:
        right_graph = kwargs['mold']['bottom_graph'] + '_' + str(kwargs['mold']['graph_index'] +1)
      
      under_graph = kwargs['mold']['bottom_graph'] + '_' + str(kwargs['mold']['graph_index'])



      self.add_node({'class':'LossNode','name':'loss','opts':{'a_func':{'name':'softmax_loss'},'fan_in':1,'fan_out':1}})
      self.add_node({'class':'TargetNode','name':'target','opts':{'fan_in':1,'fan_out':1,'y':kwargs['mold']['y']}})
      self.add_node({'class':'DotProductNode','name':'dot-classify','opts':{'fan_in':mem_cells,'fan_out':100}})
      self.add_node({'class':'VectorAddNode','name':'add-combiner','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      # self.add_node({'class':'VectorAddNode','name':'add-classifier','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'FunctionNode','name':'softmax','opts':{'dropout':False,'a_func':{'name':'sigmoid'},'fan_in':100,'fan_out':100}})

      self.add_children({'parent':'loss','children':['target','softmax']})
      self.add_children({'parent':'softmax','children':['dot-classify']})
      # self.add_children({'parent':'add-classifier','children':['dot-classify',['row_conv_classifier_origin','ones-classify']]})
      self.add_children({'parent':'dot-classify','children':['add-combiner',['row_conv_classifier_origin','w-classify']]})


      self.add_children({'parent':'add-combiner','children':[ [left_graph,'gate-right'] , [right_graph,'gate-left'] , [under_graph,'gate-up'] ]})
      # self.add_children({'parent':'add-combiner','children':[ [left_graph,'gate-up'] , [right_graph,'gate-up'] , [under_graph,'gate-up'] ]})
      # self.add_children({'parent':'add-combiner','children':[ [under_graph,'gate-up'] ]})

    elif kwargs['mold']['type'] == 'flow_classifier_origin':
      mem_cells = 1024
          
      self.add_node({'class':'WeightNode','name':'w-classify','opts':{'word_weight_mapper':True,'fan_in':28,'fan_out':28,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.0,'alpha':0.005,'init_scaler':1.1}})
      self.add_node({'class':'WeightNode','name':'w-classify-0','opts':{'fan_in':mem_cells,'fan_out':28,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.0,'alpha':0.005,'init_scaler':0.1}})

    elif kwargs['mold']['type'] == 'flow_classifier':
      mem_cells = 1024

      left_graph = kwargs['mold']['bottom_graph'] + '_' + str(kwargs['mold']['graph_index'] )
      right_graph = kwargs['mold']['bottom_graph'] + '_' + str(kwargs['mold']['graph_index'] + 1 )
      
    
      self.add_node({'class':'LossNode','name':'loss','opts':{'a_func':{'name':'softmax_loss'},'fan_in':1,'fan_out':1}})
      self.add_node({'class':'TargetNode','name':'target','opts':{'fan_in':1,'fan_out':1,'y':kwargs['mold']['y']}})
      self.add_node({'class':'DotProductNode','name':'dot-classify','opts':{'fan_in':mem_cells,'fan_out':100}})
      self.add_node({'class':'DotProductNode','name':'dot-classify-0','opts':{'fan_in':mem_cells,'fan_out':100}})
      self.add_node({'class':'VectorAddNode','name':'add-combiner','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'FunctionNode','name':'softmax','opts':{'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'sinc','opts':{'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})

      self.add_children({'parent':'loss','children':['target','softmax']})
      self.add_children({'parent':'softmax','children':['dot-classify']})

      self.add_children({'parent':'dot-classify','children':['sinc',['flow_classifier_origin','w-classify']]})
      self.add_children({'parent':'sinc','children':['dot-classify-0']})
      self.add_children({'parent':'dot-classify-0','children':['add-combiner',['flow_classifier_origin','w-classify-0']]})
      self.add_children({'parent':'add-combiner','children':[ [left_graph,'add-top'],[right_graph,'add-top']  ]})

    elif kwargs['mold']['type'] == 'flow_origin':
      mem_cells = 1024  
      # self.add_node({'class':'OnesNode','name':'add-top','opts':{'fan_in':1,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.0,'alpha':0.005,'init_scaler':0.1}})
      self.add_node({'class':'WeightNode','name':'w-flow','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':0.1}})
      self.add_node({'class':'WeightNode','name':'w-delta','opts':{'dropout':False,'fan_in':mem_cells,'fan_out':mem_cells,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':0.1}})
       
    elif kwargs['mold']['type'] == 'flow':

      mem_cells = 1024
      origin_graph = "flow" +"_" + str(kwargs['mold']['graph_level'] + 1) + "_" + "origin"
      left_graph = kwargs['mold']['bottom_graph'] + '_' + str(kwargs['mold']['graph_index'] )
      right_graph = kwargs['mold']['bottom_graph'] + '_' + str(kwargs['mold']['graph_index'] +1)
      
      if kwargs['mold']['graph_index'] !=0:
        ary = self.name.split('_')
        ary[-1] = str(int(ary[-1]) -1)
        left_sibling_graph = "_".join(ary)
      else:  
        left_sibling_graph = origin_graph
      
      self.add_node({'class':'DotProductNode','name':'dot-flow','opts':{'fan_in':mem_cells,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'func-flow','opts':{'dropout':False,'a_func':{'name':'sigmoid'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'HadamardNode','name':'gate-flow','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})

      self.add_node({'class':'DotProductNode','name':'dot-delta','opts':{'fan_in':mem_cells,'fan_out':100}})
      self.add_node({'class':'FunctionNode','name':'func-delta','opts':{'dropout':False,'a_func':{'name':'tanh'},'fan_in':100,'fan_out':100}})
       
      self.add_node({'class':'VectorAddNode','name':'add-top','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      self.add_node({'class':'VectorAddNode','name':'add-combiner','opts':{'fan_in':mem_cells,'fan_out':mem_cells}})
      
      self.add_children({'parent':'gate-flow','children':['func-flow','add-combiner']})
      self.add_children({'parent':'func-flow','children':['dot-flow']})
      self.add_children({'parent':'dot-flow','children':[[origin_graph,'w-flow'],'add-combiner']})
     

      self.add_children({'parent':'add-top','children':['gate-flow','func-delta']})
      self.add_children({'parent':'func-delta','children':['dot-delta']})
      self.add_children({'parent':'dot-delta','children':['add-combiner',[origin_graph,'w-delta']]})
      
      g_i = kwargs['mold']['graph_index']
      if kwargs['mold']['graph_level'] == 0:
        self.add_children({'parent':'add-combiner','children':[  [kwargs['mold']['bottom_graph'],'slicer-'+str(g_i)] ,  [kwargs['mold']['bottom_graph'],'slicer-'+str(g_i+1)] ]})     
      else:
        self.add_children({'parent':'add-combiner','children':[  [right_graph,'add-top'] , [left_graph,'add-top'] ]})
          
    elif kwargs['mold']['type'] == 'second_order_origin':
 
      self.add_node({'class':'WeightNode','name':'w-1','opts':{'dropout':False,'fan_in':0,'fan_out':3,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':0.1}})
      self.add_node({'class':'WeightNode','name':'w-3','opts':{'dropout':False,'fan_in':3,'fan_out':3,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':0.1}})
      self.add_node({'class':'WeightNode','name':'w-2','opts':{'dropout':False,'fan_in':3,'fan_out':3,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':0.1}})
      self.add_node({'class':'WeightNode','name':'w-4','opts':{'dropout':False,'fan_in':3,'fan_out':1,'weight_update_status': True,'weight_decay':0.9999,'momentum':0.2,'alpha':0.005,'init_scaler':0.1}})
     
    elif kwargs['mold']['type'] == 'second_order':

      self.add_node({'class':'LossNode','name':'loss','opts':{'type':'square_error'}})
      self.add_node({'class':'TargetNode','name':'target','opts':{'fan_in':1,'fan_out':1,'y':kwargs['mold']['y']}})
      self.add_node({'class':'FunctionNode','name':'tanh','opts':{'dropout':False,'a_func':{'name':'tanh'} } })
      self.add_node({'class':'FunctionNode','name':'tanh_2','opts':{'dropout':False,'a_func':{'name':'sigmoid'} } })
      self.add_node({'class':'FunctionNode','name':'tanh_3','opts':{'dropout':False,'a_func':{'name':'tanh'} } })
      self.add_node({'class':'FunctionNode','name':'gaussian','opts':{'type':'response_surface','a_func':{'name':'sigmoid'} } })
      self.add_node({'class':'DotProductNode','name':'dot-1','opts':{}})
      self.add_node({'class':'DotProductNode','name':'dot-2','opts':{}})
      self.add_node({'class':'DotProductNode','name':'dot-3','opts':{}})
      self.add_node({'class':'DotProductNode','name':'dot-4','opts':{}})
      self.add_node({'class':'DataNode','name':'data_node','opts':{'key':kwargs['mold']['key']}}) 
      
      self.add_children({'parent':'loss','children':['gaussian','target']})
      self.add_children({'parent':'gaussian','children':['dot-4']})
      self.add_children({'parent':'dot-4','children':[['second_order_origin','w-4'],'tanh_3']})
      self.add_children({'parent':'tanh_3','children':['dot-2']})
      self.add_children({'parent':'dot-2','children':[['second_order_origin','w-2'],'tanh_2']})
      self.add_children({'parent':'tanh_2','children':['dot-3']})
      self.add_children({'parent':'dot-3','children':[['second_order_origin','w-3'],'tanh']})
      self.add_children({'parent':'tanh','children':['dot-1']})
      self.add_children({'parent':'dot-1','children':[['second_order_origin','w-1'],'data_node']})

       

    elif kwargs['mold']['type'] == 'test_origin':
      self.add_node({'class':'WeightNode','name':'n5','opts':{'fan_in':900,'fan_out':10,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.0,'alpha':0.05,'init_scaler':0.1}})
      self.add_node({'class':'WeightNode','name':'n10','opts':{'dropout':False,'fan_in':784,'fan_out':900,'weight_update_status': True,'weight_decay':0.99999,'momentum':0.2,'alpha':0.05,'init_scaler':4.1}}) 
      self.add_children({'parent':'n10','children':['n5']})
      self.add_children({'parent':'n5','children':[['test','n3']]})

    elif kwargs['mold']['type'] == 'test':

      self.add_node({'class':'LossNode','name': 'n1','opts':{'a_func':{'name':'classifier'},'fan_in':1,'fan_out':1}})
      self.add_node({'class':'TargetNode','name': 'n2','opts':{'fan_in':1,'fan_out':1,'y':''}})
      self.add_node({'class':'FunctionNode','name':'n3','opts':{'a_func':{'name':'softmax'},'fan_in':2400,'fan_out':2400}})
      self.add_node({'class':'DotProductNode','name':'n4','opts':{'fan_in':100,'fan_out':100}}) 
      self.add_node({'class':'DataNode','name':'n12','opts':{'key':'','img_shape':(28,28),'skip_grad': True,'a_func':{'name':'scalar','params':{'scalar':1.0}}}})
      self.add_node({'class':'FunctionNode','name':'n6','opts':{'dropout':True,'a_func':{'name':'sin'},'fan_in':100,'fan_out':100}})
      self.add_node({'class':'DotProductNode','name':'n7','opts':{'fan_in':100,'fan_out':100}})

      self.add_children({'parent':'n1','children':['n2','n3']})
      self.add_children({'parent':'n3','children':['n4']})
      self.add_children({'parent':'n4','children':['n6',['test_origin','n5']]})
      self.add_children({'parent':'n6','children':['n7']})
      self.add_children({'parent':'n7','children':['n12',['test_origin','n10']]})


