import sys

sys.path.append('../..')

import numpy as np
from random import shuffle
from IPython import embed
import core.engine.nn_functions as nn_func
from core.engine.beam import *
from core.engine.wordvec_server import WordvecServer
from core.engine.graph import *
from core.engine.neural_nodes import *
from core.engine.containers import *
from data_iterators.audio_server_2 import AudioServer2
from core.engine.word_weight_mapper import WordWeightMapper
import cProfile, pstats, io
import itertools
import random
import json

os.environ['CUDARRAY_BACKEND'] = 'cuda'
import cudarray as ca

class TrainingSession:
  def __init__(self,kwargs):
    self.iterations = self.arg_exists(kwargs,'iterations')
    self.log_error = self.arg_exists(kwargs,'log_error')
    self.log_grad = self.arg_exists(kwargs,'log_grad')
    self.log_activation = self.arg_exists(kwargs,'log_activation')
    self.batch_size = 1
    self.master_graph = MasterGraph({'batch_size':self.batch_size,'log_grad':self.log_grad,'log_activation':self.log_activation,'log_error':self.log_error})
    self.iteration_number = 0
    self.master_graph.counter = 0
    self.weight_persist_freq = self.arg_exists(kwargs,'weight_persist_freq')
    self.error_history = []
    self.classification_history = []
    self.process_id = os.getpid()
    self.normalizer = Normalizer()
    self.char_average = AverageTracker()
                                  
  def train(self):
    self.master_graph.observation_server = AudioServer2({})
    self.master_graph.word_weight_mapper = WordWeightMapper()
    self.master_graph.dt = WordvecServer({})

    # mold = {'name':'batch_normalize_origin','type': 'batch_normalize_origin','loss_gate':False}
    # g = Graph({'name':'batch_normalize_origin','master_graph':self.master_graph,'batch_size':1}).build_graph({'batch_size':self.batch_size,'name':mold['name'],'mold':mold})
    
    # mold = {'name':'gru_word_origin','type': 'gru_word_origin','loss_gate':False}
    # g = Graph({'name':'gru_word_origin','master_graph':self.master_graph,'batch_size':1}).build_graph({'batch_size':self.batch_size,'name':mold['name'],'mold':mold})

    # mold = {'name':'gru_origin','type': 'gru_origin','loss_gate':False}
    # g = Graph({'name':'gru_origin','master_graph':self.master_graph,'batch_size':1}).build_graph({'batch_size':self.batch_size,'name':mold['name'],'mold':mold})

    # mold = {'name':'conv_origin','type': 'conv_origin','size':100}
    # Graph({'name':'conv_origin','type':'conv_origin','master_graph':self.master_graph,'batch_size':1}).build_graph({'batch_size':self.batch_size,'master_graph':self.master_graph,'name':mold['name'],'mold':mold})


    mold = {'name':'test_origin','type': 'test_origin'}
    g = Graph({'name':'test_origin','master_graph':self.master_graph,'batch_size':1}).build_graph({'batch_size':self.batch_size,'name':mold['name'],'mold':mold})

    mold = {'name':'test','type': 'test'}
    g = Graph({'name':'test','master_graph':self.master_graph,'batch_size':1}).build_graph({'batch_size':self.batch_size,'name':mold['name'],'mold':mold})

    # i = 0
    # #request an observation from AudioServer, AudioServer makes the observation available with a reference
    # _iter = self.master_graph.observation_server.__iter__()
    
    # while i < self.iterations:

    #   self.master_graph.reset_except_origin()
    #   observation = _iter.next()
    #   self.master_graph.word_weight_mapper.words(observation['tokens'])
    #   seq_len = observation['audio'].shape[0]
    #   token_len = len(observation['tokens'])    
    #   print "Frames - " + str(seq_len)

      # for j in xrange(seq_len-1):
        
      #   image = observation['audio'][j]
      #   image = self.normalizer.normalize(image)
      #   image.shape = (1,1,image.shape[0],image.shape[1])

      #   name = 'conv' + "_" + str(j)
      
      #   mold = {'name':name,'type': 'conv', 'key':image,'size':1,'img_shape':image.shape}
      #   Graph({'name':name,'type':'conv','master_graph':self.master_graph,'batch_size':1}).build_graph({'batch_size':self.batch_size,'master_graph':self.master_graph,'name':mold['name'],'mold':mold})
         
      #   name = 'gru' + "_" + str(j)
       
      #   mold = {'name':name,'type': 'gru','size':512,'graph_index':j,'link_node':['conv' + "_" + str(j),'func-3'],'loss_gate':False}
      #   g = Graph({'name':name,'master_graph':self.master_graph,'batch_size':1})
      #   g.build_graph({'batch_size':self.batch_size,'master_graph':self.master_graph,'name':mold['name'],'mold':mold})

      # _children = [['gru_' + str(x),'n50'] for x in range(seq_len-1)]

      # mold = {'name':'attention','type': 'attention', '_children':_children,'size':100}
      # g = Graph({'name':'attention','master_graph':self.master_graph,'batch_size':1})
      # g.build_graph({'batch_size':self.batch_size,'master_graph':self.master_graph,'name':mold['name'],'mold':mold})

      # for k in xrange(token_len-2):

      #   y = observation['tokens'][k+1]

      #   temp_mat = np.zeros((1,self.master_graph.word_weight_mapper.weight_mat.shape[1]),dtype = np.float32)
      #   temp_mat[0,self.master_graph.word_weight_mapper.dict[y]] = 1.0
      #   # self.char_average.stack(temp_mat)
      #   # if i < 110:
      #   y = ca.array(temp_mat)
      #   # else:
      #     # ave = (self.char_average.average() == 0) + self.char_average.average()
      #     # y = ca.array(temp_mat / (ave))   
      #   x = [observation['k'],'tokens',k]

      #   size = 512
      #   name = 'gru_word' + "_" + str(k)
        
      #   mold = {'name':name,'type': 'gru_word', 'key':x,'size':size,'graph_index':k,'data_object':self.master_graph.dt,'link_node':['attention','vecs']}
      #   g = Graph({'name':name,'master_graph':self.master_graph,'batch_size':1})
      #   g.build_graph({'batch_size':self.batch_size,'master_graph':self.master_graph,'name':mold['name'],'mold':mold}) 
       
      #   name_2 = 'max_log_prob_loss' + "_" + str(k)

      #   size = 29
      #   input_node = [name,'n3']
        
      #   mold = {'name':name_2,'type': 'max_log_prob_loss','size':size,'graph_index':k,'y':y,'input_node':input_node,'seq_len':(token_len-1)}
      #   g = Graph({'name':name_2,'master_graph':self.master_graph,'batch_size':1})
      #   g.build_graph({'batch_size':self.batch_size,'master_graph':self.master_graph,'name':mold['name'],'mold':mold})
      #   if k != 0:
      #     self.master_graph.graph['max_log_prob_loss_0'].add_children({'parent':'loss','children':[[name_2,'loss-dot']]}) 
          
      
    self.master_graph.forward()
     
      # self.master_graph.backward()

      # self.master_graph.debug_grads('max_log_prob_loss_0','loss')  
        
      # self.master_graph.update_weights()

      # self.master_graph.graph['gru_word_origin'].nodes['n6'].activation_status = False
       
      # self.master_graph.print_error(i,'loss')

      # self.master_graph.reset_grads()

      # i += 1
      
  def arg_exists(self,dictio,arg):
    if arg in dictio: 
      return dictio[arg]
    else:
      return None
  
 
      

t = TrainingSession({'iterations':50000000,'time_frame':2,'batch_size':1,'log_grad':False,'log_activation':False,'log_error':True})


# pr = cProfile.Profile()
# pr.enable()


t.train()

# pr.disable()
# s = io.StringIO()
# sortby = 'cumulative'
# ps = pstats.Stats(pr, stream=s).sort_stats(sortby)

# ps.dump_stats('program.prof')

