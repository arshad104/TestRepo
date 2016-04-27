import mongowrapper as mdb
import numpy as np
import zmq
import sys
import os

from IPython import embed

db = mdb.MongoWrapper(db_name='machine_learning')

os.environ['CUDARRAY_BACKEND'] = 'cuda'
import cudarray as ca

weights_dict = {}

### Reply Socket
context = zmq.Context()
rep_socket = context.socket(zmq.REP)
rep_socket.bind('tcp://127.0.0.1:5555')

### Pull messages
context = zmq.Context()
pull_socket = context.socket(zmq.PULL)
pull_socket.bind ('tcp://127.0.0.1:5556')

### Publish the message to all subscribers
context = zmq.Context()
pub_socket = context.socket(zmq.PUB)
pub_socket.bind('tcp://127.0.0.1:5557')

### Initialize poll set
poller = zmq.Poller()
poller.register(pull_socket, zmq.POLLIN)
poller.register(rep_socket, zmq.POLLIN)

def send_reply(flags=0, copy=True, track=False):
  message = rep_socket.recv_json()
  ### do your work here  
  A = intialize_weights(message)
  ### reply back
  md = dict(
    dtype = str(A.dtype),
    shape = A.shape
  )
  rep_socket.send_json(md, flags|zmq.SNDMORE)
  rep_socket.send(A, flags, copy=copy, track=track)

def recv_array(socket, flags=0, copy=True, track=False):
  md = socket.recv_json(flags=flags)
  msg = socket.recv(flags=flags, copy=copy, track=track)
  buf = buffer(msg)
  A = np.frombuffer(buf, dtype=md['dtype'])
  A = A.reshape(md['shape'])
  key = md['graph']+'_'+md['node']
  weights_dict[key] -= A
  return weights_dict[key]  

def publish_message(flags=0, copy=True, track=False):
  #print "publishing weights"
  for k, v in weights_dict.iteritems():
    md = dict(
      dtype = str(v.dtype),
      shape = v.shape,
      key = k
    )
    pub_socket.send_json(md, flags|zmq.SNDMORE)
    pub_socket.send(v, flags, copy=copy, track=track)

def intialize_weights(dictio):
  key = dictio['graph']+'_'+dictio['node']
  ### check if weights are in cache
  if key in weights_dict:
    print "loading from cache"
    return weights_dict[key]
  else:
    print 'initializing ---> ', key
    ### check weights in db, load if exists
    obj = load_weights(dictio['fork_name'], dictio['graph'], dictio['node'])
    if obj is not None:
      a = np.array(obj['data'])
    
    ### elseif node is filternode create random array
    elif dictio['n_class'] == "FilterNode":
      a = np.random.normal(size=(dictio['n_filters'], dictio['n_channels'], dictio['filter_h'], dictio['filter_w'])) * dictio['init_factor']
    
    ### else node is weightnode create random array
    else:
      if dictio['scalar_type'] == 'ones':
        a = np.ones((dictio['fan_in'],dictio['fan_out'])) * dictio['scale']
      else:
        ### if matrix is square then get its single value precision
        if dictio['fan_in'] == dictio['fan_out']:
          rand_mat = np.random.normal(size=(dictio['fan_in'],dictio['fan_out']))
          u, s, v = np.linalg.svd(rand_mat)
          a = np.array(u)
        else:  
          init_factor = dictio['scale'] / ((dictio['fan_out']) ** (0.5))
          a = np.random.normal(size=(dictio['fan_in'],dictio['fan_out'])) * init_factor
    ### save weights to cache
    weights_dict[key] = a
    ### return weights mat
    return a

def load_weights(model_name, graph, node):
  model_obj = None
  query={'model_name': model_name}
  if db.is_exists(collection_name='model', query=query):
    inner_query = {'graph': graph, 'node': node}
    model_obj = db.load_sub_document(collection_name='model', query=query, subdoc='nodes_weight', inner_query=inner_query)
  return model_obj

should_continue = True
while should_continue:
  socks = dict(poller.poll(1000))
  if pull_socket in socks and socks[pull_socket] == zmq.POLLIN:
    nparr = recv_array(pull_socket)
    publish_message()
  if rep_socket in socks and socks[rep_socket] == zmq.POLLIN:
    send_reply()
