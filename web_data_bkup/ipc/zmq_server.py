import mongowrapper as mdb
import numpy as np
import time
import json
import uuid
import zmq
import sys
import os

from subprocess import Popen
from IPython import embed
from common import *

OFFSET = 0

ipc_uuid = str(uuid.uuid1())
url = "python networking/interface.py " + ipc_uuid
# with open(os.devnull, "w") as DEVNULL:
#   sub_proc = Popen(url.split(), stdout=DEVNULL)
sub_proc = Popen(url.split())

tensors = {}
running_model = {}

### set context
context = zmq.Context()

### count other servers
print "Finding Peers..."
time.sleep(3)

def get_offset(socket):
  socket.send_json(dict(getOffset=True))
  return socket.recv()

def register_service(socket, s):
  socket.send_json(dict(registerService=s))
  msg = socket.recv()
  print msg

### PeerDiscovery communication
ipc = context.socket(zmq.REQ)
ipc.connect("ipc://" + ipc_uuid)
msg = get_offset(ipc)
OFFSET = int(msg)
print OFFSET
embed()
### Load balanced reply for sending tesnors
replier = context.socket(zmq.REP)
port = str(5555 + OFFSET)
replier.bind('tcp://*:' + port)
register_service(ipc, port + '-replier-paramserver')

### Subscriber for gradients to recv messages
subscriber = context.socket(zmq.SUB)
port = str(5555 + OFFSET + 1)
subscriber.bind('tcp://*:' + port)
register_service(ipc, port + '-subscriber-paramserver')

### One to one client-server pairs
pair = context.socket(zmq.REP)
port = str(5555 + OFFSET + 2)
pair.bind('tcp://*:' + port)
register_service(ipc, port + '-pair-paramserver')

### Initialize poll set
poller = zmq.Poller()
poller.register(subscriber, zmq.POLLIN)
poller.register(replier, zmq.POLLIN)
poller.register(pair, zmq.POLLIN)

def send_reply(socket,flags=0, copy=True, track=False):
  md = socket.recv_json()
  if "model_name" not in running_model:
    running_model["model_name"] = md["model_name"]
  key = md["graph"]+"|"+md["node"]
  ### check if weights are in cache
  if key in tensors:
    print "loading weights from cache!"
    A = tensors[key]
  else:
    print "initializing tensor -- ", key
    A = intialize_weights(md)
    tensors[key] = A
  
  ### reply back to req
  md = dict(dtype = str(A.dtype),shape = A.shape,key = key)
  env = [json.dumps(md), A]
  socket.send_multipart(env, flags=flags, copy=copy, track=track)

def send_reply_multi(socket,flags=0, copy=True, track=False):
  print "total tensors on server -- ", len(tensors)
  msg = socket.recv()
  env = []
  for key in tensors.keys():
    A = tensors[key]
    _dict = dict(dtype = str(A.dtype),shape = A.shape,key = key)
    env.append(json.dumps(_dict))
    env.append(A)
  socket.send_multipart(env, flags=flags, copy=copy, track=track)

def subcribe_msgs(flags=0, copy=True, track=False):
  msg = subscriber.recv_multipart(flags=flags, copy=copy, track=track)
  md = json.loads(msg[1])
  buf = buffer(msg[2])
  A = np.frombuffer(buf, dtype=md["dtype"]).reshape(md["shape"])
  key = md["graph"]+"|"+md["node"]
  tensors[key] -= A
  print "applying delta ---> ", key
  return tensors[key]

def topic_subscription():
  for topic in tensors.keys():
    subscriber.setsockopt(zmq.SUBSCRIBE, topic)

subscribed_tensors = 0
counter = 0
should_continue = True
while should_continue:
  total_tensors = len(tensors) 
  if subscribed_tensors < total_tensors and total_tensors != 0:
    topic_subscription()
    subscribed_tensors = total_tensors
  socks = dict(poller.poll(1000))
  if subscriber in socks and socks[subscriber] == zmq.POLLIN:
    subcribe_msgs()
    # msg = subscriber.recv_multipart()
    # print msg
  if replier in socks and socks[replier] == zmq.POLLIN:
    send_reply_multi(replier)
  if pair in socks and socks[pair] == zmq.POLLIN:
    send_reply(pair)
  if counter % 120 == 0 and "model_name" in running_model:
    pass
    #save_params(tensors, running_model["model_name"])
  counter += 1

