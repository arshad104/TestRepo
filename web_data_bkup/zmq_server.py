import mongowrapper as mdb
import numpy as np
import random
import time
import json
import zmq
import sys
import os

from IPython import embed
from common import *

tensors = {}
#served_tensors = []

### set context
context = zmq.Context()

### Reply Socket
replier = context.socket(zmq.REP)
replier.bind('tcp://127.0.0.1:5555')

### Subscribers recv messages
subscriber = context.socket(zmq.SUB)
subscriber.bind("tcp://127.0.0.1:5556")

### Initialize poll set
poller = zmq.Poller()
poller.register(subscriber, zmq.POLLIN)
poller.register(replier, zmq.POLLIN)

def send_reply(flags=0, copy=True, track=False):
  md = replier.recv_json()
  if "get_tensor" in md:
    print "server1 : no of tensors ", len(tensors)
    key = random.choice(tensors.keys())
    A = tensors[key]
  else:
    key = md["graph"]+"|"+md["node"]
    ## check if weights are in cache
    if key in tensors:
      print "server1: loading weights from cache"
      A = tensors[key]
    else:
      print "server1: initializing --> ", key
      A = intialize_weights(md)
      tensors[key] = A

  ### reply back to req
  md = dict(
    dtype = str(A.dtype),
    shape = A.shape,
    key = key
  )
  env = [json.dumps(md), A]
  replier.send_multipart(env, flags=flags, copy=copy, track=track)

def subscribers():
  print "server1: subscribing tensors "
  for topic in tensors.keys():
    subscriber.setsockopt(zmq.SUBSCRIBE, topic)

def recv_pub_msgs(flags=0, copy=True, track=False):
  msg = subscriber.recv_multipart(flags=flags, copy=copy, track=track)
  md = json.loads(msg[1])
  buf = buffer(msg[2])
  A = np.frombuffer(buf, dtype=md["dtype"]).reshape(md["shape"])
  key = md["graph"]+"|"+md["node"]
  tensors[key] -= A
  print "received topic: ", msg[0]
  print "server1: applying delta --> ", key
  return tensors[key]

subscribed_tensors = 0
should_continue = True
while should_continue:
  total_tensors = len(tensors) 
  if subscribed_tensors < total_tensors and total_tensors != 0:
    subscribers()
    subscribed_tensors = total_tensors
  socks = dict(poller.poll(1000))
  if subscriber in socks and socks[subscriber] == zmq.POLLIN:
    recv_pub_msgs()
  if replier in socks and socks[replier] == zmq.POLLIN:
    send_reply()
