import numpy as np
import zmq
import sys
import os

from IPython import embed


### Request Socket
context = zmq.Context()
req_socket = context.socket(zmq.REQ)
req_socket.connect ("tcp://127.0.0.1:5555")

### Push messages
context = zmq.Context()
push_socket = context.socket(zmq.PUSH)
push_socket.connect("tcp://127.0.0.1:5556")

### Subscribers recv messages
context = zmq.Context()
sub_socket = context.socket(zmq.SUB)
sub_socket.connect ("tcp://127.0.0.1:5557")
sub_socket.setsockopt(zmq.SUBSCRIBE, "")

### Initialize poll set
poller = zmq.Poller()
poller.register(sub_socket, zmq.POLLIN)

def send_req(dictio):
  req_socket.send_json(dictio)
  #  Get the reply.
  data = recv_reply()
  return data

def push_array(A, graph, node, flags=0, copy=True, track=False):
  md = dict(
    dtype = str(A.dtype),
    shape = A.shape,
    graph = graph,
    node = node
  )
  push_socket.send_json(md, flags|zmq.SNDMORE)
  push_socket.send(A, flags=flags, copy=copy, track=track)

def recv_reply(flags=0, copy=True, track=False):
  md = req_socket.recv_json(flags=flags)
  msg = req_socket.recv(flags=flags, copy=copy, track=track)
  buf = buffer(msg)
  A = np.frombuffer(buf, dtype=md["dtype"])
  return A.reshape(md["shape"])

def recv_pub(flags=0, copy=True, track=False):
  md = sub_socket.recv_json(flags=flags)
  msg = sub_socket.recv(flags=flags, copy=copy, track=track)
  buf = buffer(msg)
  A = np.frombuffer(buf, dtype=md["dtype"])
  return md['key'], A.reshape(md["shape"])

def published_msgs():
  md = {}
  while True:
    socks = dict(poller.poll(1000))
    if sub_socket in socks and socks[sub_socket] == zmq.POLLIN:
      k, v = recv_pub()
      md[k] = v
    else:
      return md

def is_published():
  socks = dict(poller.poll(1000))
  if sub_socket in socks and socks[sub_socket] == zmq.POLLIN:
    return True
  return False




# send_req(dict(test="requesting"))

# A = np.random.rand(512,30)
# send_array(A)

### Subscribers recv messages
# context = zmq.Context()
# sub_socket = context.socket(zmq.SUB)
# sub_socket.connect ("tcp://127.0.0.1:5557")
# print "Connected to publisher with port: %s" % "5557"
# sub_socket.setsockopt(zmq.SUBSCRIBE, "")

# Initialize poll set
# poller = zmq.Poller()
# poller.register(sub_socket, zmq.POLLIN)

# should_continue = True
# while should_continue:
#   socks = dict(poller.poll(1000))
#   if sub_socket in socks and socks[sub_socket] == zmq.POLLIN:
#     nparr = recv_array(sub_socket)
#     print nparr
#     print "Sub: Recieved control command"

#   print "Sub: Not blocking!"