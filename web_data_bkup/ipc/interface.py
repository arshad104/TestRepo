"""Interface class for Chapter on Distributed Computing

This implements an "interface" to our network of nodes
"""
from sets import Set
import time
import uuid
from threading import Thread

import zmq
from zmq.eventloop.ioloop import IOLoop, PeriodicCallback
from zmq.eventloop.zmqstream import ZMQStream

import udplib
from IPython import embed
import sys

# =====================================================================
# Synchronous part, works in our application thread

def pipe(ctx):
    """create an inproc PAIR pipe"""
    a = ctx.socket(zmq.PAIR)
    b = ctx.socket(zmq.PAIR)
    url = "inproc://%s" % uuid.uuid1()
    a.bind(url)
    b.connect(url)
    return a, b

class Interface(object):
    """Interface class.
    Just starts a UDP ping agent in a background thread."""
    ctx = None      # Our context
    pipe = None     # Pipe through to agent

    def __init__(self):
        self.ctx = zmq.Context()
        p0, p1 = pipe(self.ctx)
        self.agent = InterfaceAgent(self.ctx, p1)
        self.agent_thread = Thread(target=self.agent.start)
        self.agent_thread.start()
        self.pipe = p0
        
    def stop(self):
        self.pipe.close()
        self.agent.stop()
        self.ctx.term()
    
    def recv(self):
        """receive a message from our interface"""
        return self.pipe.recv_multipart()
    
    
# =====================================================================
# Asynchronous part, works in the background

PING_PORT_NUMBER    = 9999
PING_INTERVAL       = 1.0  # Once per second
PEER_EXPIRY         = 5.0  # Five seconds and it's gone
UUID_BYTES          = 64

class Peer(object):
    
    uuid = None
    expires_at = None
    
    def __init__(self, uuid,ip):
        self.uuid = uuid
        self.ip = ip
        self.is_alive()
        self.ports = Set()
    
    def is_alive(self):
        """Reset the peers expiry time
        Call this method whenever we get any activity from a peer.
        """
        self.expires_at = time.time() + PEER_EXPIRY

class InterfaceAgent(object):
    """This structure holds the context for our agent so we can
    pass that around cleanly to methods that need it
    """
    
    ctx = None                 # ZMQ context
    pipe = None                # Pipe back to application
    udp = None                 # UDP object
    uuid = None                # Our UUID as binary blob
    peers = None               # Hash of known peers, fast lookup


    def __init__(self, ctx, pipe, loop=None):
        self.ctx = ctx
        self.pipe = pipe
        if loop is None:
            loop = IOLoop.instance()
        self.loop = loop
        self.udp = udplib.UDP(PING_PORT_NUMBER)
        self.uuid = uuid.uuid4().hex.encode('utf8')
        self.peers = {}
        self.ports = Set()
    
    def stop(self):
        self.pipe.close()
        self.loop.stop()
    
    def __del__(self):
        try:
            self.stop()
        except:
            pass

    def register_service(self,service):
        #port-socket_type-service_type
        self.ports.add(service)        
    
    def start(self):
        loop = self.loop
        loop.add_handler(self.udp.handle.fileno(), self.handle_beacon, loop.READ)
        stream = ZMQStream(self.pipe, loop)
        stream.on_recv(self.control_message)
        pc = PeriodicCallback(self.send_ping, PING_INTERVAL * 1000, loop)
        pc.start()
        pc = PeriodicCallback(self.reap_peers, PING_INTERVAL * 1000, loop)
        pc.start()
        pc = PeriodicCallback(self.send_ports, PING_INTERVAL * 1000, loop)
        pc.start()
        loop.start()
    
    def send_ping(self, *a, **kw):
        try:
            self.udp.send(self.uuid)
        except Exception as e:
            self.loop.stop()

    def send_ports(self, *a, **kw):
        for port in self.ports:
            try:
                self.udp.send(self.uuid + '-' + port)
            except Exception as e:
                self.loop.stop()        

    def control_message(self, event):
        """Here we handle the different control messages from the frontend."""
        print("control message: %s", msg)
    
    def handle_beacon(self, pd, event):
        message, addrinfo = self.udp.recv(UUID_BYTES)
        message_parts = message.split('-')
        uuid = message_parts[0]
       
        if uuid in self.peers:
            self.peers[uuid].is_alive()
        else:
            self.peers[uuid] = Peer(uuid,addrinfo[0])
            self.pipe.send_multipart([b'JOINED', uuid])
        if len(message_parts) > 1:
            self.peers[uuid].ports.add(message_parts[1] + "-" + message_parts[2] + "-" + message_parts[3])
           
    def reap_peers(self):
        now = time.time()
        for peer in list(self.peers.values()):
            if peer.expires_at < now:
                print("reaping %s" % peer.uuid, peer.expires_at, now)
                self.peers.pop(peer.uuid)


pd = Interface()

def register_service(s):
    print "Registering Service: " + s
    pd.agent.register_service(s)

def get_offset():
    OFFSET = 0
    for peer_name, peer_obj in pd.agent.peers.iteritems():
        for port in peer_obj.ports:
            info = port.split("-")
            if peer_name != pd.agent.uuid:
                if info[2] == 'paramserver':
                    OFFSET += 1
    return str(OFFSET)

def get_peers():
    peers_list = []
    for peer_name, peer_obj in pd.agent.peers.iteritems():
        for port in peer_obj.ports:
            if peer_name != pd.agent.uuid:
                peers_list.append(peer_obj.ip +"-"+ port)
    return peers_list

def send_reply(socket, flags=0, copy=True, track=False):
    md = socket.recv_json()
    if "getOffset" in md:
        msg = get_offset()
        socket.send(msg)
    elif "registerService" in md:
        s = md["registerService"]
        register_service(s)
        msg = "registered: " + s
        socket.send(msg)
    elif "getPeers" in md:
        socket.send_multipart(get_peers(), flags=flags, copy=copy, track=track)

### set context
context = zmq.Context()

### replier will respond to requests
replier = context.socket(zmq.REP)
url = "ipc://" + sys.argv[1]
replier.bind(url)

poller = zmq.Poller()
poller.register(replier, zmq.POLLIN)

should_continue = True
while should_continue:
    socks = dict(poller.poll(1000))
    if replier in socks and socks[replier] == zmq.POLLIN:
        send_reply(replier)