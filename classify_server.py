from deepclassify.DeepClassification import *
from deepclassify.ttypes import *

from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol
from thrift.protocol import TCompactProtocol
from thrift.server import TServer
import text_cnn_decoder as td
import socket
import time
class DeepClassificationHandler:  
  def __init__(self):  
    self.log = {}
    self.decoder=td.TextCnnDecoder('runs/checkpoints_server/model_store')
  def decode(self,query):
    #start=time.time()
    label,score,prob_map,score_map=self.decoder.decode(query)
    #print("decode use time:"+str(time.time()-start))
    return ClassifyResult(label,score,prob_map,score_map)

handler = DeepClassificationHandler()

processor = Processor(handler)
transport = TSocket.TServerSocket(port=9090)
tfactory = TTransport.TBufferedTransportFactory()
#tfactory = TSocket.TFramedTransportFactory()
pfactory = TBinaryProtocol.TBinaryProtocolFactory()
#pfactory = TCompactProtocol.TCompactProtocolFactory()
#server = TServer.TForkingServer(processor, transport, tfactory, pfactory)
#server = TServer.TSimpleServer(processor, transport, tfactory, pfactory)
server = TServer.TThreadedServer(processor, transport, tfactory, pfactory)
#server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
#server = TNonblockingServer.TNonblockingServer(processor, transport)

print "start python ..."
server.serve()
