"""Read-only salvage of already loaded Python objects; never opens source datasets."""
import os,struct,json
from pathlib import Path
ROOT=Path(__file__).parent
pid=3753433
fd=os.open('/proc/%s/mem'%pid,os.O_RDONLY)
def b(a,n):return os.pread(fd,n,a)
def q(a):return struct.unpack('<Q',b(a,8))[0]
def si(a):return struct.unpack('<q',b(a,8))[0]
def cstr(a):return b(a,200).split(b'\0')[0].decode()
cache={}
def read(a):
 if not a:return None
 if a in cache:return cache[a]
 typ=cstr(q(q(a+8)+24))
 if typ=='NoneType':return None
 if typ in ('int','bool'):
  n=si(a+16);v=sum(struct.unpack('<I',b(a+24+4*i,4))[0]<<(30*i) for i in range(abs(n)));v=v if n>=0 else -v
  return bool(v) if typ=='bool' else v
 if typ=='float':return struct.unpack('<d',b(a+16,8))[0]
 if typ=='str':
  n=si(a+16);state=struct.unpack('<I',b(a+32,4))[0];kind=(state>>2)&7;compact=(state>>5)&1;ascii_=(state>>6)&1
  assert compact
  return b(a+(48 if ascii_ else 72),n*kind).decode({1:'latin1',2:'utf-16-le',4:'utf-32-le'}[kind])
 if typ in ('list','tuple'):
  n=si(a+16);items=q(a+24) if typ=='list' else a+24
  out=[];cache[a]=out;out.extend(read(q(items+8*i)) for i in range(n));return out
 if typ=='dict':
  keys=q(a+32);values=q(a+40);size=q(keys+8);n=q(keys+32);width=1 if size<=255 else 2 if size<=65535 else 4 if size<=4294967295 else 8
  entries=keys+40+size*width;out={};cache[a]=out
  for i in range(n):
   key=q(entries+i*24+8);val=q(values+i*8) if values else q(entries+i*24+16)
   if key and val:out[read(key)]=read(val)
  return out
 if typ=='numpy.ndarray':
  data=q(a+16);nd=struct.unpack('<i',b(a+24,4))[0];dims=q(a+32);shape=[q(dims+8*i) for i in range(nd)];descr=q(a+56)
  kind=chr(b(descr+24,1)[0]);itemsize=q(descr+40)
  count=1
  for dim in shape:count*=dim
  import numpy as np
  dt={'b':'?','f':'f'+str(itemsize),'i':'i'+str(itemsize),'u':'u'+str(itemsize)}[kind]
  return np.frombuffer(b(data,count*itemsize),dtype=dt).reshape(shape).tolist()
 raise RuntimeError((typ,hex(a)))
dump=json.loads((ROOT/'LIVE_PROCESS_DUMP.json').read_text())
frame=next(f for f in dump[0]['frames'] if f['name']=='main')
out={}
for field in ['formal','oracle_rows','effect_rows','estimation','qualification','task_effects']:
 addr=next(v['addr'] for v in frame['locals'] if v['name']==field)
 out[field]=read(addr)
assert len(out['formal'])==840
assert len({r['cluster_id'] for r in out['formal']})==120
out['recovery']={'source':'already loaded live process memory, no original formal source reopened','pid':pid,'raw_formal_load_attempts_before_recovery':2,'claim_eligible':False,'selection_eligible':False}
(ROOT/'RECOVERED_LOADED_EVIDENCE.json').write_text(json.dumps(out,ensure_ascii=False,allow_nan=False))
print({k:len(v) if hasattr(v,'__len__') else None for k,v in out.items()})
