
try:
    import torchpack.distributed as dist
except:
    dist = None

__all__ = ["size", "rank", "local_rank", "init"]

def size():
    if dist:
        return dist.size()
    else:
        return 1
    
    
def rank():
    if dist:
        return dist.rank()
    else:
        return 0
    
def local_rank():
    if dist:
        return dist.local_rank()
    else:
        return 0
    
def init():
    if dist:
        return dist.init()
    else:
        pass