
import time
import queue
from threading import Thread
from torch.utils.data import DataLoader

from utils import printDebugInfo

class ParallelLoader:
    def __init__(self, dataloader, idx=0, epoch=0):
        self.idx = idx
        self.dataloader = dataloader
        self.init(epoch)
        
    def init(self, epoch):
        self.epoch = epoch
        self.sampler = self.dataloader.sampler
        if hasattr(self.sampler, 'set_epoch'):
            self.sampler.set_epoch(epoch)
    
        self.iter = iter(self.dataloader)
        self.data = next(self.iter)
        self.step = -1

    def __iter__(self):
        # can only be used once
        if self.step != -1:
            raise ValueError("Can only be used once")
        return self

    def __next__(self):
        # printDebugInfo("Loader step: ", self.step)
        if self.data is None:
            raise StopIteration
        self.step += 1
        rd = tuple(self.data)
        try :
            self.data = next(self.iter)
        except StopIteration:
            self.data = None
        return rd

    def __len__(self):
        return len(self.dataloader)

class ParallelLoaderPool:
    def __init__(self, dataset, cached_loader_size=2, name="", **kwargs):
        self.name = name
        self.dataset = dataset
        self.cached_loader_size = cached_loader_size
        printDebugInfo("Loader Cache size: ", cached_loader_size)
        self.ploader_list = []
        self.init_loader_list(**kwargs)
        self.current_idx = 0
        self.next_idx = 0
         
        self.thread = None
        
    def init_loader_list(self, **kwargs):
        for _ in range(self.cached_loader_size):
            loader = DataLoader(
            self.dataset, **kwargs)
            self.ploader_list.append(ParallelLoader(
                loader, idx = len(self.ploader_list), epoch=0))
        printDebugInfo("Loader initialized")
        
    def get_loader(self, epoch, next_epoch=True):
        # return the loader and start a new thread to load the next loader
        time_out_count = 0
        while self.current_idx != self.next_idx:
            # the thread is still running
            printDebugInfo("[%s]: Loader thread is still running, id %s" %(
                self.name, self.next_idx))
            time.sleep(0.1)
            time_out_count += 1
            if time_out_count > 10:
                raise RuntimeError("Loader thread is still running")
            
        if self.thread is not None:
            self.thread.join()
            self.thread = None
        
        loader = self.ploader_list[self.current_idx]
        
        printDebugInfo("[%s]: Loader get from list, id: %s " %(
            self.name, self.current_idx)," epoch: ", epoch)
        
        next_idx = (self.current_idx + 1) % self.cached_loader_size
        self.next_idx = next_idx
        
        if next_epoch:
            t = Thread(target=self.loader_thread, args=(next_idx, epoch))
            t.start()
            self.thread = t
        return loader

    def loader_thread(self, idx, epoch):
        printDebugInfo("[%s]: Loader thread started, id %s, at epoch %s" %(
            self.name, idx, epoch))
        start_time = time.time()
        # loader for the next epoch (reinit)
        loader = self.ploader_list[idx]
        loader.init(epoch)
        self.current_idx = idx
        end_time = time.time()
        printDebugInfo("[%s]: Loader reinit id %s, at epoch %s, time: %.3f" %(
            self.name, idx, epoch, end_time - start_time))

    # handling the case when main process is killed
    def __del__(self):
        for loader in self.ploader_list:
            del loader
        printDebugInfo("Loader deleted")