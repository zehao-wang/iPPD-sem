import torch
from torchmetrics.metric import Metric, Tensor
# from torchmetrics.metric import Metric, _LIGHTNING_AVAILABLE, Tensor

from collections import defaultdict
from utils.time_tools import execution_time
import copy
import warnings

from heapq import heapify, heappush, heappushpop, nlargest

class Node(object):
    def __init__(self, val: int):
        self.val = val

    def __repr__(self):
        return f'Node value: {self.val}'

    def __getitem__(self, idx):
        return self.val[idx]
        
    def __lt__(self, other):
        return self.val[0] < other.val[0]

class MaxHeap(object):
    def __init__(self, top_n):
        self.h = []
        self.length = top_n
        heapify( self.h)
        
    def add(self, element):
        if len(self.h) < self.length:
            heappush(self.h, Node(element))
        else:
            heappushpop(self.h, Node(element))
            
    def getTop(self, n):
        if n< self.length:
            return nlargest(n, self.h)

        return nlargest(self.length, self.h)

class MLN_SuccessRate(Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)

        self.add_state("success_rate", default=torch.tensor(0), dist_reduce_fx="mean")
        
        self.history = defaultdict(lambda: MaxHeap(top_n=1)) # has not consider sync of history dict
        # TODO: sync of history accumulator
    
    def update(self, results):
        # update metric states
        for res in results:
            res_local = copy.deepcopy(res)
            ep = res_local['traj_id'].split('-')[0]
            score = res_local['pred']
            self.history[ep].add((score, (res_local['dis_score'], res_local['ndtw'], res_local["traj_id"], res_local['scene'])))

    def compute(self):
        # compute final result
        success_rate, pred_results = self.__get_success_rate()
        self.success_rate = torch.tensor(success_rate)
        return self.success_rate, pred_results

    def reset(self) -> None:
        """ modified reset """
        self._update_called = False
        self._forward_cache = None
        # lower lightning versions requires this implicitly to log metric objects correctly in self.log
        # if not _LIGHTNING_AVAILABLE or self._LIGHTNING_GREATER_EQUAL_1_3:
        #     self._computed = None

        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, Tensor):
                setattr(self, attr, default.detach().clone().to(current_val.device))
            else:
                setattr(self, attr, [])

        setattr(self, "history", defaultdict(lambda: MaxHeap(top_n=10)))

        # reset internal states
        self._cache = None
        self._is_synced = False

    def __get_success_rate(self):
        correct = 0
        tot = 0
        pred_results = []
        if len(self.history) < 10:
            warnings.warn(f"History length only {len(self.history)}, Please check ep_id if in normal validation floop")
            
        for k,v in self.history.items():
            tot += 1
            value = v.getTop(1)[0]
            if value[1][0] > 4: # in 3 meter to goal point
                correct += 1
                pred_results.append({"episode_id": k, "traj_id": value[1][2], "pred": value[0], "is_correct": True, "ndtw": value[1][1], "dist_score": value[1][0], 'scene_name': value[1][3]})
            else:
                pred_results.append({"episode_id": k, "traj_id": value[1][2], "pred": value[0], "is_correct": False, "ndtw": value[1][1], "dist_score": value[1][0], 'scene_name': value[1][3]})
        print('\033[92m'+f'Evaluated {tot} samples {correct} correct'+'\033[0m')
        if tot == 0:
            return 0, {}
        return correct / tot, pred_results

# TODO: correct the code
class MLN_Test_Metric(Metric):
    def __init__(self, compute_on_step=False, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(compute_on_step=compute_on_step, dist_sync_on_step=dist_sync_on_step)
 
        self.history = defaultdict(list) # has not consider sync of history dict
        # TODO: sync of history accumulator
    
    def update(self, results):
        # update metric states
        for res in results:
            res_local = copy.deepcopy(res)
            ep = res_local['traj_id'].split('-')[0]
            score = res_local['pred']
            self.history[ep].add((score, (res_local['dis_score'], res_local['ndtw'], res_local["traj_id"], res_local['scene'])))
    
    def reset(self) -> None:
        """ modified reset """
        self._update_called = False
        self._forward_cache = None
        # lower lightning versions requires this implicitly to log metric objects correctly in self.log
        # if not _LIGHTNING_AVAILABLE or self._LIGHTNING_GREATER_EQUAL_1_3:
        #     self._computed = None

        for attr, default in self._defaults.items():
            current_val = getattr(self, attr)
            if isinstance(default, Tensor):
                setattr(self, attr, default.detach().clone().to(current_val.device))
            else:
                setattr(self, attr, [])

        setattr(self, "history", defaultdict(list))

        # reset internal states
        self._cache = None
        self._is_synced = False

    def compute(self):
        # compute final result
        return None, dict(self.history)

