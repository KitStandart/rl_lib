import numpy as np
from saving_data.utils import save_data, load_data
from .random_buffer import Random_Buffer, Random_Recurrent_Buffer

class Sum_Tree:
    def __init__(self, size):
        self.tree = np.zeros(2*size - 1, dtype = np.float64)
        self.size = size
        self.count = 0
        self.real_size = 0

    def clear(self, ):
        self.tree = np.zeros(self.tree.shape, dtype = np.float64)
        self.count = 0
        self.real_size = 0

    @property
    def total(self):
        return self.tree[0]

    def update(self, data_idx, value):
        assert type(data_idx)!=np.array and type(data_idx)!=list and type(data_idx)!=tuple
        idx = data_idx + self.size - 1
        change = value - self.tree[idx]
        self.tree[idx] = value
        parent = (idx - 1) // 2
        idx =[]
        idx.append(parent)
        while parent > 0:
              parent = (parent - 1) // 2
              idx.append(parent)
        parent = np.asarray(idx, dtype = np.int32)
        self.tree[parent] += change

    def add(self, value):
        self.update(self.count, value)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, s):
        assert np.any(s <= self.total)
        batch_size = s.shape[0]
        idx = np.zeros(batch_size, dtype=np.int32)
        left = 2 * idx + 1
        right = left + 1
        while np.any(idx != left):
          idx = np.where(s <= self.tree[left], left, right)
          s = np.where(s <= self.tree[left], s, s - self.tree[left])
          left = 2 * idx + 1
          left = np.where(left >= self.tree.shape[0], idx, left)
          right = np.where(left >= self.tree.shape[0], right, left + 1)

        data_idx = idx - self.size + 1
        return data_idx, self.tree[idx]   

class Prioritized_Replay_Buffer(Random_Buffer):
    '''
    Аргументы:
        size: int
        n_step: int
        discount_factor: float
        num_var: int (Кол-во сохраянемых переменных, по умполчанию 5 (s, a, r, d, s'))
        eps: float
        alpha: float
        beta: float
        beta_changing: float
        beta_changing_curve: str
        max_priority: float
    '''
    def __init__(self, **kwargs):
        size = kwargs.get("size", 100000)
        Random_Buffer.__init__(self, **kwargs)

        self.tree = Sum_Tree(size=size)
        # PER params
        self.eps = kwargs.get("eps", 1e-2)  
        self.alpha = kwargs.get("alpha", 0.6)  
        self.beta = kwargs.get("beta", 0.4)  
        self.beta_changing = kwargs.get("beta_changing", 5e-4)
        self.beta_changing_curve = kwargs.get("beta_changing_curve", 'linear')
        self.max_priority = kwargs.get("max_priority", 1e-2) 

    def clear(self, ):
        Random_Buffer.clear(self,)
        self.tree.clear()

    def add(self, samples, priority = None):
        '''samples -> tuple(s,a,r,d,s')
            priority -> float если передается, то приоритет в буфере выставлется по преданному числу,
                                                                 по умолчанию вычисляется по self.max_priotiry
        '''
        if Random_Buffer.add(self, samples):
            self.tree.add(self.max_priority if priority == None else priority)
        assert self.count == self.tree.count and self.real_size == self.tree.real_size, "tree and has same real sizes"

    def sample(self, batch_size):
        data_idxs, weights = self._get_idx(batch_size)
        return {**Random_Buffer.sample(self, batch_size, data_idxs), 'data_idxs': data_idxs, 'weights': weights}
    
    def update_priorities(self, data_idxs, priorities):
        priorities = self._calculate_new_priority(priorities)
        self.max_priority = max(self.max_priority, max(priorities))
        for data_idx, priority in zip(data_idxs, priorities):
          self.tree.update(data_idx, priority)

    def save(self, path):
        Random_Buffer.save(self, path)
        save_data(path, {
                'tree': self.tree.tree,
                'tree_count': self.tree.count, 
                'tree_real_size': self.tree.real_size
                    })

    def load(self, path):
        Random_Buffer.load(self, path)        
        data = load_data(path)
        self.tree.tree = data['tree']
        self.tree.count = data['tree_count']
        self.tree.real_size = data['tree_real_size']               
    
    def _get_idx(self, batch_size):
        assert self.real_size >= batch_size, "buffer contains less samples than batch size"

        segment = self.tree.total / batch_size
        segment_array = np.random.uniform(segment * np.arange(batch_size), segment * (np.arange(batch_size) + 1))
        data_idxs, priorities = self.tree.get(segment_array)
        weights = self._calculate_weights(priorities)

        if self.beta_changing_curve.lower() == 'exponential':
            precision = len(str(self.beta_changing).split('.')[1])
            self.beta = round(1 - np.power(np.exp, -self.beta*self.beta_changing), precision)
        else: self.beta = min(1, self.beta*self.beta_changing)

        return data_idxs, weights

    def _calculate_new_priority(self, error):
        return np.power(error + self.eps, self.alpha)

    def _calculate_weights(self, weights):
        weights = weights / self.tree.total
        weights = np.power(self.real_size * weights,  -self.beta)
        weights = weights / weights.max()
        weights = weights.astype(np.float32)
        return weights

class Prioritized_Replay_Recurrent_Buffer(Prioritized_Replay_Buffer, Random_Recurrent_Buffer, Random_Buffer):
    def __init__(self, **kwargs):
        kwargs["num_var"] = 7
        self.trace_length = kwargs.get("trace_length", 10)
        Prioritized_Replay_Buffer.__init__(self, **kwargs)
        Random_Recurrent_Buffer.__init__(self, **kwargs)
        
        kwargs["size"] = self.trace_length
        self.trace_window = Random_Buffer(**kwargs) #нужно для того чтобы граничные индексы кольцевого буфера из приоритетного выбора были с историческими данными

    def clear(self, ):
        Prioritized_Replay_Buffer.clear(self,)
        self.trace_window.clear()

    def add(self, samples, priority = None):
        if self.trace_window.real_size != self.trace_length:
          self.trace_window.add(samples)
        else:
          if self.real_size != self.size: self.trace_window.add(samples)
          else: self.trace_window.add(self.data[self.count])
          Prioritized_Replay_Buffer.add(self, samples, priority)

    def sample(self, batch_size):
        if self.data[-1][1] == 0: self.data[-self.trace_length:] = self.trace_window.data
        data_idxs, weights = Prioritized_Replay_Buffer._get_idx(self, batch_size)
        data = Random_Recurrent_Buffer.sample(self, batch_size, data_idxs)
        data = self.add_trace_window(data, data_idxs)
        return {**data, 'data_idxs': data_idxs, 'weights': weights}       

    def add_trace_window(self, data, data_idxs):
        error_idx = np.where((data_idxs < self.count + self.trace_length) & (data_idxs > self.count))[0]
        errors = data_idxs[error_idx]
        count = abs(errors - self.count - self.trace_length)
        for e, c in zip(error_idx, count):
            repair_data = self.get_repair_data(c)
            z = np.arange(c, 0, -1)-1
            for key in data.keys():
                if key in ('h_t', 'c_t'): z = np.arange(2, 0, -1)-1
                data[key][e][z] = repair_data[key][-2:] if key in ('h_t', 'c_t') else repair_data[key]
        return data

    def get_repair_data(self, count):
        l = np.arange(count) + 1
        return self.trace_window.sample(count, self.trace_window.count - l)

    def save(self, path):
        Prioritized_Replay_Buffer.save(self, path)
        self.trace_window.save(path+'_small_buffer')

    def load(self, path):
        Prioritized_Replay_Buffer.load(self, path)
        self.trace_window.load(path+'_small_buffer')        
