
class Batch:
    def __init__(self, size, train_func):
        self.size = size
        self.train_func = train_func
        self.rewards = [0.0] * size
        self.actions = [0] * size
        self.__remain_idx = set(range(size))
    
    def update(self, idx, reward, action):
        assert(idx < self.size)
        self.rewards[idx] = reward
        self.actions[idx] = action
        if idx in self.__remain_idx:
            self.__remain_idx.remove(idx)
    
    def done(self):
        return len(self.__remain_idx) == 0