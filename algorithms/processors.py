import numpy as np

class ActionProcessor(object):
    '''
    Processing a [-1,1]^n action to match the desired bounds
    specified by "low" and "high" (which are either int or
    numpy.ndarray)
    '''
    def __init__(self, low, high):
        self.low = low
        self.high = high
        if np.any(np.isinf(self.low)) or np.any(np.isinf(self.high)):
            raise ValueError("inappropriate bounds for ActionProcessor: some of them are infinity")

    def transform(self, action):
        '''
        maps the action 
        '''
        # return self.low + 0.5 * (action + 1.) * (self.high - self.low)
        return 0.5 * (self.high - self.low) * action + 0.5 * (self.high + self.low)

    def __call__(self, action):
        return self.transform(action)


class ActionProcessor_Multi (ActionProcessor):
    def transform (self, action):
        if action.shape[0] == len(self.low):
            return 0.5 * (self.high[:,None] - self.low[:,None]) * action + 0.5 * (self.high[:,None] + self.low[:,None])
        elif action.shape[1] == len(self.low):
            return 0.5 * (self.high[None,:] - self.low[None,:]) * action + 0.5 * (self.high[None,:] + self.low[None,:])
