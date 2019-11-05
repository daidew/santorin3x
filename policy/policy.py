class Policy():

    '''
    Thou shall always inherit this policy.
    '''

    def __init__(self):
        pass
    
    def step(self, state):
        pass

    def optimize_policy(self, data):
        pass