class Problem:
    def __init__(self, campoints=None, campoints_true = None, robposes=None):
        self._campoints = campoints
        self._robposes = robposes
        self._campoints_true = campoints_true

    @property
    def campoints(self):
        return self._campoints

    @campoints.setter
    def campoints(self, campoints):
        self._campoints = campoints
    
    @property
    def campoints_true(self):
        return self._campoints_true

    @campoints_true.setter
    def campoints_true(self, campoints_true):
        self._campoints_true = campoints_true

    @property
    def robposes(self):
        return self._robposes

    @robposes.setter
    def robposes(self, robposes):
        self._robposes = robposes