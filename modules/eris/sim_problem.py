class Problem:
    def __init__(self, campoints1=None, campoints2=None, robposes1=None, robposes2=None):
        self._campoints1 = campoints1
        self._campoints2 = campoints2
        self._robposes1 = robposes1
        self._robposes2 = robposes2

    @property
    def campoints1(self):
        return self._campoints1

    @campoints1.setter
    def campoints1(self, campoints1):
        self._campoints1 = campoints1

    @property
    def campoints2(self):
        return self._campoints2

    @campoints2.setter
    def campoints2(self, campoints2):
        self._campoints2 = campoints2

    @property
    def robposes1(self):
        return self._robposes1

    @robposes1.setter
    def robposes1(self, robposes1):
        self._robposes1 = robposes1

    @property
    def robposes2(self):
        return self._robposes2

    @robposes2.setter
    def robposes2(self, robposes2):
        self._robposes2 = robposes2