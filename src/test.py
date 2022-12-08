class Fastener:
    def __init__(self, klass, dist):
        self._type = klass
        self._dist = dist

    def get_type(self):
        return str(self._type)

    def set_type(self, klass):
        self._type = str(klass)

    def get_dist(self):
        return float(self._dist)

    def set_dist(self, dist):
        self._dist = dist

    def increase_dist(self, dist):
        self._dist += dist

class LoadedBelt:
    diameter = 34.  # millimeters
    radius = diameter/2

    def __init__(self, fastener_list=None):
        # fasteners is a list of Fasteners
        self._fastener_list = [] if fastener_list is None else fastener_list

    def rotate(self, deg):
        [fastener.increase_dist(deg) for fastener in self._fastener_list]

    def get_pos(self):
        [print(fastener.get_dist()) for fastener in self._fastener_list]

belt = LoadedBelt([Fastener("bolt", 1), Fastener("screw", 5)])
belt.rotate(3)
print(belt.get_pos())