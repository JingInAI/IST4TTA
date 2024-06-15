

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class MovingAverageMeter:
    """Computes and stores the moving average and current value"""
    def __init__(self, windowsize):
        self.windowsize = windowsize
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.var = 0
        self.sum = 0
        self.count = 0
        self.valuequeue = []

    def update(self, val, n=1):
        self.val = val
        self.valuequeue.extend([val] * n)
        if len(self.valuequeue) > self.windowsize:
            self.valuequeue = self.valuequeue[-self.windowsize:]
        self.count = len(self.valuequeue)
        self.sum = sum(self.valuequeue)
        self.avg = self.sum / self.count
        self.var = sum([v * v for v in self.valuequeue]) / self.count - self.avg * self.avg

    def __str__(self):
        return f"count: {self.count}, val: {self.val}, sum: {self.sum}, avg: {self.avg}, var: {self.var}"
