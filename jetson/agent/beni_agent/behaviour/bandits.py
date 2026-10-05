"""Thompson-sampling Beta-Bernoulli bandits for proactive behaviours (§11.11), stored in memory.kv.

One kv row per person ("bandit.<pid>" -> {"<arm>|<bucket>": [alpha, beta]}) so a reward costs one small write.
alpha+beta is capped (sliding-window forgetting) so habits that change are re-learned within weeks.
"""
import random
import time

PRIOR = (1.0, 1.0)
CAP = 40.0
BUCKETS = 6            # 4-hour buckets of the day


def bucket(t=None):
    return time.localtime(t or time.time()).tm_hour * BUCKETS // 24


class Bandits:
    def __init__(self, kv_get, kv_set, rng=None, prior=PRIOR):
        self.kv_get, self.kv_set = kv_get, kv_set
        self.rng = rng or random.Random()
        self.prior = prior
        self._cache = {}

    def _table(self, person):
        k = "bandit.%s" % (person or "_anyone")
        if k not in self._cache:
            self._cache[k] = self.kv_get(k, {}) or {}
        return k, self._cache[k]

    def params(self, arm, person=None, t=None):
        _, tab = self._table(person)
        return tab.get("%s|%d" % (arm, bucket(t)), list(self.prior))

    def choose(self, arms, person=None, t=None):
        """Sample theta ~ Beta(a, b) per arm, pick the max. Returns (arm, theta)."""
        best, best_s = None, -1.0
        for arm in arms:
            a, b = self.params(arm, person, t)
            s = self.rng.betavariate(a, b)
            if s > best_s:
                best, best_s = arm, s
        return best, best_s

    def update(self, arm, reward, person=None, t=None):
        """reward in [-1, 1]: +1 engaged, 0 ignored, -1 'not now' (counts as a strong failure)."""
        k, tab = self._table(person)
        key = "%s|%d" % (arm, bucket(t))
        a, b = tab.get(key, list(self.prior))
        if reward > 0:
            a += reward
        else:
            b += 1.0 + (-reward)
        if a + b > CAP:
            f = CAP / (a + b)
            a, b = max(self.prior[0], a * f), max(self.prior[1], b * f)
        tab[key] = [round(a, 3), round(b, 3)]
        self.kv_set(k, tab)
        return a, b
