from myd3rlpy.iterators.base import TransitionIterator
from d3rlpy.iterators.random_iterator import RandomIterator as OldRandomIterator


class RandomIterator(TransitionIterator, OldRandomIterator):
    pass
