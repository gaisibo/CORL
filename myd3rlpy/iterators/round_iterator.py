from myd3rlpy.iterators.base import TransitionIterator
from d3rlpy.iterators.round_iterator import RoundIterator as OldRoundIterator


class RoundIterator(TransitionIterator, OldRoundIterator):
    pass
