from numpy import random
import pytest


@pytest.fixture
def generator(request):
    seed_string = request.node.nodeid
    seed = [ord(character) for character in seed_string]
    return random.default_rng(seed)
