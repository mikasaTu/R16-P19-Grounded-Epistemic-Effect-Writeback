import numpy as np

from r16p19.phase5_policy_broker import FrozenPolicyBroker
from r16p19.phase5_types import PolicyRequest


def test_policy_broker_infers_once_and_returns_identical_bytes():
    calls = []
    broker = FrozenPolicyBroker(lambda request: calls.append(request.key()) or np.arange(70, dtype=np.float32).reshape(10, 7))
    request = PolicyRequest("o", "h", "t", "e", "EXECUTE", "ckpt", "cfg", 1)
    one = broker.action_chunk(request)
    two = broker.action_chunk(request)
    assert one.tobytes() == two.tobytes()
    assert len(calls) == 1
    assert broker.inference_count[request.key()] == 1
