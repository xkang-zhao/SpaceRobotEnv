import numpy as np


def test_zmq_protocol_round_trips_infer_reply_actions():
    from scripts.evals.lerobot.lerobot_zmq_protocol import make_action_reply, parse_action_reply

    actions = np.array([[1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1]], dtype=np.float32)

    reply = make_action_reply(actions)
    parsed = parse_action_reply(reply)

    assert reply["type"] == "action_chunk"
    assert reply["num_actions"] == 2
    assert parsed.shape == (2, 7)
    np.testing.assert_allclose(parsed, actions)


def test_zmq_protocol_rejects_wrong_action_shape():
    import pytest

    from scripts.evals.lerobot.lerobot_zmq_protocol import make_action_reply

    with pytest.raises(ValueError, match="shape"):
        make_action_reply(np.zeros((2, 6), dtype=np.float32))
