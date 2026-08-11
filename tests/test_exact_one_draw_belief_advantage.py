import unittest

from scripts.audit_exact_one_draw_belief_advantage import candidate_override
from xiamen_mahjong.agents import GameAction


class ExactOneDrawBeliefAdvantageTests(unittest.TestCase):
    class ControlledAgent:
        def __init__(self, action):
            self.action = action

        def choose_turn_action(self, _game, _actor_seat):
            return self.action

    class State:
        tour_state = None
        gold_discard_lock_seat = None

    def test_only_changed_discard_is_an_override(self):
        teacher_action = GameAction("discard", 1)
        candidate_action = GameAction("discard", 2)
        self.assertEqual(
            candidate_override(
                self.ControlledAgent(candidate_action),
                self.ControlledAgent(teacher_action),
                self.State(),
                0,
                (teacher_action, candidate_action),
            ),
            (teacher_action, candidate_action),
        )
        self.assertIsNone(
            candidate_override(
                self.ControlledAgent(teacher_action),
                self.ControlledAgent(teacher_action),
                self.State(),
                0,
                (teacher_action, candidate_action),
            )
        )

    def test_response_or_special_actions_are_not_exported(self):
        passed = GameAction("pass")
        pong = GameAction("pong", 2, (2, 2))
        self.assertIsNone(
            candidate_override(
                self.ControlledAgent(pong),
                self.ControlledAgent(passed),
                self.State(),
                0,
                (passed, pong),
            )
        )


if __name__ == "__main__":
    unittest.main()
