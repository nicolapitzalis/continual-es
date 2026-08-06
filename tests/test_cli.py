import unittest

from es.cli.evaluate import build_parser as build_evaluate_parser
from es.cli.plot import build_parser as build_plot_parser
from es.cli.train import build_parser as build_train_parser


class CliTest(unittest.TestCase):
    def test_train_parser_accepts_minimal_command(self):
        args = build_train_parser().parse_args(["--env", "CartPole-v1"])
        self.assertEqual(args.env, ["CartPole-v1"])
        self.assertEqual(args.num_directions, 384)

    def test_evaluate_parser_accepts_checkpoint(self):
        args = build_evaluate_parser().parse_args(
            ["--checkpoint", "policy.pth"]
        )
        self.assertEqual(args.checkpoint, "policy.pth")

    def test_plot_parser_accepts_curve_log(self):
        args = build_plot_parser().parse_args(["curves", "run.csv"])
        self.assertEqual(args.command, "curves")
        self.assertEqual(args.logs, ["run.csv"])


if __name__ == "__main__":
    unittest.main()
