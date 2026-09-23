"""Routing tests only; no torch import or training."""
import tempfile
import unittest
from pathlib import Path
from run_test2 import command_for


class RoutingTests(unittest.TestCase):
    def test_routes(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            cmd = command_for(run, run)
            self.assertEqual(cmd[-2:], ['--stage', 'all'])
            self.assertTrue(any(x.endswith('b4_offset.json') for x in cmd))
            with self.assertRaises(FileNotFoundError): command_for(run, run, predict_only=True)
            (run/'last.pt').touch()
            self.assertEqual(command_for(run, run)[-1], '--resume')
            self.assertEqual(command_for(run, run, check_only=True)[-1], '--check-only')
            (run/'completed.json').touch()
            with self.assertRaises(FileNotFoundError): command_for(run, run)
            (run/'best.pt').touch()
            self.assertEqual(command_for(run, run)[-2:], ['--stage', 'predict'])
            (run/'completed.json').unlink(); (run/'last.pt').unlink()
            with self.assertRaises(RuntimeError): command_for(run, run)
            self.assertEqual(command_for(run, run, predict_only=True)[-2:], ['--stage', 'predict'])


if __name__ == '__main__': unittest.main()
