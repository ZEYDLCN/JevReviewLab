import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import lab


class CreditTests(unittest.TestCase):
    def test_two_attempt_limit_and_duplicate_prevention(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(lab, 'OUT', Path(directory)):
            lab.reserve_credit('first')
            with self.assertRaises(RuntimeError):
                lab.reserve_credit('first')
            lab.reserve_credit('second')
            with self.assertRaises(RuntimeError):
                lab.reserve_credit('third')

    def test_response_requires_exact_ids_and_valid_labels(self):
        request = {'questions': {'r1': {}}}
        self.assertEqual(lab.parse_answers(request, {'answers': {'r1': {'choice': 'positive'}}}),
                         [{'id': 'r1', 'prediction': 'positive'}])
        for response in [{'answers': {}}, {'answers': {'r1': {'choice': 'neutral'}}},
                         {'answers': {'r1': {'choice': 'positive'}, 'r2': {'choice': 'negative'}}}]:
            with self.assertRaises(ValueError):
                lab.parse_answers(request, response)

    def test_normalization(self):
        self.assertEqual(lab.normalize('İYİ ÜRÜN!'), lab.normalize('iyi ürün'))

    def test_dry_run_never_sends_or_reserves_credit(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(lab, 'OUT', Path(directory)):
            self.make_request(Path(directory))
            with patch('requests.post') as post:
                lab.run_jev(1, False)
                post.assert_not_called()
            self.assertFalse((Path(directory)/'credit-ledger').exists())

    def test_network_failure_is_not_retried_and_cannot_be_replayed(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(lab, 'OUT', Path(directory)):
            self.make_request(Path(directory))
            with patch.dict('os.environ', {'JEV_AI_API_KEY': 'test-only'}), patch('requests.post', side_effect=TimeoutError) as post:
                with self.assertRaises(TimeoutError):
                    lab.run_jev(1, True)
                with self.assertRaises(RuntimeError):
                    lab.run_jev(1, True)
                self.assertEqual(post.call_count, 1)
            self.assertEqual(len(list((Path(directory)/'credit-ledger').glob('slot-*.json'))), 1)
            self.assertFalse((Path(directory)/'credit-ledger/request.lock').exists())

    @staticmethod
    def make_request(directory):
        lab.save(directory/'jev-request-1.json', {
            'model': 'jev-1.13.0',
            'state': [{'id': str(i), 'text': 'test review'} for i in range(12)],
            'questions': {str(i): {} for i in range(12)}
        })


if __name__ == '__main__':
    unittest.main()
