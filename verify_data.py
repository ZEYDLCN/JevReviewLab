"""Check generated split integrity without invoking any model API."""
import json
from pathlib import Path

import pandas as pd

from lab import normalize

root = Path(__file__).resolve().parent
frames = {name: pd.read_csv(root/'data'/f'{name}.csv') for name in ['train', 'validation', 'test', 'pilot']}
for name, expected in [('train', 8400), ('validation', 1200), ('test', 2400), ('pilot', 24)]:
    frame = frames[name]
    assert len(frame) == expected and frame.id.is_unique
    assert frame.label.value_counts().to_dict() == {'negative': expected//2, 'positive': expected//2}
    assert frame.text.map(normalize).is_unique
for a, b in [('train', 'test'), ('train', 'validation'), ('validation', 'test')]:
    assert not set(frames[a].text.map(normalize)) & set(frames[b].text.map(normalize))
assert set(frames['pilot'].id) <= set(frames['test'].id)
request_ids = set()
for index in (1, 2):
    payload = json.loads((root/'artifacts'/f'jev-request-{index}.json').read_text(encoding='utf-8'))
    assert len(payload['questions']) == 12
    ids = {row['id'] for row in payload['state']}
    assert not ids & request_ids
    assert ids == set(payload['questions'])
    assert all(set(row) == {'id', 'text'} for row in payload['state'])
    request_ids |= ids
assert request_ids == set(frames['pilot'].id)
print('PASS: balanced splits, no normalized exact overlap, pilot membership, label-free API payloads.')
