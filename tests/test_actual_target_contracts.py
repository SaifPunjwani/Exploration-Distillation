import unittest

import torch

from tmx.config import RunConfig
from tmx.data import (
    build_rewarded_dataset_from_trajectories,
    build_supervised_dataset_from_trajectories,
)


class _TokenBatch(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


class FakeTokenizer:
    chat_template = None
    pad_token_id = 0
    eos_token_id = 1

    def _encode(self, text: str):
        words = [w for w in str(text).strip().split() if w]
        return list(range(2, 2 + len(words)))

    def __call__(
        self,
        texts,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        add_special_tokens=True,
    ):
        single = isinstance(texts, str)
        rows = [texts] if single else list(texts)
        encoded = []
        for row in rows:
            token_ids = self._encode(row)
            if truncation and max_length is not None:
                token_ids = token_ids[:max_length]
            encoded.append(token_ids)

        if padding:
            width = max((len(ids) for ids in encoded), default=0)
        else:
            width = None

        input_ids = []
        attention_mask = []
        for token_ids in encoded:
            if width is None:
                padded = list(token_ids)
                mask = [1] * len(token_ids)
            else:
                pad = max(0, width - len(token_ids))
                padded = list(token_ids) + ([self.pad_token_id] * pad)
                mask = ([1] * len(token_ids)) + ([0] * pad)
            input_ids.append(padded)
            attention_mask.append(mask)

        if return_tensors == "pt":
            return _TokenBatch(
                {
                    "input_ids": torch.tensor(input_ids, dtype=torch.long),
                    "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                }
            )

        if single:
            return _TokenBatch(
                {
                    "input_ids": input_ids[0] if input_ids else [],
                    "attention_mask": attention_mask[0] if attention_mask else [],
                }
            )

        return _TokenBatch({"input_ids": input_ids, "attention_mask": attention_mask})


class ActualTargetContractTests(unittest.TestCase):
    def setUp(self):
        self.cfg = RunConfig(dataset_name="gsm8k", actual_target_contract="canonical")
        self.tokenizer = FakeTokenizer()
        self.trajectories = [
            {
                "prompt": "What is 6 times 7?",
                "completion_raw": "The final answer is 42 because 6 times 7 equals 42",
                "completion_full_raw": "The final answer is 42 because 6 times 7 equals 42",
                "completion_first_line": "The final answer is 42 because 6 times 7 equals 42",
                "completion_scored": "Final answer: 42",
                "completion_canonical": "Final answer: 42",
                "actual_target_canonical": "Final answer: 42",
                "actual_target_full_trace": "The final answer is 42 because 6 times 7 equals 42",
                "reward": 1.0,
                "r_correct": 1.0,
                "r_novel": 0.3,
                "is_correct": True,
                "valid_answer_presentation": True,
            }
        ]

    def test_supervised_dataset_respects_target_contract(self):
        canonical_dataset, canonical_meta = build_supervised_dataset_from_trajectories(
            self.trajectories,
            self.tokenizer,
            self.cfg,
            max_examples=8,
            max_total_len=128,
            target_contract="canonical",
        )
        trace_dataset, trace_meta = build_supervised_dataset_from_trajectories(
            self.trajectories,
            self.tokenizer,
            self.cfg,
            max_examples=8,
            max_total_len=128,
            target_contract="full_trace",
        )

        self.assertEqual(canonical_meta["target_contract"], "canonical")
        self.assertEqual(trace_meta["target_contract"], "full_trace")
        self.assertEqual(len(canonical_dataset), 1)
        self.assertEqual(len(trace_dataset), 1)
        self.assertLess(canonical_meta["target_length"]["mean"], trace_meta["target_length"]["mean"])

        canonical_labels = canonical_dataset.tensors[2]
        trace_labels = trace_dataset.tensors[2]
        self.assertLess(
            int((canonical_labels != -100).sum().item()),
            int((trace_labels != -100).sum().item()),
        )

    def test_rewarded_dataset_respects_target_contract(self):
        canonical_dataset, canonical_meta = build_rewarded_dataset_from_trajectories(
            self.trajectories,
            self.tokenizer,
            self.cfg,
            max_examples=None,
            max_total_len=128,
            reward_key="reward",
            target_contract="canonical",
        )
        trace_dataset, trace_meta = build_rewarded_dataset_from_trajectories(
            self.trajectories,
            self.tokenizer,
            self.cfg,
            max_examples=None,
            max_total_len=128,
            reward_key="reward",
            target_contract="full_trace",
        )

        self.assertEqual(canonical_meta["target_contract"], "canonical")
        self.assertEqual(trace_meta["target_contract"], "full_trace")
        self.assertEqual(len(canonical_dataset), 1)
        self.assertEqual(len(trace_dataset), 1)
        self.assertLess(canonical_meta["target_length"]["mean"], trace_meta["target_length"]["mean"])

        canonical_labels = canonical_dataset.tensors[2]
        trace_labels = trace_dataset.tensors[2]
        self.assertLess(
            int((canonical_labels != -100).sum().item()),
            int((trace_labels != -100).sum().item()),
        )


if __name__ == "__main__":
    unittest.main()
