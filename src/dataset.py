import numpy as np

import torch
import torch.nn as nn

import datasets
import accelerate


def fim_transform(
    example,
    np_rng,
    fim_rate,
    fim_spm_rate,
    prefix_tok_id,
    suffix_tok_id,
    middle_tok_id,
    pad_tok_id,
    truncate_or_pad,
):
    """
    This function performs FIM transformation on a single example (list of tokens)
    """
    if np_rng.binomial(1, fim_rate):
        boundaries = sorted(np_rng.randint(low=0, high=len(example) + 1, size=2))

        prefix = example[: boundaries[0]]
        middle = example[boundaries[0] : boundaries[1]]
        suffix = example[boundaries[1] :]

        if truncate_or_pad:
            total_length = len(prefix) + len(middle) + len(suffix) + 3
            diff = total_length - len(example)
            if diff > 0:
                suffix = suffix[: max(0, len(suffix) - diff)]
            elif diff < 0:
                suffix.extend([pad_tok_id] * (-diff))

        if np_rng.binomial(1, fim_spm_rate):
            # Apply Suffix-Prefix-Middle (SPM) transformation
            transformed_example = (
                [prefix_tok_id, suffix_tok_id]
                + suffix
                + [middle_tok_id]
                + prefix
                + middle
            )
        else:
            # Apply Prefix-Suffix-Middle (PSM) transformation
            transformed_example = (
                [prefix_tok_id]
                + prefix
                + [suffix_tok_id]
                + suffix
                + [middle_tok_id]
                + middle
            )
    else:
        transformed_example = example

    return transformed_example


# Below function is the one you are supposed to call in the .map() function
def apply_fim(examples):
    """
    Apply FIM transformation to a batch of examples
    """
    fim_transform_ids = [fim_transform(ids) for ids in examples["input_ids"]]
    examples["input_ids"] = fim_transform_ids
    examples["labels"] = fim_transform_ids
    # If your application requires custom attention mask, please adjust this function's below line
    # since FIM transformation increases the number of tokens in input_ids and labels
    # but leaves the number of tokens unchanged in attention_masks which would cause problems
    examples["attention_mask"] = [[1] * len(mask) for mask in examples["input_ids"]]
    return examples
