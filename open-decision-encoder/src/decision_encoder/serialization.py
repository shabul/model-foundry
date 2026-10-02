"""Token-budget-aware serialization with explicit candidate positions."""

from decision_encoder.data.schema import validate_record


def serialize(record, tokenizer, max_length=512, overflow="error"):
    validate_record(record, require_target=False)
    if overflow not in {"error", "truncate_state"}:
        raise ValueError("Unknown overflow policy")
    if tokenizer.mask_token_id is None or tokenizer.pad_token_id is None:
        raise ValueError("Encoder tokenizer must provide mask and pad tokens")

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    prefix = encode("State:\n")
    state = encode(record["state"])
    suffix = encode("\nQuestion:\n" + record["question"] + "\nOptions:\n")
    relative_positions = []
    for option in record["options"]:
        relative_positions.append(len(suffix))
        suffix.append(tokenizer.mask_token_id)
        suffix.extend(
            encode("\nLabel: " + option["label"] + "\nDescription: " + option["description"] + "\n")
        )
    special_count = tokenizer.num_special_tokens_to_add(pair=False)
    budget = max_length - special_count - len(prefix) - len(suffix)
    if budget < 1:
        raise ValueError("Question/options exceed token budget; refusing to drop candidates")
    truncated = len(state) > budget
    if truncated and overflow == "error":
        raise ValueError("State exceeds token budget; opt in to truncate_state explicitly")
    state = state[:budget]
    content = prefix + state + suffix
    ids = tokenizer.build_inputs_with_special_tokens(content)
    # Determine wrappers by wrapping unique negative sentinels, not scanning mask IDs.
    sentinel = -123456789
    wrapped = tokenizer.build_inputs_with_special_tokens([sentinel])
    offset = wrapped.index(sentinel)
    positions = [offset + len(prefix) + len(state) + p for p in relative_positions]
    if len(ids) > max_length or any(ids[i] != tokenizer.mask_token_id for i in positions):
        raise ValueError("Tokenizer special-token layout is incompatible")
    return {
        "input_ids": ids,
        "candidate_positions": positions,
        "truncated": truncated,
        "original_state_tokens": len(encode(record["state"])),
    }
