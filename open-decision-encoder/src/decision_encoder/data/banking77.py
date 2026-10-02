from decision_encoder.data.public import transform_intent


def transform(row, taxonomy, source_split, row_id, **kwargs):
    return transform_intent(
        row["text"], row["category"], taxonomy, "banking77", source_split, row_id, **kwargs
    )
