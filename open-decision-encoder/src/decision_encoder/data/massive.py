from decision_encoder.data.public import transform_intent


def transform(row, taxonomy, source_split, row_id, intent_names, **kwargs):
    return transform_intent(
        row["utt"], intent_names[row["intent"]], taxonomy, "massive", source_split, row_id, **kwargs
    )
