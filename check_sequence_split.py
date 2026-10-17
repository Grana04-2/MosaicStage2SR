from data.stage2_dataset import (
    load_manifest,
    split_records_by_sequence,
    get_sequence_id_from_stem,
)


dataset_root = (
    "./datasets/stage2_1280x720"
)

records = load_manifest(
    dataset_root
)

train_records, val_records = (
    split_records_by_sequence(
        records,
        val_ratio=0.10,
        seed=1234
    )
)

train_sequences = sorted(
    {
        get_sequence_id_from_stem(
            record["stem"]
        )
        for record in train_records
    }
)

val_sequences = sorted(
    {
        get_sequence_id_from_stem(
            record["stem"]
        )
        for record in val_records
    }
)

overlap = (
    set(train_sequences)
    &
    set(val_sequences)
)

print(
    f"Total frames     : "
    f"{len(records)}"
)

print(
    f"Train frames     : "
    f"{len(train_records)}"
)

print(
    f"Validation frames: "
    f"{len(val_records)}"
)

print()

print(
    f"Train sequences  : "
    f"{len(train_sequences)}"
)

print(
    f"Val sequences    : "
    f"{len(val_sequences)}"
)

print()

print(
    "Validation sequences:"
)

for sequence_id in val_sequences:

    count = sum(
        1
        for record in val_records
        if (
            get_sequence_id_from_stem(
                record["stem"]
            )
            == sequence_id
        )
    )

    print(
        f"  {sequence_id}: "
        f"{count} frames"
    )

print()

print(
    f"Sequence overlap : "
    f"{len(overlap)}"
)

if overlap:

    print(
        sorted(
            overlap
        )
    )

else:

    print(
        "OK: train / validation "
        "are completely separated."
    )