import sys


def build_eef_argv(argv, *, hybrid: bool = False):
    suffix = (
        "-objpc-semantic-pointwise-hybrid-eef-absolute6d-rightbase"
        if hybrid
        else "-objpc-semantic-pointwise-eef-absolute6d-rightbase"
    )
    forwarded = list(argv) + [
        f"--output_suffix={suffix}",
        "--action_mode=eef_absolute6d",
    ]
    if hybrid:
        forwarded.append("--keep_feature_placeholders_in_context")
    return forwarded


if __name__ == "__main__":
    from process_data_semantic_pointwise import main

    main(build_eef_argv(sys.argv[1:], hybrid=False))
