import sys


def has_output_suffix(argv):
    return any(arg == "--output_suffix" or arg.startswith("--output_suffix=") for arg in argv)


def build_eef_argv(argv, *, hybrid: bool = False):
    suffix = (
        "-objpc-semantic-pointwise-hybrid-eef-absolute6d-rightbase"
        if hybrid
        else "-objpc-semantic-pointwise-eef-absolute6d-rightbase"
    )
    forwarded = list(argv)
    if not has_output_suffix(forwarded):
        forwarded.append(f"--output_suffix={suffix}")
    forwarded.append("--action_mode=eef_absolute6d")
    if hybrid:
        forwarded.append("--keep_feature_placeholders_in_context")
    return forwarded


if __name__ == "__main__":
    from process_data_semantic_pointwise import main

    main(build_eef_argv(sys.argv[1:], hybrid=False))
