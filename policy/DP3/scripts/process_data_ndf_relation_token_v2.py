import sys

from process_data_ndf_pointwise import main


def has_option(argv, option):
    return option in argv or any(value.startswith(f"{option}=") for value in argv)


def build_relation_token_v2_argv(argv):
    forwarded = list(argv)
    defaults = {
        "--output_suffix": "-objpc-ndf-relation-token-v2",
        "--relation_token_schema_version": "2",
        "--relation_token_projection_dim": "16",
        "--relation_token_projection_seed": "0",
    }
    for option, value in defaults.items():
        if not has_option(forwarded, option):
            forwarded.append(f"{option}={value}")
    forwarded.extend(
        [
            "--keep_feature_placeholders_in_context",
            "--save_relation_tokens",
            "--relation_token_gate_geometry",
        ]
    )
    return forwarded


if __name__ == "__main__":
    main(build_relation_token_v2_argv(sys.argv[1:]))
