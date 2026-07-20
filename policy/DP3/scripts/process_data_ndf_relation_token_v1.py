import sys

from process_data_ndf_pointwise import main


def has_option(argv, option):
    return option in argv or any(value.startswith(f"{option}=") for value in argv)


def build_relation_token_argv(argv):
    forwarded = list(argv)
    if not has_option(forwarded, "--output_suffix"):
        forwarded.append("--output_suffix=-objpc-ndf-relation-token-v1")
    forwarded.extend(["--keep_feature_placeholders_in_context", "--save_relation_tokens"])
    return forwarded


if __name__ == "__main__":
    main(build_relation_token_argv(sys.argv[1:]))
