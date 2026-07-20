import sys

from process_data_ndf_pointwise import main


def has_option(argv, option):
    return any(arg == option or arg.startswith(f"{option}=") for arg in argv)


def build_v2_argv(argv):
    forwarded = list(argv)
    if not has_option(forwarded, "--output_suffix"):
        forwarded.append("--output_suffix=-objpc-ndf-relation-v2")
    if not has_option(forwarded, "--relation_schema_version"):
        forwarded.extend(["--relation_schema_version", "2"])
    forwarded.extend(["--keep_feature_placeholders_in_context", "--save_relation_point_clouds"])
    return forwarded


if __name__ == "__main__":
    main(build_v2_argv(sys.argv[1:]))
