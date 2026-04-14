import sys

from process_data_ndf_pointwise import main


def build_hybrid_argv(argv):
    return list(argv) + [
        "--output_suffix=-objpc-ndf-pointwise-hybrid-feat5000",
        "--keep_feature_placeholders_in_context",
    ]


if __name__ == "__main__":
    main(build_hybrid_argv(sys.argv[1:]))
