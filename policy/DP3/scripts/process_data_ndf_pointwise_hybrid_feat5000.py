import sys

from process_data_ndf_pointwise import main


def _extract_ndf_num_points(argv):
    for idx, token in enumerate(argv):
        if token == "--ndf_num_points" and idx + 1 < len(argv):
            return int(argv[idx + 1])
    return 5000


def has_output_suffix(argv):
    return any(arg == "--output_suffix" or arg.startswith("--output_suffix=") for arg in argv)


def build_hybrid_argv(argv):
    ndf_num_points = _extract_ndf_num_points(argv)
    forwarded = list(argv)
    if not has_output_suffix(forwarded):
        forwarded.append(f"--output_suffix=-objpc-ndf-pointwise-hybrid-feat{ndf_num_points}")
    forwarded.append("--keep_feature_placeholders_in_context")
    return forwarded


if __name__ == "__main__":
    main(build_hybrid_argv(sys.argv[1:]))
