import sys


def has_output_suffix(argv):
    return any(arg == "--output_suffix" or arg.startswith("--output_suffix=") for arg in argv)


def build_eef_argv(argv):
    forwarded = list(argv)
    if not has_output_suffix(forwarded):
        forwarded.append("--output_suffix=-eef-absolute6d-global")
    forwarded.append("--action_mode=eef_absolute6d")
    return forwarded


if __name__ == "__main__":
    from process_data import main

    main(build_eef_argv(sys.argv[1:]))
