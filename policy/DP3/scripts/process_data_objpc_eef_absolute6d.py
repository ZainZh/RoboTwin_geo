import sys


def build_eef_argv(argv):
    return list(argv) + [
        "--output_suffix=-objpc-eef-absolute6d-rightbase",
        "--action_mode=eef_absolute6d",
    ]


if __name__ == "__main__":
    from process_data_objpc import main

    main(build_eef_argv(sys.argv[1:]))
