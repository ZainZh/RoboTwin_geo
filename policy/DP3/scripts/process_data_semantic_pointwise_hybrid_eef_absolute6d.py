import sys

from process_data_semantic_pointwise_eef_absolute6d import build_eef_argv


if __name__ == "__main__":
    from process_data_semantic_pointwise import main

    main(build_eef_argv(sys.argv[1:], hybrid=True))
