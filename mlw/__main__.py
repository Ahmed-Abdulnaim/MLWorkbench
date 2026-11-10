"""`python -m mlw` -> GUI when no arguments are given, otherwise the CLI."""
import sys

if __name__ == "__main__":
    if len(sys.argv) == 1:
        from .gui import main
        main()
    else:
        from .cli import main
        sys.exit(main())
