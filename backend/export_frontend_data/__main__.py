"""Entry point: python -m export_frontend_data [--onboarding]"""

import argparse
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    # Quick check for --onboarding flag before full arg parsing
    if "--onboarding" in sys.argv:
        sys.argv.remove("--onboarding")
        from .onboarding_export import add_onboarding_args, run_onboarding_export

        parser = argparse.ArgumentParser(
            description="Export onboarding stage binaries for frontend visualization."
        )
        add_onboarding_args(parser)
        args = parser.parse_args()
        run_onboarding_export(args)
    else:
        from .full_export import add_full_export_args, run_full_export

        parser = argparse.ArgumentParser(
            description="Export climate simulation data as binary for frontend visualization."
        )
        add_full_export_args(parser)
        args = parser.parse_args()
        run_full_export(args)


if __name__ == "__main__":
    main()
