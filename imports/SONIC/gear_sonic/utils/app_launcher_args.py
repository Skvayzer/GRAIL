"""Keep already-consumed Hydra arguments out of Kit's native argument parser."""
import sys


def consume_app_launcher_args(parser):
    """Preserve parsed AppLauncher options, remove everything from Kit's argv.

    Call inside a Hydra entry point, after Hydra has produced the config. Kit
    parses sys.argv again; Hydra's --config-path/--config-name can collide with
    its native config parser and crash app.startup. AppLauncher receives its
    parsed options explicitly through the returned namespace.
    """
    args, _ = parser.parse_known_args()
    sys.argv = sys.argv[:1]
    return args
