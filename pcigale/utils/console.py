from rich.console import Console
from rich.theme import Theme
from rich.traceback import install

custom_theme = Theme(
    {
        "data": "dark_sea_green4",
        "info": "yellow4",
        "warning": "red",
        "error": "bold red",
        "repr.number": "bold bright_blue",
        "rule.line": "bright_yellow",
        "panel": "yellow4",
    }
)
console = Console(theme=custom_theme)
INFO = "[[info]INFO[/info]]"
WARNING = "[[warning]WARNING[/warning]]"
ERROR = "[[error]ERROR[/error]]"

install()
