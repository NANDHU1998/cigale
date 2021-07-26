from rich.console import Console
from rich.theme import Theme
from rich.traceback import install

custom_theme = Theme({
    "info" : "dim cyan",
    "warning": "red",
    "error": "bold red",
    "repr.number": "bold bright_blue",
    "rule.line": "bright_yellow"
})
console = Console(theme=custom_theme)
INFO = "[[info]INFO[/info]]"
WARNING = "[[warning]WARNING[/warning]]"
ERROR = "[[error]ERROR[/error]]"

install()
