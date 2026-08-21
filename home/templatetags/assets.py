"""A `{% vstatic %}` tag that busts the browser cache when a file changes.

Django's `{% static %}` emits a bare path, so a browser is free to keep serving
a stylesheet or script it fetched earlier. During development that means edits
appear not to take effect; for a marker who opened the site once before, it
means seeing an older version of the page than the one that was submitted. Both
of those happened to us.

This appends the file's modification time, so the URL changes whenever the file
does and stays stable when it does not.
"""

from pathlib import Path

from django import template
from django.contrib.staticfiles import finders
from django.templatetags.static import static

register = template.Library()


@register.simple_tag
def vstatic(path):
    """Like `{% static %}`, with a version stamp taken from the file itself."""
    url = static(path)

    located = finders.find(path)
    if not located:
        return url

    try:
        stamp = int(Path(located).stat().st_mtime)
    except OSError:
        return url

    separator = "&" if "?" in url else "?"
    return f"{url}{separator}v={stamp}"
