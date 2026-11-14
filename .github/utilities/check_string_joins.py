"""Checks implicitly concatenated strings for a missing space at the join.

Adjacent string literals are joined by Python, so ``"one of the" "following"``
becomes ``"one of thefollowing"``. This flags joins where the left string ends in a
letter, digit or sentence punctuation and the right string starts with a letter.
Add ``# allow-string-join`` to the line of the left string to allow a join.

Usage: ``python check_string_joins.py FILE [FILE ...]``. Exits with 1 if any join is
flagged.
"""

import io
import sys
import tokenize

ALLOW_COMMENT = "# allow-string-join"
END_CHARS = ".,:;!?)"
BACKSLASH = chr(92)

# f-strings (3.12+) and t-strings (3.14+) are split into several tokens
_STARTS = {
    getattr(tokenize, name)
    for name in ("FSTRING_START", "TSTRING_START")
    if hasattr(tokenize, name)
}
_MIDDLES = {
    getattr(tokenize, name)
    for name in ("FSTRING_MIDDLE", "TSTRING_MIDDLE")
    if hasattr(tokenize, name)
}
_ENDS = {
    getattr(tokenize, name)
    for name in ("FSTRING_END", "TSTRING_END")
    if hasattr(tokenize, name)
}


def _string_units(tokens):
    """Return (first, last) token indices of each string literal."""
    units = []
    i = 0
    while i < len(tokens):
        if tokens[i].type == tokenize.STRING:
            units.append((i, i))
        elif tokens[i].type in _STARTS:
            start, depth = i, 0
            while True:
                if tokens[i].type in _STARTS:
                    depth += 1
                elif tokens[i].type in _ENDS:
                    depth -= 1
                    if depth == 0:
                        break
                i += 1
            units.append((start, i))
        i += 1
    return units


def _text(literal):
    """Return the text between the quotes of a single-token string literal."""
    body = literal.lstrip("rRbBuUfFtT")
    quote = body[:3] if body[:3] in ('"""', "'''") else body[:1]
    return body[len(quote) : -len(quote)]


def _last_char(tokens, unit):
    """Return the last character of a literal's text, or None for an escape."""
    token = tokens[unit[1]]
    if token.type == tokenize.STRING:
        text = _text(token.string)
    elif tokens[unit[1] - 1].type in _MIDDLES:
        text = tokens[unit[1] - 1].string
    else:
        return "}"
    if not text or (len(text) >= 2 and text[-2] == BACKSLASH):
        return None
    return text[-1]


def _first_char(tokens, unit):
    token = tokens[unit[0]]
    if token.type == tokenize.STRING:
        text = _text(token.string)
    elif tokens[unit[0] + 1].type in _MIDDLES:
        text = tokens[unit[0] + 1].string
    else:
        return "{"
    return text[:1]


def check_file(path):
    """Return (line, text) for each flagged join in a file."""
    with open(path, encoding="utf-8") as f:
        source = f.read()
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (SyntaxError, tokenize.TokenError):
        return []

    lines = source.splitlines()
    units = _string_units(tokens)
    flagged = []
    for left, right in zip(units, units[1:]):
        between = tokens[left[1] + 1 : right[0]]
        if not all(t.type in (tokenize.NL, tokenize.COMMENT) for t in between):
            continue
        row = tokens[left[1]].end[0]
        if ALLOW_COMMENT in lines[row - 1]:
            continue
        last, first = _last_char(tokens, left), _first_char(tokens, right)
        if last is None or not first:
            continue
        if (last.isalnum() or last in END_CHARS) and first.isalpha():
            flagged.append((row, lines[row - 1].strip()))
    return flagged


def main(paths):
    """Print each flagged join and return the exit code."""
    failed = False
    for path in paths:
        for row, text in check_file(path):
            print(f"{path}:{row}: missing space at string join: {text}")  # noqa: T201
            failed = True
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
