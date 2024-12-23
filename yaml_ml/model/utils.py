def format_table(data, indent=0):
    """
    Formats a dictionary into a well-aligned table-like string with borders.
    """
    output = ""
    spacer = " " * indent
    header = f"{spacer}┌{'─' * 20}┬{'─' * 20}┐\n"
    header += f"{spacer}│ {'param':<18} │ {'val':<18} │\n"
    header += f"{spacer}├{'─' * 20}┼{'─' * 20}┤\n"

    if indent == 0:
        output += header

    for key, value in data.items():
        if isinstance(value, dict) or isinstance(value, list):
            output += f"{spacer}│ {key:<18} │ {str(value):<18} │\n"
        else:
            output += f"{spacer}│ {key:<18} │ {value:<18} │\n"

    output += f"{spacer}└{'─' * 20}┴{'─' * 20}┘\n"

    return output
