def say(output, style=None):

    # Normal output
    if style is None:
        print(output)

    #
    # -----
    # Frame
    # -----
    #
    if style == "frame":
        line = str("-" * len(output))
        print("\n{}\n{}\n{}\n".format(line, output, line))

    #
    # Title
    # =====
    #
    if style == "title":
        print("\n{}\n{}\n".format(output, str("=" * len(output))))

    #
    # Subtitle
    # --------
    #
    if style == "subtitle":
        print("\n{}\n{}\n".format(output, str("-" * len(output))))



