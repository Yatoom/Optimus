def say(output, verbose, style=None):

    # No output
    if not verbose:
        return

    # Normal output
    if style is None:
        print(output)

    #
    # Title
    # =====
    if style == "title":
        print("{}\n{}\n".format(output, str("=" * len(output))))

    #
    # Subtitle
    # --------
    if style == "subtitle":
        print("\n{}\n{}".format(output, str("-" * len(output))))
