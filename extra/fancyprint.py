def say(output, verbose, style=None):

    # No output
    if not verbose:
        return

    # Normal output
    if style is None:
        print(output)

    #
    # ======
    # HEADER
    # ======
    if style == "header":
        print("\n{1}\n{0}\n{1}".format(output, str("=" * len(output))))

    #
    # Title
    # =====
    if style == "title":
        print("\n{}\n{}".format(output, str("=" * len(output))))

    #
    # Subtitle
    # --------
    if style == "subtitle":
        print("\n{}\n{}".format(output, str("-" * len(output))))
