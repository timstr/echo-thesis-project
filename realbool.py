def realbool(s):
    assert isinstance(s, str)
    sl = s.lower()
    if sl in ["1", "true", "t", "yes", "y"]:
        return True
    elif sl in ["0", "false", "f", "no", "n"]:
        return False
    else:
        raise Exception(f'"{sl}" is not a real boolean')
