"""Misc functions used by loggers


"""


def seconds2str(sec):
    sec_value = sec % (24 * 3600)
    hour_value = sec_value // 3600
    sec_value %= 3600
    min_value = sec_value // 60
    sec_value %= 60
    if hour_value > 0:
        return f"{hour_value:02d}:{min_value:02d}:{sec_value:02d}"
    else:
        return f"{min_value:02d}:{sec_value:02d}"
