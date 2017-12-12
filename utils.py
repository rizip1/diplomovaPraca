import re


class Colors:
    BLUE = '\033[94m'
    ENDC = '\033[0m'


def color_print(text, color=Colors.BLUE):
    print(color + text + Colors.ENDC)


def parse_hour(date):
    m = re.search(
        r'^[0-9]{4}-[0-9]{2}-[0-9]{2} ([0-9]{2}):[0-9]{2}:[0-9]{2}$',
        date)
    hour = int(m.group(1))
    if (hour == 0):
        hour = 24
    return hour


def parse_month(date):
    m = re.search(
        r'^[0-9]{4}-([0-9]{2})-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$',
        date)
    return int(m.group(1))
