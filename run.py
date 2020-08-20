import re 


if __name__ == "__main__":
    string = r'^((ID?)(.*?)\d+)'

    other_string = r'(?<=(ID:))(.+)'
    pattern = re.compile(string, re.IGNORECASE)
    matches = re.match(pattern, "ID: 23 j333o")
    if (matches):
        print(matches.group(0))