def halve_until_1(start):
    element = start
    while True:
        if element < 1:
            break
        yield element
        element = int(element / 2)
