import sys


def main(argv):
    try:
        print("TODO")
    except Exception as err:
        print >>sys.stderr, err.msg
        return 2

if __name__ == "__main__":
    sys.exit(main(sys.argv))
