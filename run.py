import argparse
from bot import Bot

if __name__ == "__main__":
    parser = argparse.ArgumentParser('Number of lines?')
    parser.add_argument('lines', type=int)
    args = parser.parse_args()
    bot = Bot(args.lines)
