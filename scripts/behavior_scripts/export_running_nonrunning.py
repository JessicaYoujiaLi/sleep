from argparse import ArgumentParser
from src import behavior_class as bc


def main():
    parser = ArgumentParser()
    parser.add_argument("--mouse", type=str, required=True)
    args = parser.parse_args()
    behavior = bc.behaviorData(args.mouse)


if __name__ == "__main__":
    main()
