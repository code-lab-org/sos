import argparse

parser = argparse.ArgumentParser(prog='SOS Optimizer')
parser.add_argument('date')

if __name__ == "__main__":
    args = parser.parse_args()
    print(f"The passed date is {args.date}")

    from dotenv import load_dotenv
    load_dotenv()

    import os
    date = os.environ.get("date", "default")
    print(f"The environ var date is {date}")



vbudgfiuer =1 