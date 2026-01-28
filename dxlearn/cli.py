import argparse

def main():
    parser = argparse.ArgumentParser("DX-learn CLI")
    parser.add_argument("--version", action="store_true")
    args = parser.parse_args()

    if args.version:
        print("dx-learn 0.1.0")
    else:
        print("DX-learn initialized")
