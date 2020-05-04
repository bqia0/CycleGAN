import os
import options
import networks
import utils
from argparse import ArgumentParser


def main():
    args = options.get_args()

    cycleGAN = networks.CycleGAN(args)

    if args.train:
        print("Training")
        print(args)
        cycleGAN.train(args)
    elif args.test:
        print("Testing")
        cycleGAN.test(args)
    else:
        print("What are we even doing here?")

if __name__ == "__main__":
    main()