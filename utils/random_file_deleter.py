import os
import argparse
import glob
import random




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='File deleter settings')
    parser.add_argument("--file_path", default=None,
        help='Path to your MIDI files.')
    args = parser.parse_args()

    if not args.file_path:
        print("You have to set the path to you MIDI files using --file_path flag")
    else:
        files = glob.glob(args.file_path + '*.mid')
        length = len(files)
        print("There are {} MIDI files".format(length))
    reduce_by = float(input("How many files would you like to keep? In percent 10% = 0.1: "))
    random.shuffle(files)
    for f in files:
        os.remove(f)
        if(len(glob.glob(args.file_path + '*.mid')) <= (reduce_by * length)):
            break
    print("There are {} MIDI files left".format(len(glob.glob(args.file_path + '*.mid'))))
