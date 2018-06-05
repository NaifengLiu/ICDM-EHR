import numpy as np
from tqdm import tqdm

with open("./data/selected_events") as f1:
    selected_events = f1.readline().split(",")


def main(input_file_path, output_file_path):
    with open(output_file_path, "w") as w:
        with open(input_file_path) as f:
            lines = f.readlines()
            for i in tqdm(range(len(lines))):
                line = lines[i]
                if line.split(",")[1] in selected_events:
                    w.write(line)
            f.close()
        w.close()


main("./data/nonhae_sorted.csv", "./data/nonhae_sorted_feature_extraction")
main("./data/hae.csv", "./data/hae_feature_extraction")


