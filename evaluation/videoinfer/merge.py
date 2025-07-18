import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Merge JSON files from a folder.")
    parser.add_argument(
        '--folder_path', type=str, required=True, help="Path to the folder containing result files"
    )
    parser.add_argument(
        '--subset_num', type=int, required=True, help="Number of subsets (e.g., 8 for result_0.json to result_7.json)"
    )
    return parser.parse_args()


def main():

    args = parse_args()
    folder_path = args.folder_path
    subset_num = args.subset_num
    output_file = f"{folder_path}/merged_result.json"

    merged_result = {}

    for i in range(subset_num):

        file_path = os.path.join(folder_path, f"pred_{i}.json")


        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                result_data = json.load(f)


            for vid_id, vid_data in result_data.items():
                if vid_id not in merged_result:
                    merged_result[vid_id] = {}

                for exp_id, exp_data in vid_data.items():
                    if exp_id not in merged_result[vid_id]:
                        merged_result[vid_id][exp_id] = {}

                    for qa_id, text_output in exp_data.items():

                        merged_result[vid_id][exp_id][qa_id] = text_output


            os.remove(file_path)
        else:
            print(f"Warning: {file_path} does not exist and will be skipped.")


    with open(output_file, "w") as f:
        json.dump(merged_result, f, indent=4)

    print(f"Merged result saved to {output_file}")


if __name__ == "__main__":
    main()
