import os


def list_subdirectories_with_file_count(path):
    print(path)
    for root, dirs, _ in os.walk(path):
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            file_count = sum([len(files) for _, _, files in os.walk(dir_path)])
            print(f"{dir_path}: {file_count} files")


if __name__ == "__main__":
    path = "/Users/alexdemeo/Desktop/merged-SD/"  # Change this to the path of the directory you want to analyze
    list_subdirectories_with_file_count(path)

