import os

def test_path_creation():
    base_dir = os.getcwd()
    target_dir = os.path.join(base_dir, 'ml', 'models', 'registry')
    dummy_file_path = os.path.join(target_dir, 'test_file.txt')

    print(f"Attempting to create directory: {target_dir}")
    os.makedirs(target_dir, exist_ok=True)
    print(f"Directory exists: {os.path.exists(target_dir)}")

    print(f"Attempting to create file: {dummy_file_path}")
    with open(dummy_file_path, 'w') as f:
        f.write("Hello, world!")
    print(f"File exists: {os.path.exists(dummy_file_path)}")

if __name__ == "__main__":
    test_path_creation()