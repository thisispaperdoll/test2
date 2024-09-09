import os
import shutil
import re

TARGET_DIR = "/Users/sangmin/Desktop/09_ML/test"

def organize_files():
    pattern = re.compile(r'(\d[LR])\.png$')
    
    # 모든 가능한 디렉토리 이름 리스트
    directories = ['5L', '4L', '3L', '2L', '1L', '0L', '5R', '4R', '3R', '2R', '1R', '0R']
    
    # 디렉토리가 없으면 생성
    for dir_name in directories:
        dir_path = os.path.join(TARGET_DIR, dir_name)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            print(f"Created directory: {dir_path}")

    # TARGET_DIR의 모든 파일을 검사
    for filename in os.listdir(TARGET_DIR):
        if filename.endswith('.png'):
            match = pattern.search(filename)
            if match:
                dir_name = match.group(1)
                if dir_name in directories:
                    source_path = os.path.join(TARGET_DIR, filename)
                    target_dir = os.path.join(TARGET_DIR, dir_name)
                    target_path = os.path.join(target_dir, filename)
                    
                    # 파일 이동
                    shutil.move(source_path, target_path)
                    print(f"Moved {filename} to {target_dir}/")
                else:
                    print(f"Skipped {filename}: No matching directory")
            else:
                print(f"Skipped {filename}: Doesn't match the pattern")

if __name__ == "__main__":
    organize_files()
    print("File organization complete.")