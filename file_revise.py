input_file = "/home/soeunan/Sign-IDD/Data/P2014T_Ben/test.files"     # 원본 파일
output_file = "/home/soeunan/Sign-IDD/Data/P2014T_Ben/test.files"   # 저장할 파일

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 각 줄 앞에 "dev/" 붙이기
new_lines = ["test/" + line.strip() + "\n" for line in lines]

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(new_lines)

print("완료!")