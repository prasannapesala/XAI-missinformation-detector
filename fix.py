import os

file_path = r'D:\xai_misinformation\src\collection\data_loader.py'
with open(file_path, 'r', encoding='utf-8') as f:
    content = f.read()

# Fix paths
content = content.replace(
    "DATA_DIR   = os.path.join(os.path.dirname(__file__), 'data')",
    "DATA_DIR   = r'D:\\xai_misinformation\\data'"
)

# Fix dataset
content = content.replace(
    '"hamzab/fake-news-detection"',
    '"GonzaloA/fake_news"'
)

with open(file_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('✅ File patched!')