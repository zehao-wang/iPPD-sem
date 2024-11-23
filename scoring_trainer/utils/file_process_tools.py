import os

def find_all_ext(root_path, ext):
  paths = []
  for root, dirs, files in os.walk(root_path, topdown=True):
      for name in files:
          file_path = os.path.join(root, name)
          if file_path.endswith(ext):
              paths.append(file_path)
  return paths