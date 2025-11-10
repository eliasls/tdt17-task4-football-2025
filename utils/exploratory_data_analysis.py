def print_lines_in_file(file_path, number_of_lines):
  with open(file_path, "r") as f:
    lines = f.readlines()

  for line in lines[:number_of_lines]:
      print(line.strip())
  