import os

a = ["!", "2"]
with open("test.txt", 'w') as f:
    f.writelines(a)