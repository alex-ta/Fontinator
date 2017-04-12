import os
from pathlib import Path
print("Hello, this script helps create image files")
# ask an validate input
def print_val(var_name):
    correct = "no"
    val = ""
    while "y" not in correct:
        val = input("Please enter a value for "+var_name+": ")
        correct = input("Is it correct? (y/n) \n"+var_name+": "+val+"\n")
    return val

sentences = print_val("Textfile with sentence")
folder = Path(print_val("Folder with font style types"))
img_width = print_val("Image width (good default 800)")
img_height = print_val("Image height (good default 500)")

for file in folder.iterdir():
    os.system("python toImage.py "+sentences+" "+str(file)+" img "+img_width+" "+img_height+"")
    print("File created in "+str(file).rsplit('.',1)[0])
