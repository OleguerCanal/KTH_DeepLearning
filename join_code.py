from glob import glob
import os

files = [f for f in glob('/home/oleguer/Documents/p4/deepstuff/KTH_DeepLearning/**', recursive=True) if os.path.isfile(f)]
files = [f for f in files if ".py" in f and ".pyc" not in f\
                                        and "/examples/" not in f\
                                        and "Assignment_1" not in f\
                                        and "Assignment_2" not in f]

with open("joint_code.py", 'wb') as list_file:
    for file in files:
        with open(file, 'rb') as f:
            f_content = f.read()
            list_file.write(("#####################################################\n").encode('utf-8'))
            list_file.write(('# The file %s contains:\n ' % file).encode('utf-8'))
            list_file.write(("#####################################################\n").encode('utf-8'))
            list_file.write(f_content)
            list_file.write(b'\n')
            list_file.write(b'\n')