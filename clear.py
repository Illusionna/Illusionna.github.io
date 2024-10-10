'''
System --> MacOS & Python 3.10.0
File ----> clear.py
Author --> Illusionna
Create --> 2024-10-09 13:22:56
Website -> https://orzzz.net
'''

import os

os.system('clear')

for dirpath, dirnames, filenames in os.walk(os.getcwd()):
    for name in filenames:
        if name == '.DS_Store':
            file = os.path.join(dirpath, name)
            try:
                os.remove(file)
                print(f'* {file}')
            except Exception as e:
                print(f'! {file}, {str(e)}')