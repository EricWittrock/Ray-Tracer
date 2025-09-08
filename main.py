from PIL import Image
import os
import numpy as np
import time
import sys


DEV = False

def build():
    start_time = time.time()
    if DEV:
        os.system("g++ -g --std=c++17 ./cpu/*.cpp -o ./build/main -Wall")
    else:
        os.system("g++ -Ofast -march=native -flto --std=c++17 ./cpu/*.cpp -o ./build/main")
    end_time = time.time()
    print(f"build (DEV={DEV}) complete in {end_time - start_time} seconds")


def run():
    start_time = time.time()
    std_out = os.popen(".\\build\\main.exe").read()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    return std_out


def png_from_ppm(content, filename):
    content = content.split('\n')[:-1]
    width, height = content[1].split(' ')

    img = np.zeros((int(height), int(width), 3), dtype=np.uint8)
    for i, line in enumerate(content[3:]):
        r, g, b = line.split(' ')
        img[i // int(width), i % int(width)] = [int(r), int(g), int(b)]

    img = Image.fromarray(img)
    img.save(f'./output/{filename}.png')


def main():
    build()

    std_out = run()
    png_from_ppm(std_out, time.strftime("%Y.%m.%d-%H.%M.%S"))
    


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "dev":
        DEV = True
    main()
    