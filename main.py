from PIL import Image
import os
import numpy as np
import time
import cv2

DEV = True

def build():
    start_time = time.time()
    if DEV:
        os.system("g++ -g --std=c++17 ./src/*.cpp -o ./build/main -Wall")
    else:
        os.system("g++ -Ofast -march=native -flto --std=c++17 ./src/*.cpp -o ./build/main")
    end_time = time.time()
    print(f"build complete in {end_time - start_time} seconds")


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


def get_hdr_data():
    img = cv2.imread('./textures/cape_hill_1k.hdr', flags=cv2.IMREAD_ANYDEPTH)
    arr = np.array(img)
    shape = arr.shape
    arr = np.reshape(arr, (-1, 3))
    print(shape)
    np.savetxt('./textures/autogen/cape_hill_1k.txt', arr, fmt='%.8f')


def main():
    build()

    std_out = run()
    png_from_ppm(std_out, time.strftime("%Y.%m.%d-%H.%M.%S"))
    


if __name__ == "__main__":
    main()
    # get_hdr_data()