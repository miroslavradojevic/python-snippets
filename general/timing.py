import time

if __name__ == '__main__':
    s_time = time.time()
    for i in range(3):
        print("{:04d}".format(i+1))
        time.sleep(1.0)
    e_time = time.time()
    print("--time: {:3.4f} [seconds]".format(e_time - s_time))
