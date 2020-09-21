import multiprocessing
from concurrent.futures import ProcessPoolExecutor,as_completed,wait
import time

def echo(n):
    return n

class mptest(object):

    def __init__(self):
        self.l=0
        self.r=100
    
    def mp(self):
        start_time = time.time()
        executor = ProcessPoolExecutor(max_workers=12)
        # task_list = [executor.submit(fib, n) for n in range(3, 35)]
        # process_results = [task.result() for task in as_completed(task_list)]
        task_list = [executor.submit(echo, n) for n in range(self.l,self.r)]
        process_results = [task.result() for task in as_completed(task_list)]
        print(process_results)
        print("ProcessPoolExecutor time is: {}".format(time.time() - start_time))    

def main():
    obj=mptest()
    obj.mp()

if __name__ == "__main__":
    main()