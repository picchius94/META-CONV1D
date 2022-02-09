from multiprocessing import Pool
import subprocess
from datetime import datetime

collection_file = "./collect_dataset_parse.py"
n_jobs = 17

current_time = datetime.now().strftime("%Y-%m-%d %H-%M-%S")
path_experiment = "Exp_{}/".format(current_time)

def f(job):        
    print("Job: {}".format(job))
    arg = ["python3", collection_file, "-j", str(job), "-p", path_experiment]
    val = subprocess.run(arg, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    my_output = (val.stderr.decode("utf-8"))
    
    return my_output

if __name__ == '__main__':
    with Pool() as p:
        outputs = p.map(f, list(range(n_jobs)))
        for output in outputs:
            print(output)
            print()
