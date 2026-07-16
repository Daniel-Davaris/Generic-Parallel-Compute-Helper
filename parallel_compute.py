import concurrent.futures
import os
import sys
from tqdm import tqdm

_WORKER_FUNCTION = None
_WORKER_DATA = None
_WORKER_STDOUT = None
_WORKER_STDERR = None


### ONLY USED BY MULTIPROCESSING ### 
# Prepare each worker so per-item calls are fast and consistent
def _init_process_worker(function, data):
    global _WORKER_FUNCTION, _WORKER_DATA, _WORKER_STDOUT, _WORKER_STDERR

    # Set the target function in worker global variable
    _WORKER_FUNCTION = function
    _WORKER_DATA = data

    # Ensure processes only use one thread
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    # Silence output from both Python streams and low-level fd writes.
    _WORKER_STDOUT = open(os.devnull, "w")
    _WORKER_STDERR = open(os.devnull, "w")
    os.dup2(_WORKER_STDOUT.fileno(), 1)
    os.dup2(_WORKER_STDERR.fileno(), 2)
    sys.stdout = _WORKER_STDOUT
    sys.stderr = _WORKER_STDERR

### ONLY USED BY MULTIPROCESSING ### 
# Entry point called by each worker process — delegates to the globally stored function.
def _run_process_worker_item(item_index):
    return _WORKER_FUNCTION(_WORKER_DATA[item_index])


# Runs function over every item in data. 
def run_parallel(function, data):
    data = list(data)
    if not data:
        return []

    ## helpers ##
    def starting_message(max_workers):
        print(f"os={sys.platform} cpu_count={os.cpu_count()} max_assigned_workers={max_workers}")
    
     # Windows logic
    def windows():
        if sys.platform == "win32":
            starting_message(max_workers=1)
            results = []
            for item in tqdm(data, desc="Processing..", unit="item", dynamic_ncols=True, position=0, leave=True):
                results.append(function(item))
            return results
    windows_results = windows()
    if windows_results is not None:
        return windows_results

    # Linux logic
    def linux():

        def calculate_number_of_workers():
            cpu_workers = 45
            number_of_workers = max(1, min(cpu_workers, len(data)))

            return number_of_workers
        
        number_of_workers = calculate_number_of_workers()

        starting_message(max_workers=number_of_workers)

        # Create mutliple processes
        with concurrent.futures.ProcessPoolExecutor(
            max_workers=number_of_workers,
            initializer=_init_process_worker,
            initargs=(function, data),
        ) as executor:
            
            futures = {
                executor.submit(_run_process_worker_item, i): i for i in range(len(data))
            }

            results = [None] * len(data)
            progress = tqdm(total=len(data), desc="Processing..", unit="item", dynamic_ncols=True, position=0, leave=True)
            try:
                for future in concurrent.futures.as_completed(futures):
                    results[futures[future]] = future.result()
                    progress.update(1)
            finally:
                progress.close()

            return results
    return linux()