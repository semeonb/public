import multiprocessing


def run_in_procs(func, func_args, processes, maxtasksperchild=1):
    # Create a pool of worker processes
    pool = multiprocessing.Pool(processes=processes, maxtasksperchild=maxtasksperchild)

    try:
        # Use the map method to apply the function to each argument
        res = pool.map(func, func_args)
    except Exception as e:
        # Handle any exceptions that may occur during execution
        print("Error:", e)
    finally:
        # Close, join and terminate the pool
        pool.close()
        pool.join()
        pool.terminate()
    return res
