import multiprocessing


def run_in_procs(func, func_args, processes, maxtasksperchild=1):

    def initializer():
        print('Starting', multiprocessing.current_process().name)

    try:
        pool = multiprocessing.Pool(processes=processes,
                                    maxtasksperchild=maxtasksperchild,
                                    initializer=initializer)
        res = pool.map(func, func_args)
    finally:
        pool.close()
        pool.join()
        pool.terminate()
    return res
